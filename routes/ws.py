"""
Raw WebSocket endpoints — low-latency binary model output and annotated video.

Endpoints
---------
/ws?routes=start_model_output
    Binary model data frames at inference rate.
    Frame layout set by POST /set-config; GET /get-config returns current schema.
    Default: 24-byte little-endian frame (see schema below).

/ws/video
    Annotated JPEG frames at camera rate.
    Each WebSocket message is a raw JPEG image with bounding boxes and emotion
    labels drawn on it. Connect, set binaryType = 'arraybuffer', draw on canvas.

Default /ws wire format (24 bytes, little-endian — no set-config called)
-------------------------------------------------------------------------
    offset  0   int32   face_detected    0 or 1
    offset  4   int32   face_count       total faces in frame
    offset  8   int32   dominant_emotion angry=0 disgust=1 fear=2 happy=3
                                         sad=4 surprise=5 neutral=6 none=-1
    offset 12   float32 color_r          dominant-emotion colour 0.0–1.0
    offset 16   float32 color_g
    offset 20   float32 color_b

JavaScript /ws/video client example
-------------------------------------
    const ws = new WebSocket('ws://localhost:5005/ws/video');
    ws.binaryType = 'arraybuffer';
    ws.onmessage = async ({ data }) => {
        const bitmap = await createImageBitmap(new Blob([data], { type: 'image/jpeg' }));
        ctx.drawImage(bitmap, 0, 0);
    };
"""
import os

import cv2
from flask import request
from flask_sock import Sock

import state as _state
from frame_utils import annotate_frame, get_background_remover
from output_registry import DEFAULT_SCHEMA, DEFAULT_SPECS, run_extractors
from routes.registry import FieldSpec, define

_VIDEO_JPEG_QUALITY = int(os.environ.get('STREAM_JPEG_QUALITY', 70))
_jpeg_params        = [cv2.IMWRITE_JPEG_QUALITY, _VIDEO_JPEG_QUALITY]

sock = Sock()


define(
    name        = 'ws',
    path        = '/ws',
    methods     = ['WEBSOCKET'],
    description = (
        'Raw WebSocket endpoint for low-latency binary model output. '
        'Connect with ?routes=start_model_output to begin receiving binary frames. '
        'Frame layout is set by POST /set-config (GET /get-config returns current schema). '
        'Disconnect to stop.'
    ),
    factory     = True,
    output      = {
        'binary_frame': FieldSpec('string', 'Little-endian binary frame — layout defined by active output config (see GET /get-config)'),
    },
)

define(
    name        = 'ws_video',
    path        = '/ws/video',
    methods     = ['WEBSOCKET'],
    description = (
        'Raw WebSocket endpoint for annotated MJPEG-style video. '
        'Each message is a raw JPEG binary with emotion bounding boxes and labels drawn. '
        'Shares inference results with /ws — no duplicate face detection. '
        'Frontend: set ws.binaryType = "arraybuffer", decode with createImageBitmap.'
    ),
    factory     = True,
    output      = {
        'jpeg_frame': FieldSpec('binary', 'Raw JPEG bytes — annotated frame with emotion bounding boxes and labels'),
    },
)


@sock.route('/ws')
def ws_endpoint(ws):
    routes = request.args.get('routes', '')

    if 'start_model_output' in routes:
        _stream(ws)
        return

    # No query param — wait for a text command (30 s timeout).
    try:
        msg = ws.receive(timeout=30)
    except Exception:
        return

    if msg and msg.strip() == 'start_model_output':
        _stream(ws)


def _active_schema():
    """Return the compiled (schema, specs) from state, falling back to defaults."""
    with _state.output_config_lock:
        schema = _state.compiled_schema
        specs  = _state.compiled_specs
    if schema is None or not specs:
        return DEFAULT_SCHEMA, DEFAULT_SPECS
    return schema, specs


def _stream(ws):
    """Activate inference, send binary frames until the client disconnects."""
    with _state.emotion_client_lock:
        _state.emotion_active_clients += 1
    print(f"[ws] stream started   active={_state.emotion_active_clients}")

    # Send one zeroed frame immediately so the client knows streaming has started.
    schema, specs = _active_schema()
    try:
        ws.send(schema.pack(run_extractors({}, specs)))
    except Exception:
        pass

    try:
        while True:
            with _state.emotion_condition:
                _state.emotion_condition.wait(timeout=1.0)

            # Reread schema on every frame — picks up set-config changes instantly.
            schema, specs = _active_schema()

            with _state.emotion_result_lock:
                result = dict(_state.latest_emotion_result)

            flat = run_extractors(result, specs)
            ws.send(schema.pack(flat))

    except Exception as e:
        print(f"[ws] stream error: {type(e).__name__}: {e}")
    finally:
        with _state.emotion_client_lock:
            _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
        print(f"[ws] stream stopped   active={_state.emotion_active_clients}")


# ---------------------------------------------------------------------------
# /ws/video — annotated JPEG frame stream
# ---------------------------------------------------------------------------

@sock.route('/ws/video')
def ws_video_endpoint(ws):
    _video_stream(ws)


def _video_stream(ws):
    """Activate inference, send annotated JPEG frames until the client disconnects."""
    with _state.emotion_client_lock:
        _state.emotion_active_clients += 1
    print(f"[ws/video] stream started   active={_state.emotion_active_clients}")

    last_seq = -1
    try:
        while True:
            # Wait for the next camera frame (fires at camera rate, ~15–30 fps).
            with _state.frame_condition:
                _state.frame_condition.wait_for(
                    lambda: _state.frame_seq != last_seq, timeout=1.0
                )
                frame    = (_state.latest_frame_flipped.copy()
                            if _state.latest_frame_flipped is not None else None)
                last_seq = _state.frame_seq

            if frame is None:
                continue

            # Remove background — initialises MediaPipe on first call (lazy).
            frame = get_background_remover().remove(frame)

            # Read latest emotion result — already computed by the inference loop.
            with _state.emotion_result_lock:
                result = dict(_state.latest_emotion_result)

            annotate_frame(frame, result)

            ret, buf = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                ws.send(buf.tobytes())

    except Exception as e:
        print(f"[ws/video] stream error: {type(e).__name__}: {e}")
    finally:
        with _state.emotion_client_lock:
            _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
        print(f"[ws/video] stream stopped   active={_state.emotion_active_clients}")
