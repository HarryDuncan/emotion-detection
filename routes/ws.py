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
import json
import os
import threading
import time

import cv2
import requests as _requests
from flask import request
from flask_sock import Sock

import state as _state
from frame_utils import (annotate_data_layer, annotate_frame,
                         colorize_frame, get_background_remover)
from output_registry import DEFAULT_SCHEMA, DEFAULT_SPECS, run_extractors
from routes.ollama import parse_modelfile
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

            # Determine dominant-emotion colour for this frame.
            with _state.emotion_result_lock:
                result = _state.latest_emotion_result
                faces  = result.get('faces', [])
                if faces:
                    bg_bgr = faces[0].get('emotion_color_bgr', (0, 0, 0))
                else:
                    bg_bgr = (0, 0, 0)

            frame = get_background_remover().remove(frame, bg_color=bg_bgr)
            if faces:
                frame = colorize_frame(frame, bg_bgr, strength=0.25)

            ret, buf = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                ws.send(buf.tobytes())

    except Exception as e:
        print(f"[ws/video] stream error: {type(e).__name__}: {e}")
    finally:
        with _state.emotion_client_lock:
            _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
        print(f"[ws/video] stream stopped   active={_state.emotion_active_clients}")


# ---------------------------------------------------------------------------
# Ollama helpers — used by the data-layer stream
# ---------------------------------------------------------------------------

_MODELFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Modelfile',
)
_OLLAMA_PROMPT_INTERVAL = 30.0
_TYPEWRITER_CHARS_PER_FRAME = 2


def _init_ollama() -> None:
    """Connect to Ollama and create the model (blocking). Run in a bg thread."""
    base_url = (
        _state.ollama_base_url
        or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    ).rstrip('/')

    try:
        resp = _requests.get(f'{base_url}/api/version', timeout=5)
        resp.raise_for_status()
        _state.ollama_connected = True
        _state.ollama_base_url  = base_url
        print(f'[ollama] Connected to {base_url} (v{resp.json().get("version")})')
    except Exception as exc:
        print(f'[ollama] Connection failed: {exc}')
        return

    if _state.ollama_model_ready:
        return

    model_name = os.environ.get('OLLAMA_MODEL_NAME', 'emotion-ai')
    try:
        from_model, system, params = parse_modelfile(_MODELFILE_PATH)
        payload = {
            'model': model_name,
            'from': from_model,
            'system': system,
            'parameters': params,
            'stream': False,
        }
        print(f'[ollama] Creating model "{model_name}" from "{from_model}"')
        print(f'[ollama] Create payload: {json.dumps(payload, indent=2)[:500]}')
        resp = _requests.post(
            f'{base_url}/api/create', json=payload, timeout=600,
        )
        if resp.status_code != 200:
            print(f'[ollama] Create response {resp.status_code}: {resp.text[:500]}')
        resp.raise_for_status()
        _state.ollama_model_name  = model_name
        _state.ollama_model_ready = True
        print(f'[ollama] Model "{model_name}" ready')
    except Exception as exc:
        print(f'[ollama] Model creation failed: {exc}')


def _build_emotion_prompt(result: dict) -> str:
    """Format the current emotion result into a concise prompt."""
    faces = result.get('faces', [])
    if not faces:
        return 'No subjects are currently detected. The room appears empty.'
    lines = [f'Number of subjects: {len(faces)}']
    for i, face in enumerate(faces, 1):
        emotions = face.get('emotions', {})
        dominant = face.get('dominant_emotion', 'unknown')
        top3 = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]
        breakdown = ', '.join(f'{n} ({s:.0f}%)' for n, s in top3)
        lines.append(
            f'Subject {i}: dominant emotion is {dominant}. '
            f'Breakdown: {breakdown}'
        )
    return '\n'.join(lines)


def _prompt_ollama_bg(result: dict) -> None:
    """Send emotion data to Ollama in a background thread; update state."""
    if not _state.ollama_model_ready:
        return

    def _do():
        prompt = _build_emotion_prompt(result)
        try:
            resp = _requests.post(
                f'{_state.ollama_base_url}/api/chat',
                json={
                    'model': _state.ollama_model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'stream': False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json().get('message', {}).get('content', '')

            # Strip markdown code fences the LLM sometimes wraps JSON in
            clean = content.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[-1]
            if clean.endswith('```'):
                clean = clean.rsplit('```', 1)[0]
            clean = clean.strip()

            narrative = clean
            try:
                parsed = json.loads(clean)
                # Try common key names the LLM might use
                for key in ('narrative_text', 'narrative', 'analysis',
                            'description', 'narrative_description'):
                    if key in parsed:
                        narrative = parsed[key]
                        break
                else:
                    first_str = next(
                        (v for v in parsed.values() if isinstance(v, str)),
                        clean,
                    )
                    narrative = first_str
            except (json.JSONDecodeError, ValueError):
                pass

            with _state.ollama_llm_lock:
                _state.ollama_llm_text = narrative
            print(f'[ollama] Narrative ({len(narrative)} chars): '
                  f'{narrative[:120]}…' if len(narrative) > 120 else
                  f'[ollama] Narrative ({len(narrative)} chars): {narrative}')
        except Exception as exc:
            print(f'[ollama] Prompt failed: {exc}')

    threading.Thread(target=_do, daemon=True, name='ollama-prompt').start()


# ---------------------------------------------------------------------------
# /ws/data_layer — annotation-only transparent PNG stream
# ---------------------------------------------------------------------------

define(
    name        = 'ws_data_layer',
    path        = '/ws/data_layer',
    methods     = ['WEBSOCKET'],
    description = (
        'WebSocket stream of transparent RGBA PNG frames containing only the '
        'annotation layer (bounding boxes and emotion labels). No camera image '
        'is included. Composite over /ws/video or /video_feed on a canvas element. '
        'Frontend: set ws.binaryType = "arraybuffer", decode with createImageBitmap.'
    ),
    factory     = True,
    output      = {
        'png_frame': FieldSpec('binary', 'Raw RGBA PNG bytes — transparent frame with annotation shapes only'),
    },
)

@sock.route('/ws/data_layer')
def ws_data_layer_endpoint(ws):
    _data_layer_stream(ws)


def _data_layer_stream(ws):
    """Activate inference, send transparent annotation PNG frames until the client disconnects."""
    with _state.emotion_client_lock:
        _state.emotion_active_clients += 1
    print(f"[ws/data_layer] stream started   active={_state.emotion_active_clients}")

    # Kick off Ollama connection + model creation in the background.
    threading.Thread(target=_init_ollama, daemon=True,
                     name='ollama-init').start()

    last_seq        = -1
    start_time      = time.monotonic()
    last_prompt_at  = 0.0

    # Typewriter state
    current_full    = ''
    reveal_idx      = 0
    _logged_first   = False

    # Session statistics — tracked per stream connection
    prev_dominant: str | None = None
    emotion_counts: dict[str, int] = {}   # times each emotion became dominant
    change_count   = 0

    try:
        while True:
            with _state.frame_condition:
                _state.frame_condition.wait_for(
                    lambda: _state.frame_seq != last_seq, timeout=1.0
                )
                frame    = _state.latest_frame_flipped
                last_seq = _state.frame_seq

            h, w = frame.shape[:2] if frame is not None else (480, 640)

            with _state.emotion_result_lock:
                result = dict(_state.latest_emotion_result)

            elapsed = time.monotonic() - start_time

            # --- update session statistics ---
            faces = result.get('faces', [])
            if faces:
                dominant = faces[0].get('dominant_emotion')
                if dominant and dominant != prev_dominant:
                    if prev_dominant is not None:
                        change_count += 1
                    emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
                    prev_dominant = dominant

            top3 = sorted(emotion_counts.items(),
                          key=lambda kv: kv[1], reverse=True)[:3]
            session_stats = {
                'top3':            top3,
                'change_count':    change_count,
                'emotion_counts':  emotion_counts,
            }

            # --- periodic Ollama prompt ---
            if (elapsed - last_prompt_at >= _OLLAMA_PROMPT_INTERVAL
                    and _state.ollama_model_ready):
                last_prompt_at = elapsed
                _prompt_ollama_bg(result)

            # --- typewriter animation ---
            with _state.ollama_llm_lock:
                full_text = _state.ollama_llm_text
            if full_text != current_full:
                current_full = full_text
                reveal_idx   = 0
                if not _logged_first:
                    print(f'[ws/data_layer] First LLM text arrived '
                          f'({len(current_full)} chars)')
                    _logged_first = True
            reveal_idx  = min(reveal_idx + _TYPEWRITER_CHARS_PER_FRAME,
                              len(current_full))
            visible     = current_full[:reveal_idx]

            canvas = annotate_data_layer(result, h, w,
                                         elapsed_s=elapsed,
                                         llm_text=visible,
                                         stats=session_stats)

            ret, buf = cv2.imencode('.png', canvas)
            if ret:
                ws.send(buf.tobytes())

    except Exception as e:
        print(f"[ws/data_layer] stream error: {type(e).__name__}: {e}")
    finally:
        with _state.emotion_client_lock:
            _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
        print(f"[ws/data_layer] stream stopped   active={_state.emotion_active_clients}")
