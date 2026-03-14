"""
Raw WebSocket endpoint — low-latency binary model output.

Connect
-------
    ws://host:5005/ws?routes=start_model_output

The server immediately begins sending 20-byte binary frames on every
inference result.  Disconnect to stop.

You can also connect without the query param and send a text message:
    start_model_output   — begin streaming
    stop_model_output    — stop streaming (server closes)

Wire format (little-endian, 20 bytes)
--------------------------------------
    offset  0   int32   face_detected    0 or 1
    offset  4   int32   emotion_id       angry=0 disgust=1 fear=2 happy=3
                                         sad=4 surprise=5 neutral=6 none=-1
    offset  8   float32 r                dominant-emotion colour 0.0–1.0
    offset 12   float32 g                dominant-emotion colour 0.0–1.0
    offset 16   float32 b                dominant-emotion colour 0.0–1.0

JavaScript client example
--------------------------
    const ws = new WebSocket('ws://localhost:5005/ws?routes=start_model_output');
    ws.binaryType = 'arraybuffer';
    ws.onmessage = ({ data }) => {
        const v = new DataView(data);
        const face_detected = v.getInt32(0,  true);
        const emotion_id    = v.getInt32(4,  true);  // -1 = no face
        const r             = v.getFloat32(8,  true);
        const g             = v.getFloat32(12, true);
        const b             = v.getFloat32(16, true);
    };
"""
from flask import request
from flask_sock import Sock

import state as _state
from binary_pack import EMOTION_OUTPUT_SCHEMA
from routes.registry import FieldSpec, define

sock = Sock()

_EMPTY_FRAME = EMOTION_OUTPUT_SCHEMA.pack({
    'face_detected':    False,
    'dominant_emotion': None,
    'color_rgb':        [0, 0, 0],
})

define(
    name        = 'ws',
    path        = '/ws',
    methods     = ['WEBSOCKET'],
    description = (
        'Raw WebSocket endpoint. '
        'Connect with ?routes=start_model_output to immediately receive '
        '20-byte little-endian binary frames on every inference result: '
        'int32 face_detected, int32 emotion_id, float32 r, float32 g, float32 b. '
        'emotion_id: angry=0 disgust=1 fear=2 happy=3 sad=4 surprise=5 neutral=6 none=-1. '
        'Disconnect to stop streaming.'
    ),
    factory     = True,
    output      = {
        'binary_frame': FieldSpec(
            'string',
            '20-byte little-endian struct: '
            '[int32 face_detected] [int32 emotion_id] [float32 r] [float32 g] [float32 b]',
        ),
    },
)


@sock.route('/ws')
def ws_endpoint(ws):
    routes = request.args.get('routes', '')

    if 'start_model_output' in routes:
        _stream(ws)
        return

    # No query param — wait for a text command message (30 s timeout).
    try:
        msg = ws.receive(timeout=30)
    except Exception:
        return

    if msg and msg.strip() == 'start_model_output':
        _stream(ws)


def _stream(ws):
    """Activate inference, send binary frames until the client disconnects."""
    with _state.emotion_client_lock:
        _state.emotion_active_clients += 1
    print(f"[ws] stream started   active={_state.emotion_active_clients}")

    try:
        ws.send(_EMPTY_FRAME)   # immediate zero frame so the client knows we started
        while True:
            with _state.emotion_condition:
                _state.emotion_condition.wait(timeout=1.0)

            with _state.emotion_result_lock:
                result = dict(_state.latest_emotion_result)

            faces = result.get('faces', [])
            face  = faces[0] if faces else {}

            ws.send(EMOTION_OUTPUT_SCHEMA.pack({
                'face_detected':    result.get('face_detected', False),
                'dominant_emotion': face.get('dominant_emotion'),
                'color_rgb':        list(face.get('emotion_color_rgb', (0, 0, 0))),
            }))

    except Exception:
        # Client disconnected or connection error — exit cleanly.
        pass
    finally:
        with _state.emotion_client_lock:
            _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
        print(f"[ws] stream stopped   active={_state.emotion_active_clients}")
