"""
Raw WebSocket endpoint — low-latency binary model output.

Connect
-------
    ws://host:5005/ws?routes=start_model_output

The server immediately begins sending binary frames on every inference result.
The frame layout is determined by the active output config (POST /set-config).
Disconnect to stop.

You can also connect without the query param and send a text message:
    start_model_output   — begin streaming
    stop_model_output    — stop streaming (server closes)

Default wire format (24 bytes, little-endian — no set-config called)
---------------------------------------------------------------------
    offset  0   int32   face_detected    0 or 1
    offset  4   int32   face_count       total faces in frame
    offset  8   int32   dominant_emotion angry=0 disgust=1 fear=2 happy=3
                                         sad=4 surprise=5 neutral=6 none=-1
    offset 12   float32 color_r          dominant-emotion colour 0.0–1.0
    offset 16   float32 color_g
    offset 20   float32 color_b

Call GET /get-config to retrieve the current schema so your client can parse
the frames correctly after any set-config call.

JavaScript client example
--------------------------
    const ws = new WebSocket('ws://localhost:5005/ws?routes=start_model_output');
    ws.binaryType = 'arraybuffer';
    ws.onmessage = ({ data }) => {
        // parse according to GET /get-config schema
        const v = new DataView(data);
        const face_detected    = v.getInt32(0,  true);
        const face_count       = v.getInt32(4,  true);
        const dominant_emotion = v.getInt32(8,  true);  // -1 = no face
        const r                = v.getFloat32(12, true);
        const g                = v.getFloat32(16, true);
        const b                = v.getFloat32(20, true);
    };
"""
from flask import request
from flask_sock import Sock

import state as _state
from output_registry import DEFAULT_SCHEMA, DEFAULT_SPECS, run_extractors
from routes.registry import FieldSpec, define

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

    except Exception:
        # Client disconnected or send error — exit cleanly.
        pass
    finally:
        with _state.emotion_client_lock:
            _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
        print(f"[ws] stream stopped   active={_state.emotion_active_clients}")
