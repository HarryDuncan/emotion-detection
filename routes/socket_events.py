"""
Socket.IO real-time emotion events.

Namespace:  /emotion

Server → client events
----------------------
dominant_emotion_color
    Emitted whenever the dominant emotion changes.
    Payload:
        face_detected    bool
        dominant_emotion str | null   — e.g. "happy"
        color_hex        str | null   — e.g. "#FFD700"
        color_rgb        list[int]    — [r, g, b]  0-255
        color_bgr        list[int]    — [b, g, r]  0-255 (OpenCV order)
        face_count       int          — total faces in frame

Lifecycle
---------
Connecting to the /emotion namespace activates the emotion inference loop.
Disconnecting idles it when no other clients remain and /start_detection
has not been called.

Client example (JavaScript)
----------------------------
    const socket = io('/emotion');
    socket.on('dominant_emotion_color', data => {
        console.log(data.dominant_emotion, data.color_hex);
    });
"""
import threading

import state as _state
from sio import socketio
from flask import request
from flask_socketio import emit
from routes.registry import FieldSpec, define

# sid → True for every client currently subscribed to model_output streaming
_model_output_sids: set = set()
_model_output_lock  = threading.Lock()


# ---------------------------------------------------------------------------
# Background broadcaster
# ---------------------------------------------------------------------------

def start_broadcaster():
    """Start the single shared broadcaster thread.  Called once at app init."""
    socketio.start_background_task(_emotion_broadcaster)


def _emotion_broadcaster():
    """
    Waits for the inference loop to signal emotion_condition, then emits
    the latest dominant-emotion color to all /emotion namespace clients.

    Only emits when the dominant emotion or face count actually changes to
    avoid flooding clients with identical events.
    """
    last_key = None
    while True:
        # Block until the inference loop produces a new result (or 1 s timeout
        # so the thread doesn't hang forever if inference never runs).
        with _state.emotion_condition:
            _state.emotion_condition.wait(timeout=1.0)

        with _state.emotion_result_lock:
            result = dict(_state.latest_emotion_result)

        faces = result.get('faces', [])

        if not result.get('face_detected') or not faces:
            payload = {
                'face_detected':    False,
                'dominant_emotion': None,
                'color_hex':        None,
                'color_rgb':        None,
                'color_bgr':        None,
                'face_count':       0,
            }
        else:
            face    = faces[0]
            emotion = face.get('dominant_emotion', '')
            bgr     = face.get('emotion_color_bgr', (128, 128, 128))
            b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            payload = {
                'face_detected':    True,
                'dominant_emotion': emotion,
                'color_hex':        f'#{r:02X}{g:02X}{b:02X}',
                'color_rgb':        [r, g, b],
                'color_bgr':        [b, g, r],
                'face_count':       len(faces),
            }

        # Deduplicate — only emit when something meaningful changed.
        key = (payload['dominant_emotion'], payload['face_count'])
        if key != last_key:
            socketio.emit('dominant_emotion_color', payload, namespace='/emotion')
            last_key = key


# ---------------------------------------------------------------------------
# Namespace lifecycle
# ---------------------------------------------------------------------------

@socketio.on('connect', namespace='/emotion')
def on_emotion_connect():
    with _state.emotion_client_lock:
        _state.emotion_active_clients += 1
    print(f"[socketio/emotion] connect     active_clients={_state.emotion_active_clients}")


@socketio.on('disconnect', namespace='/emotion')
def on_emotion_disconnect():
    with _state.emotion_client_lock:
        _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
    print(f"[socketio/emotion] disconnect  active_clients={_state.emotion_active_clients}")

    if _state.emotion_active_clients == 0 and not _state.emotion_explicitly_enabled:
        with _state.emotion_result_lock:
            _state.latest_emotion_result.clear()
            _state.latest_emotion_result.update({'face_detected': False, 'faces': []})


# ---------------------------------------------------------------------------
# Generic model output / stream events (default namespace)
# ---------------------------------------------------------------------------

define(
    name        = 'start_model_output',
    path        = 'start_model_output',
    methods     = ['SOCKET'],
    description = (
        'Begin streaming dominant-emotion results for the first detected face. '
        'Server emits model_output events on every new inference result until '
        'the client disconnects or emits stop_model_output.'
    ),
    factory     = True,
    output      = {
        'face_detected':    FieldSpec('boolean', 'True when at least one face is visible'),
        'dominant_emotion': FieldSpec('string',  'Dominant emotion of the first face', nullable=True,
                                     enum=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']),
    },
)

@socketio.on('start_model_output')
def handle_start_model_output(data):
    sid = request.sid
    with _model_output_lock:
        if sid in _model_output_sids:
            return  # already streaming for this client
        _model_output_sids.add(sid)

    with _state.emotion_client_lock:
        _state.emotion_active_clients += 1
    print(f"[model_output] stream started   sid={sid}  active={_state.emotion_active_clients}")

    emit('model_output', {'face_detected': False, 'dominant_emotion': None, 'status': 'streaming'})
    socketio.start_background_task(_model_output_stream, sid)


def _model_output_stream(sid):
    """Per-client background task: emit dominant emotion on every new inference result."""
    while True:
        with _model_output_lock:
            if sid not in _model_output_sids:
                break

        with _state.emotion_condition:
            _state.emotion_condition.wait(timeout=1.0)

        with _model_output_lock:
            if sid not in _model_output_sids:
                break

        with _state.emotion_result_lock:
            result = dict(_state.latest_emotion_result)

        faces = result.get('faces', [])
        dominant = faces[0].get('dominant_emotion') if faces else None

        socketio.emit('model_output', {
            'face_detected':    result.get('face_detected', False),
            'dominant_emotion': dominant,
        }, to=sid)

    print(f"[model_output] stream stopped   sid={sid}")


def _stop_model_output(sid):
    """Deactivate a client's model_output stream and decrement the inference counter."""
    with _model_output_lock:
        if sid not in _model_output_sids:
            return
        _model_output_sids.discard(sid)

    with _state.emotion_client_lock:
        _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
    print(f"[model_output] stream unregistered  sid={sid}  active={_state.emotion_active_clients}")


define(
    name        = 'stop_model_output',
    path        = 'stop_model_output',
    methods     = ['SOCKET'],
    description = 'Stop the model_output stream started by start_model_output.',
    factory     = True,
)

@socketio.on('stop_model_output')
def handle_stop_model_output(data):
    _stop_model_output(request.sid)


define(
    name        = 'start_model_stream',
    path        = 'start_model_stream',
    methods     = ['SOCKET'],
    description = 'Reserved for future use — continuous raw model output stream.',
    factory     = True,
)

@socketio.on('start_model_stream')
def handle_start_model_stream(data):
    emit('model_stream', {'status': 'streaming'})


@socketio.on('disconnect')
def handle_default_disconnect():
    """Clean up any active model_output stream when a default-namespace client disconnects."""
    _stop_model_output(request.sid)
