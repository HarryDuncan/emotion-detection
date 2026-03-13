"""
Emotion detection control routes (stubs in appv2).

Endpoints
---------
POST /start_detection  — stub
POST /stop_detection   — stub
GET  /get_emotions     — stub
"""
from flask import Blueprint, jsonify

import state as _state
from routes.registry import FieldSpec, define

bp = Blueprint('detection', __name__)

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

_detection_state_output = {
    'status':             FieldSpec('string', 'Current inference state', enum=['enabled', 'disabled', 'still_active']),
    'active_clients':     FieldSpec('integer', 'Clients currently streaming /video_dominant_emotion', example=0),
    'explicitly_enabled': FieldSpec('boolean', 'Whether /start_detection is in effect'),
}

define(
    name        = 'start_detection',
    path        = '/start_detection',
    methods     = ['POST'],
    description = (
        'Explicitly enable background emotion inference. '
        'Inference also activates automatically when a client connects to '
        '/video_dominant_emotion and stops when they disconnect.'
    ),
    output      = {
        **_detection_state_output,
        'status': FieldSpec('string', enum=['enabled']),
    },
)

define(
    name        = 'stop_detection',
    path        = '/stop_detection',
    methods     = ['POST'],
    description = (
        'Explicitly disable background emotion inference. '
        'Returns still_active if /video_dominant_emotion clients remain connected.'
    ),
    output      = {
        **_detection_state_output,
        'status': FieldSpec('string', enum=['disabled', 'still_active']),
    },
)

define(
    name        = 'get_emotions',
    path        = '/get_emotions',
    methods     = ['GET'],
    description = (
        '(Stub) Not implemented in appv2 — '
        'use /video_dominant_emotion for live annotated stream '
        'or /get_dominant_emotion_color for the current snapshot.'
    ),
    output      = {
        'emotions':         FieldSpec('object', 'Always empty in appv2', example={}),
        'scores_out_of_10': FieldSpec('object', 'Always empty in appv2', example={}),
        'timestamp':        FieldSpec('null',   'Always null in appv2'),
        'message':          FieldSpec('string', example='Emotion detection disabled in appv2.'),
    },
)

_emotions_enum = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

define(
    name        = 'get_dominant_emotion_color',
    path        = '/get_dominant_emotion_color',
    methods     = ['GET'],
    description = (
        'Returns the dominant emotion and its associated color for the most '
        'prominent detected face. Color is provided as hex, RGB, and BGR. '
        'Returns face_detected=false when no face is visible or inference is not running. '
        'Start inference first with POST /start_detection or by opening /video_dominant_emotion.'
    ),
    output      = {
        'face_detected':    FieldSpec('boolean', 'False when no face is visible or inference is idle'),
        'dominant_emotion': FieldSpec('string',  'Most confident emotion for the largest face', nullable=True, enum=_emotions_enum),
        'color_hex':        FieldSpec('string',  'CSS hex color matching the dominant emotion', nullable=True, example='#FFD700'),
        'color_rgb':        FieldSpec('array',   '[r, g, b] each 0-255', nullable=True, items=FieldSpec('integer'), example=[255, 215, 0]),
        'color_bgr':        FieldSpec('array',   '[b, g, r] each 0-255 — OpenCV channel order', nullable=True, items=FieldSpec('integer'), example=[0, 215, 255]),
        'face_count':       FieldSpec('integer', 'Total faces currently detected in the frame', example=1),
    },
)

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@bp.route('/start_detection', methods=['POST'])
def start_detection():
    _state.emotion_explicitly_enabled = True
    return jsonify({
        'status':             'enabled',
        'active_clients':     _state.emotion_active_clients,
        'explicitly_enabled': _state.emotion_explicitly_enabled,
    })


@bp.route('/stop_detection', methods=['POST'])
def stop_detection():
    _state.emotion_explicitly_enabled = False
    still_active = _state.emotion_active_clients > 0
    return jsonify({
        'status':             'still_active' if still_active else 'disabled',
        'active_clients':     _state.emotion_active_clients,
        'explicitly_enabled': _state.emotion_explicitly_enabled,
    })


@bp.route('/get_emotions', methods=['GET'])
def get_emotions():
    return jsonify({
        'emotions':         {},
        'scores_out_of_10': {},
        'timestamp':        None,
        'message':          'Emotion detection disabled in appv2.',
    })


@bp.route('/get_dominant_emotion_color', methods=['GET'])
def get_dominant_emotion_color():
    with _state.emotion_result_lock:
        result = dict(_state.latest_emotion_result)

    faces = result.get('faces', [])
    if not result.get('face_detected') or not faces:
        return jsonify({
            'face_detected':    False,
            'dominant_emotion': None,
            'color_hex':        None,
            'color_rgb':        None,
            'color_bgr':        None,
            'face_count':       0,
        })

    # Use the first face (largest, as RetinaFace returns them sorted by size).
    face      = faces[0]
    emotion   = face.get('dominant_emotion', '')
    bgr       = face.get('emotion_color_bgr', (128, 128, 128))
    b, g, r   = int(bgr[0]), int(bgr[1]), int(bgr[2])

    return jsonify({
        'face_detected':    True,
        'dominant_emotion': emotion,
        'color_hex':        f'#{r:02X}{g:02X}{b:02X}',
        'color_rgb':        [r, g, b],
        'color_bgr':        [b, g, r],
        'face_count':       len(faces),
    })
