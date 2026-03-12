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
from routes.registry import define

bp = Blueprint('detection', __name__)

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

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
        'status':           'str — "enabled"',
        'active_clients':   'int — number of /video_dominant_emotion clients',
        'explicitly_enabled': 'bool',
    },
)

define(
    name        = 'stop_detection',
    path        = '/stop_detection',
    methods     = ['POST'],
    description = (
        'Explicitly disable background emotion inference. '
        'Has no effect while clients are connected to /video_dominant_emotion.'
    ),
    output      = {
        'status':           'str — "disabled" | "still_active" (clients still connected)',
        'active_clients':   'int',
        'explicitly_enabled': 'bool',
    },
)

define(
    name        = 'get_emotions',
    path        = '/get_emotions',
    methods     = ['GET'],
    description = (
        '(Stub) Returns current emotion data. Disabled in appv2 — '
        'use /video_dominant_emotion for live per-frame inference.'
    ),
    output      = {
        'emotions':        'dict — empty in appv2',
        'scores_out_of_10':'dict — empty in appv2',
        'timestamp':       'null',
        'message':         'str',
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
