"""
Emotion detection control routes (stubs in appv2).

Endpoints
---------
POST /start_detection  — stub
POST /stop_detection   — stub
GET  /get_emotions     — stub
"""
from flask import Blueprint, jsonify

from routes.registry import define

bp = Blueprint('detection', __name__)

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

define(
    name        = 'start_detection',
    path        = '/start_detection',
    methods     = ['POST'],
    description = '(Stub) Start emotion detection session. Disabled in appv2.',
    output      = {
        'status':  'str — "disabled"',
        'message': 'str',
    },
)

define(
    name        = 'stop_detection',
    path        = '/stop_detection',
    methods     = ['POST'],
    description = '(Stub) Stop emotion detection session. Disabled in appv2.',
    output      = {
        'status':  'str — "disabled"',
        'message': 'str',
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
    return jsonify({
        'status':  'disabled',
        'message': 'Emotion detection is disabled in appv2. Use app.py for emotion detection.',
    })


@bp.route('/stop_detection', methods=['POST'])
def stop_detection():
    return jsonify({
        'status':  'disabled',
        'message': 'Emotion detection is disabled in appv2.',
    })


@bp.route('/get_emotions', methods=['GET'])
def get_emotions():
    return jsonify({
        'emotions':         {},
        'scores_out_of_10': {},
        'timestamp':        None,
        'message':          'Emotion detection disabled in appv2.',
    })
