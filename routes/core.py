"""
Core routes — system health and status.

Endpoints
---------
GET /          — main web UI
GET /status    — raw initialisation status object
GET /health    — health check (200 ok / 503 degraded)
"""
from flask import Blueprint, jsonify, render_template

import state as _state
from routes.registry import FieldSpec, define

bp = Blueprint('core', __name__)

# ---------------------------------------------------------------------------
# Route definitions (registered into REGISTRY at import time)
# ---------------------------------------------------------------------------

define(
    name        = 'index',
    path        = '/',
    methods     = ['GET'],
    description = 'Serves the main web UI.',
    factory     = True,
    output      = {
        'body': FieldSpec('string', 'text/html — rendered index.html template'),
    },
)

_init_status_properties = {
    'camera_ready':            FieldSpec('boolean', 'Camera device opened successfully'),
    'tensorflow_ready':        FieldSpec('boolean', 'TensorFlow imported without error'),
    'tensorflow_gpu':          FieldSpec('boolean', 'At least one GPU visible to TensorFlow'),
    'tensorflow_gpu_devices':  FieldSpec('array',   'GPU device names', items=FieldSpec('string'), example=['GPU:0']),
    'emotion_models_loaded':   FieldSpec('boolean', 'DeepFace emotion models loaded into memory'),
    'initializing':            FieldSpec('boolean', 'True while startup is still in progress'),
}

define(
    name        = 'status',
    path        = '/status',
    methods     = ['GET'],
    description = 'Raw initialisation status for every subsystem plus live inference state.',
    factory     = True,
    output      = {
        **_init_status_properties,
        'emotion_active_clients':     FieldSpec('integer', 'Clients currently streaming /video_dominant_emotion', example=0),
        'emotion_explicitly_enabled': FieldSpec('boolean', 'True after POST /start_detection, false after POST /stop_detection'),
        'emotion_running':            FieldSpec('boolean', 'True if inference is actively executing (active_clients > 0 OR explicitly_enabled)'),
    },
)

define(
    name        = 'health',
    path        = '/health',
    methods     = ['GET'],
    description = (
        'Health check endpoint. '
        'Returns HTTP 200 when all subsystems are ready, HTTP 503 otherwise. '
        'Suitable for Docker HEALTHCHECK, load-balancer probes, and monitoring.'
    ),
    factory     = True,
    output      = {
        'healthy': FieldSpec('boolean', 'True when fully initialised and all subsystems ready'),
        'message': FieldSpec('string',  'Human-readable status summary', enum=['ok', 'initializing', 'degraded']),
        'status':  FieldSpec('object',  'Full initialization_status snapshot', properties=_init_status_properties),
    },
)

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/status', methods=['GET'])
def status():
    return jsonify({
        **_state.initialization_status,
        'emotion_active_clients':    _state.emotion_active_clients,
        'emotion_explicitly_enabled': _state.emotion_explicitly_enabled,
        'emotion_running':           (
            _state.emotion_active_clients > 0
            or _state.emotion_explicitly_enabled
        ),
    })


@bp.route('/health', methods=['GET'])
def health():
    s = _state.initialization_status
    ready = (
        not s.get('initializing', True)
        and s.get('emotion_models_loaded', False)
    )

    if ready:
        message = 'ok'
    elif s.get('initializing', True):
        message = 'initializing'
    else:
        missing = [
            k for k in ('camera_ready', 'tensorflow_ready', 'emotion_models_loaded')
            if not s.get(k)
        ]
        message = 'degraded: ' + ', '.join(missing)

    return jsonify({'healthy': ready, 'message': message, 'status': s}), (200 if ready else 503)
