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
from routes.registry import define

bp = Blueprint('core', __name__)

# ---------------------------------------------------------------------------
# Route definitions (registered into REGISTRY at import time)
# ---------------------------------------------------------------------------

define(
    name        = 'index',
    path        = '/',
    methods     = ['GET'],
    description = 'Serves the main web UI.',
    output      = {'html': 'text/html — index.html template'},
)

define(
    name        = 'status',
    path        = '/status',
    methods     = ['GET'],
    description = 'Raw initialisation status for every subsystem.',
    output      = {
        'camera_ready':           'bool',
        'tensorflow_ready':       'bool',
        'tensorflow_gpu':         'bool',
        'tensorflow_gpu_devices': 'list[str]',
        'emotion_models_loaded':  'bool',
        'initializing':           'bool',
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
    output      = {
        'healthy': 'bool — true when fully initialised and ready',
        'message': 'str — "ok" | "initializing" | "degraded: <subsystem list>"',
        'status':  'object — full initialization_status snapshot',
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
