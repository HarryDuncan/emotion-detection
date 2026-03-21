"""
Ollama LLM routes — connection management, model listing, and model creation.

Endpoints
---------
POST /ollama/connect       — verify Ollama is reachable, store connection
GET  /ollama/tags          — list models available in Ollama
POST /ollama/create-model  — create a custom model from the project Modelfile
"""
import os
import re

import requests as _requests
from flask import Blueprint, jsonify, request

import state as _state
from routes.registry import FieldSpec, define

bp = Blueprint('ollama', __name__)

_MODELFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Modelfile',
)
_DEFAULT_MODEL_NAME = 'emotion-ai'


def parse_modelfile(path: str) -> tuple[str, str, dict]:
    """Parse an Ollama Modelfile into (from_model, system_prompt, parameters).

    Handles multi-line SYSTEM values delimited by triple-quotes.
    """
    with open(path) as f:
        content = f.read()

    from_model = ''
    system     = ''
    params: dict = {}

    # Extract SYSTEM (may span multiple lines with triple-quotes)
    m = re.search(r'SYSTEM\s+"""(.+?)"""', content, re.DOTALL)
    if m:
        system = m.group(1).strip()
    else:
        m = re.search(r'SYSTEM\s+"([^"]*)"', content)
        if m:
            system = m.group(1).strip()
        else:
            m = re.search(r'SYSTEM\s+(.+)', content)
            if m:
                system = m.group(1).strip()

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith('FROM '):
            from_model = stripped[5:].strip()
        elif stripped.upper().startswith('PARAMETER '):
            parts = stripped[10:].strip().split(None, 1)
            if len(parts) == 2:
                key, val = parts
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                params[key] = val

    return from_model, system, params

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

define(
    name        = 'ollama_connect',
    path        = '/ollama/connect',
    methods     = ['POST'],
    description = (
        'Test connectivity to the Ollama server and store the base URL. '
        'Optionally pass a custom base_url; defaults to OLLAMA_BASE_URL env '
        'var or http://localhost:11434.'
    ),
    input       = {
        'base_url': FieldSpec('string', 'Ollama server URL', nullable=True,
                              example='http://localhost:11434'),
    },
    output      = {
        'connected': FieldSpec('boolean', 'True if Ollama responded'),
        'base_url':  FieldSpec('string',  'URL that was tested'),
        'version':   FieldSpec('string',  'Ollama version string', nullable=True),
        'error':     FieldSpec('string',  'Error message on failure', nullable=True),
    },
)

define(
    name        = 'ollama_tags',
    path        = '/ollama/tags',
    methods     = ['GET'],
    description = 'List all models available in the connected Ollama instance.',
    output      = {
        'models': FieldSpec('array', 'Models available in Ollama',
                            items=FieldSpec('object')),
    },
)

define(
    name        = 'ollama_create_model',
    path        = '/ollama/create-model',
    methods     = ['POST'],
    description = (
        'Create (or recreate) a custom Ollama model from the project Modelfile. '
        'Pulls the base model automatically if not already present — this can '
        'take several minutes on first run. '
        'Optionally pass a custom model name; defaults to "emotion-ai".'
    ),
    input       = {
        'name': FieldSpec('string', 'Name for the created model', nullable=True,
                          example='emotion-ai'),
    },
    output      = {
        'model_ready': FieldSpec('boolean', 'True when model was created successfully'),
        'model_name':  FieldSpec('string',  'The Ollama model name'),
        'status':      FieldSpec('string',  'Final status from Ollama'),
        'error':       FieldSpec('string',  'Error message on failure', nullable=True),
    },
)

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@bp.route('/ollama/connect', methods=['POST'])
def ollama_connect():
    data     = request.get_json(silent=True) or {}
    base_url = (
        data.get('base_url')
        or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    ).rstrip('/')

    try:
        resp = _requests.get(f'{base_url}/api/version', timeout=5)
        resp.raise_for_status()
        version = resp.json().get('version')
    except Exception as exc:
        _state.ollama_connected = False
        _state.ollama_base_url  = ''
        return jsonify({
            'connected': False,
            'base_url':  base_url,
            'version':   None,
            'error':     str(exc),
        }), 502

    _state.ollama_connected = True
    _state.ollama_base_url  = base_url
    print(f'[ollama] Connected to {base_url} (v{version})')
    return jsonify({
        'connected': True,
        'base_url':  base_url,
        'version':   version,
        'error':     None,
    })


@bp.route('/ollama/tags', methods=['GET'])
def ollama_tags():
    if not _state.ollama_connected or not _state.ollama_base_url:
        return jsonify({
            'error': 'Ollama not connected — call POST /ollama/connect first.',
        }), 400

    try:
        resp = _requests.get(
            f'{_state.ollama_base_url}/api/tags', timeout=10,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 502


@bp.route('/ollama/create-model', methods=['POST'])
def ollama_create_model():
    if not _state.ollama_connected or not _state.ollama_base_url:
        return jsonify({
            'error': 'Ollama not connected — call POST /ollama/connect first.',
        }), 400

    data       = request.get_json(silent=True) or {}
    model_name = (
        data.get('name')
        or os.environ.get('OLLAMA_MODEL_NAME', _DEFAULT_MODEL_NAME)
    )

    try:
        from_model, system, params = parse_modelfile(_MODELFILE_PATH)
    except FileNotFoundError:
        return jsonify({
            'model_ready': False,
            'model_name':  model_name,
            'status':      'error',
            'error':       f'Modelfile not found at {_MODELFILE_PATH}',
        }), 404

    import json as _json
    payload = {
        'model': model_name,
        'from': from_model,
        'system': system,
        'parameters': params,
        'stream': False,
    }
    print(f'[ollama] Creating model "{model_name}" from "{from_model}"')
    print(f'[ollama] Create payload: {_json.dumps(payload, indent=2)[:500]}')
    try:
        resp = _requests.post(
            f'{_state.ollama_base_url}/api/create',
            json=payload, timeout=600,
        )
        if resp.status_code != 200:
            print(f'[ollama] Create response {resp.status_code}: {resp.text[:500]}')
        resp.raise_for_status()
        status_msg = resp.json().get('status', 'success')
    except Exception as exc:
        _state.ollama_model_ready = False
        print(f'[ollama] Create failed: {exc}')
        return jsonify({
            'model_ready': False,
            'model_name':  model_name,
            'status':      'error',
            'error':       str(exc),
        }), 502

    _state.ollama_model_name  = model_name
    _state.ollama_model_ready = True
    print(f'[ollama] Model "{model_name}" ready')
    return jsonify({
        'model_ready': True,
        'model_name':  model_name,
        'status':      status_msg,
        'error':       None,
    })
