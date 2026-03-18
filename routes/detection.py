"""
Detection control and output routes.

Endpoints
---------
POST /start_detection         — explicitly enable inference
POST /stop_detection          — explicitly disable inference
GET  /get_emotions            — all emotion scores for first face
GET  /get_dominant_emotion_color — dominant emotion + hex/rgb/bgr color
POST /connect-inputs          — (re)connect camera
POST /set-config              — configure /ws binary output fields
GET  /get-config              — current output config + schema description
"""
from flask import Blueprint, jsonify, request

import state as _state
from routes.registry import FieldSpec, define
from output_registry import (
    OUTPUT_REGISTRY,
    DEFAULT_CONFIG,
    compile_schema,
    run_extractors,
    schema_description,
    spec_description,
    extract_face_detected,
    extract_dominant_emotion,
    extract_dominant_emotion_color,
    extract_all_emotions,
    ID_TO_EMOTION,
)

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

_emotions_enum = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

define(
    name        = 'get_emotions',
    path        = '/get_emotions',
    methods     = ['GET'],
    description = 'Smoothed emotion scores (0–100) for the first detected face.',
    output      = {
        'face_detected': FieldSpec('boolean', 'True when a face is visible'),
        'face_count':    FieldSpec('integer', 'Total detected faces', example=1),
        'emotions':      FieldSpec('object',  'Map of emotion name → score 0–100'),
    },
)

define(
    name        = 'get_dominant_emotion_color',
    path        = '/get_dominant_emotion_color',
    methods     = ['GET'],
    description = (
        'Dominant emotion and its color for the most prominent detected face. '
        'Color in hex, RGB, and BGR. Returns face_detected=false when inference is idle.'
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

define(
    name        = 'connect_inputs',
    path        = '/connect-inputs',
    methods     = ['POST'],
    description = (
        'Attempt to (re)connect all hardware inputs (camera). '
        'Blocks for up to 5 s while GStreamer transitions to PLAYING — '
        'start the stream sender before calling.'
    ),
    factory     = True,
    output      = {
        'camera_ready': FieldSpec('boolean', 'True if the camera is open after this call'),
        'status':       FieldSpec('string',  'Outcome', enum=['already_connected', 'connected', 'failed']),
        'message':      FieldSpec('string',  'Human-readable detail'),
    },
)

_schema_field_props = {
    'name':   FieldSpec('string',  'Field name in the packed frame'),
    'type':   FieldSpec('string',  'C type', enum=['int32', 'float32', 'int16', 'int8']),
    'offset': FieldSpec('integer', 'Byte offset in the frame'),
    'size':   FieldSpec('integer', 'Byte size of this field'),
}

define(
    name        = 'set_config',
    path        = '/set-config',
    methods     = ['POST'],
    description = (
        'Configure which data fields are packed into each /ws binary frame. '
        'Returns the compiled schema so the client can build its DataView parser. '
        'Available outputs: ' + ', '.join(OUTPUT_REGISTRY.keys()) + '.'
    ),
    factory     = True,
    input       = {
        'outputs': FieldSpec('array', 'Ordered list of extractor names to include', items=FieldSpec('string'), example=DEFAULT_CONFIG),
    },
    output      = {
        'config':    FieldSpec('array',   'Active extractor list', items=FieldSpec('string')),
        'schema':    FieldSpec('object',  'Compiled binary schema description'),
        'available': FieldSpec('object',  'All available extractors: name → description'),
    },
)

define(
    name        = 'get_config',
    path        = '/get-config',
    methods     = ['GET'],
    description = 'Return the current /ws output config and its compiled binary schema.',
    factory     = True,
    output      = {
        'config':    FieldSpec('array',  'Active extractor list', items=FieldSpec('string')),
        'schema':    FieldSpec('object', 'Compiled binary schema description'),
        'available': FieldSpec('object', 'All available extractors: name → description'),
    },
)

define(
    name        = 'available_outputs',
    path        = '/available-outputs',
    methods     = ['GET'],
    description = (
        'List every available output extractor with its name, description, '
        'individual binary field definitions, and byte size. '
        'Use this to populate a frontend config UI before calling POST /set-config.'
    ),
    factory     = True,
    output      = {
        'outputs': FieldSpec(
            'array',
            'All extractors the server supports',
            items=FieldSpec('object', properties={
                'name':        FieldSpec('string',  'Extractor identifier — pass in set-config outputs list'),
                'description': FieldSpec('string',  'Human-readable summary'),
                'schema':      FieldSpec('object',  'Binary schema produced by this extractor alone — use to set up DataView parsing'),
            }),
        ),
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
    with _state.emotion_result_lock:
        result = dict(_state.latest_emotion_result)

    face_data    = extract_face_detected(result)
    emotion_data = extract_all_emotions(result)

    # Strip 'emotion_' prefix for the response keys
    scores = {k.replace('emotion_', ''): round(v, 2) for k, v in emotion_data.items()}
    return jsonify({
        'face_detected': bool(face_data['face_detected']),
        'face_count':    face_data['face_count'],
        'emotions':      scores,
    })


@bp.route('/get_dominant_emotion_color', methods=['GET'])
def get_dominant_emotion_color():
    with _state.emotion_result_lock:
        result = dict(_state.latest_emotion_result)

    face_data  = extract_face_detected(result)
    emot_data  = extract_dominant_emotion(result)
    color_data = extract_dominant_emotion_color(result)

    if not face_data['face_detected']:
        return jsonify({
            'face_detected':    False,
            'dominant_emotion': None,
            'color_hex':        None,
            'color_rgb':        None,
            'color_bgr':        None,
            'face_count':       0,
        })

    emotion_name = ID_TO_EMOTION.get(emot_data['dominant_emotion_id'], '')
    r = round(color_data['color_r'] * 255)
    g = round(color_data['color_g'] * 255)
    b = round(color_data['color_b'] * 255)

    return jsonify({
        'face_detected':    True,
        'dominant_emotion': emotion_name,
        'color_hex':        f'#{r:02X}{g:02X}{b:02X}',
        'color_rgb':        [r, g, b],
        'color_bgr':        [b, g, r],
        'face_count':       face_data['face_count'],
    })


@bp.route('/connect-inputs', methods=['POST'])
def connect_inputs():
    cam = _state.camera_input
    if cam is None:
        return jsonify({'camera_ready': False, 'status': 'failed', 'message': 'System still starting up.'}), 503

    if cam.isOpened():
        return jsonify({'camera_ready': True, 'status': 'already_connected', 'message': 'Camera already open.'})

    print('[connect-inputs] Attempting camera reconnect...')
    try:
        success = cam._initialize_camera()
    except Exception as e:
        print(f'[connect-inputs] Error: {e}')
        success = False

    _state.initialization_status['camera_ready'] = bool(success)
    msg = ('Camera connected successfully.' if success
           else 'Camera failed — check the GStreamer sender is running on the correct IP/port.')
    print(f'[connect-inputs] {"OK" if success else "FAILED"}')
    return jsonify({
        'camera_ready': bool(success),
        'status':       'connected' if success else 'failed',
        'message':      msg,
    }), (200 if success else 502)


@bp.route('/set-config', methods=['POST'])
def set_config():
    data    = request.get_json(silent=True) or {}
    names   = data.get('outputs', DEFAULT_CONFIG)
    invalid = [n for n in names if n not in OUTPUT_REGISTRY]

    if invalid:
        return jsonify({
            'error':     f'Unknown output(s): {invalid}',
            'available': list(OUTPUT_REGISTRY.keys()),
        }), 400

    # Binary extractors → compiled schema for /ws frames.
    # Non-binary outputs (e.g. video_stream, data_layer_stream) are handled by
    # their own dedicated endpoints.
    schema, specs        = compile_schema(names)
    video_enabled        = 'video_stream' in names
    data_layer_enabled   = 'data_layer_stream' in names

    with _state.output_config_lock:
        _state.output_config               = names
        _state.compiled_schema             = schema
        _state.compiled_specs              = specs
        _state.video_stream_enabled        = video_enabled
        _state.data_layer_stream_enabled   = data_layer_enabled

    print(f'[set-config] config={names}  video={video_enabled}  data_layer={data_layer_enabled}  schema={schema}')
    return jsonify({
        'config':               names,
        'schema':               schema_description(schema),
        'video_stream':         video_enabled,
        'data_layer_stream':    data_layer_enabled,
        'streams': {
            '/ws':              'binary model data (always active when config set)',
            '/ws/video':        'annotated JPEG frames' if video_enabled else None,
            '/ws/data_layer':   'transparent PNG annotation layer' if data_layer_enabled else None,
        },
    })


@bp.route('/get-config', methods=['GET'])
def get_config():
    with _state.output_config_lock:
        config             = _state.output_config
        schema             = _state.compiled_schema
        video_enabled      = _state.video_stream_enabled
        data_layer_enabled = _state.data_layer_stream_enabled

    if config is None or schema is None:
        from output_registry import DEFAULT_SCHEMA
        config             = DEFAULT_CONFIG
        schema             = DEFAULT_SCHEMA
        video_enabled      = False
        data_layer_enabled = False

    return jsonify({
        'config':             config,
        'schema':             schema_description(schema),
        'video_stream':       video_enabled,
        'data_layer_stream':  data_layer_enabled,
        'streams': {
            '/ws':            'binary model data (always active when config set)',
            '/ws/video':      'annotated JPEG frames' if video_enabled else None,
            '/ws/data_layer': 'transparent PNG annotation layer' if data_layer_enabled else None,
        },
    })


@bp.route('/available-outputs', methods=['GET'])
def available_outputs():
    outputs = [spec_description(spec) for spec in OUTPUT_REGISTRY.values()]
    return jsonify({'outputs': outputs})
