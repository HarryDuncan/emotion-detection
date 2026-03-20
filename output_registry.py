"""
Output extractor registry.

Each OutputSpec is a named, standalone extractor function that takes a raw
detection result dict and returns a flat dict of typed values.  Extractors
are composable: call as many as you like after a single detection pass.

Registration
------------
    OUTPUT_REGISTRY['dominant_emotion']  → OutputSpec
    OUTPUT_REGISTRY['dominant_emotion_color']  → OutputSpec
    ...

Schema compilation
------------------
When POST /set-config is called, compile_schema(names) builds a BinarySchema
from the ordered union of fields across the selected extractors.  The compiled
schema is stored in state and used by /ws to pack every frame — one struct.pack
call, zero JSON overhead.

    schema, specs = compile_schema(['face_detected', 'dominant_emotion_color'])
    flat = run_extractors(detection_result, specs)
    binary = schema.pack(flat)
"""
import struct
from dataclasses import dataclass

from binary_pack import BinarySchema, Field


# ---------------------------------------------------------------------------
# Emotion ID tables (shared by extractors + HTTP routes)
# ---------------------------------------------------------------------------

EMOTION_IDS: dict[str, int] = {
    'angry':    0,
    'disgust':  1,
    'fear':     2,
    'happy':    3,
    'sad':      4,
    'surprise': 5,
    'neutral':  6,
}

ID_TO_EMOTION: dict[int, str] = {v: k for k, v in EMOTION_IDS.items()}

_EMOTIONS_ORDER = list(EMOTION_IDS.keys())  # stable order for all_emotions packing


# ---------------------------------------------------------------------------
# OutputSpec
# ---------------------------------------------------------------------------

@dataclass
class OutputSpec:
    """
    Describes one named model output.

    name            — unique key used in set-config requests
    description     — shown in GET /available-outputs
    extract         — callable(detection_result: dict) → flat dict of values
                      None for non-binary outputs (e.g. video_stream)
    fields          — binary fields this extractor contributes to the packed frame
                      empty list for non-binary outputs
    kind            — 'binary' (packed into /ws frames) | 'video' (separate stream)
    endpoint        — WebSocket endpoint URL for kind='video'; empty for kind='binary'
    is_array        — True when the payload is a fixed-length array of uniform items.
                      Wire format is identical to a plain binary spec; this flag lets
                      JS clients use Float32Array + uniform2fv/uniform3fv directly
                      instead of parsing each field individually with DataView.
    array_length    — Number of items in the array (e.g. 6 face slots). 0 if not an array.
    item_components — Number of float32 components per item (e.g. 2 for vec2, 3 for vec3).
                      item_stride_bytes = item_components × 4.  0 if not an array.
    """
    name:            str
    description:     str
    extract:         object   # Callable[[dict], dict] | None
    fields:          list     # list[Field]
    kind:            str = 'binary'        # 'binary' | 'video'
    endpoint:        str = ''              # ws endpoint for non-binary outputs
    format:          str = 'jpeg_binary'   # wire format hint for non-binary outputs
    is_array:        bool = False
    array_length:    int  = 0
    item_components: int  = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _faces(result: dict) -> list:
    return result.get('faces') or []

def _first_face(result: dict) -> dict:
    f = _faces(result)
    return f[0] if f else {}


# ---------------------------------------------------------------------------
# Standalone extractor functions
# ---------------------------------------------------------------------------

def extract_face_detected(result: dict) -> dict:
    """Whether any face is visible and the total face count."""
    return {
        'face_detected': int(result.get('face_detected', False)),
        'face_count':    len(_faces(result)),
    }


def extract_dominant_emotion(result: dict) -> dict:
    """Dominant emotion ID for the first/largest face (-1 if no face)."""
    face    = _first_face(result)
    emotion = face.get('dominant_emotion') if face else None
    return {
        'dominant_emotion_id':   EMOTION_IDS.get(emotion, -1),
        'dominant_emotion_name': emotion or '',
    }


def extract_dominant_emotion_color(result: dict) -> dict:
    """RGB colour (normalised 0.0–1.0) for the dominant emotion of the first face."""
    face = _first_face(result)
    if not face:
        return {'color_r': 0.0, 'color_g': 0.0, 'color_b': 0.0}
    rgb = list(face.get('emotion_color_rgb') or (0, 0, 0))
    return {
        'color_r': float(rgb[0]) / 255.0,
        'color_g': float(rgb[1]) / 255.0,
        'color_b': float(rgb[2]) / 255.0,
    }


def extract_all_emotions(result: dict) -> dict:
    """Smoothed score (0–100) for every emotion of the first face."""
    face    = _first_face(result)
    scores  = face.get('emotions') or {} if face else {}
    return {f'emotion_{k}': float(scores.get(k, 0.0)) for k in _EMOTIONS_ORDER}


def extract_face_bbox(result: dict) -> dict:
    """Bounding box (x, y, w, h pixels) for the first/largest face."""
    face = _first_face(result)
    if not face:
        return {'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 0, 'bbox_h': 0}
    x, y, w, h = face.get('face_bbox') or (0, 0, 0, 0)
    return {'bbox_x': int(x), 'bbox_y': int(y), 'bbox_w': int(w), 'bbox_h': int(h)}


# Maximum number of simultaneous faces tracked across all multi-face outputs.
# Both face_positions and face_colors use this constant so slot i is always
# the same face in both outputs within a single frame.
_FACE_SLOTS = 6


def extract_face_positions(result: dict) -> dict:
    """
    Normalised UV centre for up to _FACE_SLOTS faces.

    Per slot i (0 … 5):
        face_i_pos_x  float32  0.0–1.0, -1.0 = empty
        face_i_pos_y  float32  0.0–1.0, -1.0 = empty

    Origin top-left; x right, y down (image convention).
    Flip y in the shader (1.0 - y) to match WebGL UV space.
    Wire size: 6 × 2 × 4 = 48 bytes.
    """
    faces = _faces(result)[:_FACE_SLOTS]
    fw    = result.get('frame_width')  or 1
    fh    = result.get('frame_height') or 1
    flat: dict = {}
    for i in range(_FACE_SLOTS):
        if i < len(faces):
            x, y, w, h = faces[i].get('face_bbox') or (0, 0, 0, 0)
            flat[f'face_{i}_pos_x'] = float(x + w * 0.5) / fw
            flat[f'face_{i}_pos_y'] = float(y + h * 0.5) / fh
        else:
            flat[f'face_{i}_pos_x'] = -1.0
            flat[f'face_{i}_pos_y'] = -1.0
    return flat


def extract_face_colors(result: dict) -> dict:
    """
    Dominant emotion colour for up to _FACE_SLOTS faces.

    Slot ordering is identical to extract_face_positions — slot i refers to
    the same face in both outputs so they can be used independently or together.

    Per slot i (0 … 5):
        face_i_color_r  float32  0.0–1.0
        face_i_color_g  float32  0.0–1.0
        face_i_color_b  float32  0.0–1.0

    Empty slots have color (0.0, 0.0, 0.0).
    Wire size: 6 × 3 × 4 = 72 bytes.
    """
    faces = _faces(result)[:_FACE_SLOTS]
    flat: dict = {}
    for i in range(_FACE_SLOTS):
        if i < len(faces):
            rgb = list(faces[i].get('emotion_color_rgb') or (0, 0, 0))
            flat[f'face_{i}_color_r'] = float(rgb[0]) / 255.0
            flat[f'face_{i}_color_g'] = float(rgb[1]) / 255.0
            flat[f'face_{i}_color_b'] = float(rgb[2]) / 255.0
        else:
            flat[f'face_{i}_color_r'] = 0.0
            flat[f'face_{i}_color_g'] = 0.0
            flat[f'face_{i}_color_b'] = 0.0
    return flat


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OUTPUT_REGISTRY: dict[str, OutputSpec] = {}


def _reg(spec: OutputSpec) -> OutputSpec:
    OUTPUT_REGISTRY[spec.name] = spec
    return spec


_reg(OutputSpec(
    name        = 'face_detected',
    description = 'Whether any face is visible (int32 0/1) and total face count (int32).',
    extract     = extract_face_detected,
    fields      = [
        Field('face_detected', 'i', lambda d: d.get('face_detected', 0)),
        Field('face_count',    'i', lambda d: d.get('face_count', 0)),
    ],
))

_reg(OutputSpec(
    name        = 'dominant_emotion',
    description = (
        'Dominant emotion ID for the first face (int32). '
        'angry=0 disgust=1 fear=2 happy=3 sad=4 surprise=5 neutral=6 none=-1.'
    ),
    extract     = extract_dominant_emotion,
    fields      = [
        Field('dominant_emotion', 'i', lambda d: d.get('dominant_emotion_id', -1)),
    ],
))

_reg(OutputSpec(
    name        = 'dominant_emotion_color',
    description = 'RGB colour (float32 each, 0.0–1.0) associated with the dominant emotion.',
    extract     = extract_dominant_emotion_color,
    fields      = [
        Field('color_r', 'f', lambda d: d.get('color_r', 0.0)),
        Field('color_g', 'f', lambda d: d.get('color_g', 0.0)),
        Field('color_b', 'f', lambda d: d.get('color_b', 0.0)),
    ],
))

_reg(OutputSpec(
    name        = 'all_emotions',
    description = (
        'Smoothed score (float32, 0–100) for every emotion of the first face, '
        'in order: angry disgust fear happy sad surprise neutral.'
    ),
    extract     = extract_all_emotions,
    fields      = [
        Field(f'emotion_{k}', 'f', lambda d, k=k: d.get(f'emotion_{k}', 0.0))
        for k in _EMOTIONS_ORDER
    ],
))

_reg(OutputSpec(
    name        = 'face_bbox',
    description = 'Bounding box (int32 x, y, w, h pixels) for the first/largest face.',
    extract     = extract_face_bbox,
    fields      = [
        Field('bbox_x', 'i', lambda d: d.get('bbox_x', 0)),
        Field('bbox_y', 'i', lambda d: d.get('bbox_y', 0)),
        Field('bbox_w', 'i', lambda d: d.get('bbox_w', 0)),
        Field('bbox_h', 'i', lambda d: d.get('bbox_h', 0)),
    ],
))

_reg(OutputSpec(
    name            = 'face_positions',
    description     = (
        f'Normalised UV centre for up to {_FACE_SLOTS} faces (48 bytes). '
        f'Fixed array of {_FACE_SLOTS} slots × (pos_x f32, pos_y f32). '
        'Origin top-left; flip pos_y in shader (1.0 - y) for WebGL UV space. '
        'Slot i matches slot i in face_colors. '
        'Empty slots: pos_x = pos_y = -1.0.'
    ),
    extract         = extract_face_positions,
    fields          = [
        Field(f'face_{i}_{c}', 'f',
              (lambda k: lambda d: d.get(k, -1.0))(f'face_{i}_{c}'))
        for i in range(_FACE_SLOTS)
        for c in ('pos_x', 'pos_y')
    ],
    is_array        = True,
    array_length    = _FACE_SLOTS,
    item_components = 2,   # vec2
))

_reg(OutputSpec(
    name            = 'face_colors',
    description     = (
        f'Dominant emotion colour for up to {_FACE_SLOTS} faces (72 bytes). '
        f'Fixed array of {_FACE_SLOTS} slots × (color_r f32, color_g f32, color_b f32), each 0.0–1.0. '
        'Slot i matches slot i in face_positions. '
        'Empty slots: all zeros.'
    ),
    extract         = extract_face_colors,
    fields          = [
        Field(f'face_{i}_{c}', 'f',
              (lambda k: lambda d: d.get(k, 0.0))(f'face_{i}_{c}'))
        for i in range(_FACE_SLOTS)
        for c in ('color_r', 'color_g', 'color_b')
    ],
    is_array        = True,
    array_length    = _FACE_SLOTS,
    item_components = 3,   # vec3
))

_reg(OutputSpec(
    name        = 'video_stream',
    description = (
        'Annotated JPEG video frames with emotion bounding boxes and labels. '
        'Streams on /ws/video at camera rate (~15–30 fps). '
        'Connect with ws.binaryType = "arraybuffer" and decode each message as JPEG.'
    ),
    extract     = None,
    fields      = [],
    kind        = 'video',
    endpoint    = '/ws/video',
    format      = 'jpeg_binary',
))

_reg(OutputSpec(
    name        = 'data_layer_stream',
    description = (
        'Transparent RGBA PNG frames containing only the annotation layer — '
        'bounding boxes and emotion labels on a fully transparent background. '
        'No camera image. Composite over the raw video feed to keep video and '
        'annotations as independent layers. '
        'WebSocket: /ws/data_layer at camera rate (~15–30 fps) — '
        'set ws.binaryType = "arraybuffer" and decode each message as PNG. '
        'MJPEG: /video_data_layer as multipart/x-mixed-replace with image/png parts.'
    ),
    extract     = None,
    fields      = [],
    kind        = 'video',
    endpoint    = '/ws/data_layer',
    format      = 'png_binary',
))


# ---------------------------------------------------------------------------
# Schema compilation
# ---------------------------------------------------------------------------

def compile_schema(names: list[str]) -> tuple:
    """
    Build a BinarySchema from an ordered list of extractor names.

    Non-binary outputs (kind != 'binary', e.g. video_stream) are silently
    skipped — they are handled by their own dedicated endpoints.

    Returns (schema, specs) — specs is the list of binary OutputSpec objects to
    call per frame.  Fields from all specs are concatenated in order, so the
    caller only runs struct.pack once per frame.
    """
    specs      = [OUTPUT_REGISTRY[n] for n in names
                  if n in OUTPUT_REGISTRY and OUTPUT_REGISTRY[n].kind == 'binary']
    all_fields = [f for spec in specs for f in spec.fields]
    return BinarySchema(all_fields), specs


def run_extractors(result: dict, specs: list) -> dict:
    """
    Run a list of OutputSpec extractors against one detection result.
    Returns a merged flat dict ready to pass to BinarySchema.pack().
    """
    flat: dict = {}
    for spec in specs:
        flat.update(spec.extract(result))
    return flat


def schema_description(schema: BinarySchema) -> dict:
    """
    Describe a BinarySchema in a JSON-serialisable form so JS clients can
    dynamically build their DataView parsing code.
    """
    _fmt_names = {'i': 'int32', 'f': 'float32', 'h': 'int16', 'b': 'int8', 'I': 'uint32'}
    offset = 0
    fields = []
    for f in schema.fields:
        size = struct.calcsize('<' + f.fmt)
        fields.append({'name': f.name, 'type': _fmt_names.get(f.fmt, f.fmt), 'offset': offset, 'size': size})
        offset += size
    return {'size_bytes': schema.size, 'fields': fields}


def spec_description(spec: OutputSpec) -> dict:
    """
    Describe one OutputSpec in a JSON-serialisable form for GET /available-outputs.
    Binary specs include a binary schema layout; video specs include their endpoint.

    Array specs additionally include:
        is_array        — always True
        array_length    — number of items (e.g. 6 face slots)
        item_components — float32 values per item (e.g. 2 = vec2, 3 = vec3)
        item_stride     — bytes per item (item_components × 4)

    JS usage for array specs:
        const arr = new Float32Array(buffer, byteOffset, array_length * item_components);
        gl.uniform2fv(loc, arr);  // item_components=2
        gl.uniform3fv(loc, arr);  // item_components=3
    """
    base = {
        'name':        spec.name,
        'description': spec.description,
        'kind':        spec.kind,
    }
    if spec.kind == 'binary':
        single_schema, _ = compile_schema([spec.name])
        base['schema']   = schema_description(single_schema)
        if spec.is_array:
            base['is_array']        = True
            base['array_length']    = spec.array_length
            base['item_components'] = spec.item_components
            base['item_stride']     = spec.item_components * 4
    else:
        base['endpoint'] = spec.endpoint
        base['format']   = spec.format
    return base


# ---------------------------------------------------------------------------
# Default config (used when no set_config has been called)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ['face_detected', 'dominant_emotion', 'dominant_emotion_color']
DEFAULT_SCHEMA, DEFAULT_SPECS = compile_schema(DEFAULT_CONFIG)
