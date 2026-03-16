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
    Describes one named model output extractor.

    name        — unique key used in set_config requests
    description — shown in GET /api-routes and set_config response
    extract     — callable(detection_result: dict) → flat dict of values
    fields      — binary fields this extractor contributes to the packed frame
    """
    name:        str
    description: str
    extract:     object   # Callable[[dict], dict]
    fields:      list     # list[Field]


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


# ---------------------------------------------------------------------------
# Schema compilation
# ---------------------------------------------------------------------------

def compile_schema(names: list[str]) -> tuple:
    """
    Build a BinarySchema from an ordered list of extractor names.

    Returns (schema, specs) — specs is the list of OutputSpec objects to call
    per frame.  Fields from all specs are concatenated in order, so the caller
    only runs struct.pack once per frame.
    """
    specs      = [OUTPUT_REGISTRY[n] for n in names if n in OUTPUT_REGISTRY]
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


# ---------------------------------------------------------------------------
# Default config (used when no set_config has been called)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ['face_detected', 'dominant_emotion', 'dominant_emotion_color']
DEFAULT_SCHEMA, DEFAULT_SPECS = compile_schema(DEFAULT_CONFIG)
