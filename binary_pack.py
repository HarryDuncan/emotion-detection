"""
Binary serialisation layer for real-time model outputs.

Packs Python dicts into compact struct bytes for Socket.IO transport,
dramatically reducing payload size vs JSON.

Usage
-----
Define a schema once, passing a list of Fields that map dict keys to struct
format characters and optional transform callables:

    from binary_pack import BinarySchema, Field, EMOTION_OUTPUT_SCHEMA

    packed: bytes = EMOTION_OUTPUT_SCHEMA.pack(data_dict)
    sio.emit('model_output', packed)           # binary frame

Unpack on the client (JavaScript / DataView, little-endian):

    socket.on('model_output', (buffer) => {
        const v = new DataView(buffer);
        const face_detected    = v.getInt32(0,  true);   // 0 or 1
        const emotion_id       = v.getInt32(4,  true);   // see EMOTION_IDS; -1 = no face
        const r                = v.getFloat32(8,  true); // 0.0–1.0
        const g                = v.getFloat32(12, true); // 0.0–1.0
        const b                = v.getFloat32(16, true); // 0.0–1.0
    });
"""
import struct
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """
    One field in a BinarySchema.

    name      — human-readable label (used in debug repr only)
    fmt       — single struct format character: 'i' int32, 'f' float32,
                'h' int16, 'b' int8, etc.  See Python struct docs.
    transform — callable(source_dict) -> value extracted/converted from the dict
    """
    name:      str
    fmt:       str
    transform: Callable[[dict], Any]


class BinarySchema:
    """
    Config-driven struct packer.

    All values are packed little-endian ('<') so DataView on the JS side
    uses true for the littleEndian argument.

    Attributes
    ----------
    fields  : list of Field descriptors
    fmt     : full struct format string (e.g. '<2i3f')
    size    : bytes per packed frame
    """

    def __init__(self, fields: list[Field]):
        self.fields = fields
        self.fmt    = '<' + ''.join(f.fmt for f in fields)
        self.size   = struct.calcsize(self.fmt)

    def pack(self, data: dict) -> bytes:
        """Pack *data* into bytes according to the schema."""
        values = [f.transform(data) for f in self.fields]
        return struct.pack(self.fmt, *values)

    def __repr__(self) -> str:
        names = ', '.join(f.name for f in self.fields)
        return f'<BinarySchema fmt={self.fmt!r} size={self.size}B fields=[{names}]>'


# ---------------------------------------------------------------------------
# Emotion output schema
#
# Wire format — 20 bytes, little-endian:
#   offset  0   int32   face_detected    0 = no face, 1 = face present
#   offset  4   int32   emotion_id       see EMOTION_IDS; -1 when no face
#   offset  8   float32 r                dominant-emotion colour, 0.0–1.0
#   offset 12   float32 g                dominant-emotion colour, 0.0–1.0
#   offset 16   float32 b                dominant-emotion colour, 0.0–1.0
#
# Compared to the equivalent JSON (~100 bytes) this is 20 bytes — 5× smaller.
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


def _channel(data: dict, idx: int) -> float:
    """Extract one RGB channel (0–255) from data['color_rgb'] and normalise to 0.0–1.0."""
    rgb = data.get('color_rgb') or [0, 0, 0]
    return float(rgb[idx]) / 255.0


EMOTION_OUTPUT_SCHEMA = BinarySchema([
    Field('face_detected',    'i', lambda d: int(d.get('face_detected', False))),
    Field('dominant_emotion', 'i', lambda d: EMOTION_IDS.get(d.get('dominant_emotion'), -1)),
    Field('r',                'f', lambda d: _channel(d, 0)),
    Field('g',                'f', lambda d: _channel(d, 1)),
    Field('b',                'f', lambda d: _channel(d, 2)),
])
