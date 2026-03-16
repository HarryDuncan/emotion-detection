"""
Binary serialisation primitives.

Provides Field and BinarySchema — a config-driven struct packer used by
output_registry.py to build per-frame binary frames for the /ws WebSocket.

Emotion-specific schemas and extractor functions live in output_registry.py.
"""
import struct
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Field:
    """
    One field in a BinarySchema.

    name      — identifier (appears in schema_description output)
    fmt       — single struct format character: 'i' int32, 'f' float32,
                'h' int16, 'b' int8, 'I' uint32.  See Python struct docs.
    transform — callable(flat_dict) -> value to pack
    """
    name:      str
    fmt:       str
    transform: Callable[[dict], Any]


class BinarySchema:
    """
    Config-driven struct packer.  All values packed little-endian ('<').

    Attributes
    ----------
    fields  : ordered list of Field descriptors
    fmt     : full struct format string (e.g. '<2i3f')
    size    : bytes per packed frame
    """

    def __init__(self, fields: list[Field]):
        self.fields = fields
        self.fmt    = '<' + ''.join(f.fmt for f in fields)
        self.size   = struct.calcsize(self.fmt)

    def pack(self, data: dict) -> bytes:
        """Pack *data* into bytes according to the schema."""
        return struct.pack(self.fmt, *(f.transform(data) for f in self.fields))

    def __repr__(self) -> str:
        names = ', '.join(f.name for f in self.fields)
        return f'<BinarySchema fmt={self.fmt!r} size={self.size}B [{names}]>'
