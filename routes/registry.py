"""
Route registry — single source of truth for all API documentation.

Usage
-----
Call ``define(...)`` once per route (at module import time in each blueprint
file) to register its spec.  The /api-routes endpoint reads REGISTRY and
returns the full document.

FieldSpec
---------
Use ``FieldSpec`` to describe every input/output field:

    FieldSpec('string')
    FieldSpec('boolean', 'True when ready')
    FieldSpec('string',  enum=['ok', 'error'])
    FieldSpec('array',   items=FieldSpec('integer'), example=[1, 2, 3])
    FieldSpec('object',  properties={'id': FieldSpec('integer'), ...})
    FieldSpec('string',  nullable=True, example=None)

JSON Schema type names are used: string, boolean, integer, number,
array, object, null, stream (for multipart streaming responses).
"""
from dataclasses import dataclass, field
from flask import Blueprint, jsonify


# ---------------------------------------------------------------------------
# Field schema descriptor
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """Describes a single request or response field."""
    type:        str
    description: str   = ''
    nullable:    bool  = False
    enum:        list  = field(default_factory=list)
    example:     object = None
    items:       object = None   # FieldSpec describing each array element
    properties:  dict  = field(default_factory=dict)  # {name: FieldSpec} for objects


def _field_to_dict(f: FieldSpec) -> dict:
    """Serialise a FieldSpec to a plain dict, omitting default/empty values."""
    d: dict = {'type': f.type}
    if f.description:
        d['description'] = f.description
    if f.nullable:
        d['nullable'] = True
    if f.enum:
        d['enum'] = list(f.enum)
    if f.example is not None:
        d['example'] = f.example
    if f.items is not None:
        d['items'] = _field_to_dict(f.items)
    if f.properties:
        d['properties'] = {k: _field_to_dict(v) for k, v in f.properties.items()}
    return d


def _schema_to_dict(schema: dict) -> dict:
    return {
        k: (_field_to_dict(v) if isinstance(v, FieldSpec) else v)
        for k, v in schema.items()
    }


# ---------------------------------------------------------------------------
# Route spec
# ---------------------------------------------------------------------------

@dataclass
class RouteSpec:
    name:        str
    path:        str
    methods:     list
    description: str
    input:       dict = field(default_factory=dict)   # {param: FieldSpec}
    output:      dict = field(default_factory=dict)   # {field: FieldSpec}


def _route_to_dict(spec: RouteSpec) -> dict:
    return {
        'name':        spec.name,
        'path':        spec.path,
        'methods':     spec.methods,
        'description': spec.description,
        'input':       _schema_to_dict(spec.input),
        'output':      _schema_to_dict(spec.output),
    }


REGISTRY: list[RouteSpec] = []


def define(
    name:        str,
    path:        str,
    methods:     list,
    description: str,
    input:       dict = None,
    output:      dict = None,
) -> RouteSpec:
    """Register a route spec and return it.  Call once per route at import time."""
    spec = RouteSpec(
        name=name,
        path=path,
        methods=methods,
        description=description,
        input=input or {},
        output=output or {},
    )
    REGISTRY.append(spec)
    return spec


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

bp = Blueprint('registry', __name__)

_route_spec_properties = {
    'name':        FieldSpec('string', 'Unique route identifier'),
    'path':        FieldSpec('string', 'URL path', example='/health'),
    'methods':     FieldSpec('array',  'Allowed HTTP methods', items=FieldSpec('string'), example=['GET']),
    'description': FieldSpec('string', 'Human-readable description of the route'),
    'input':       FieldSpec('object', 'Request parameter schema (field name → FieldSpec)'),
    'output':      FieldSpec('object', 'Response field schema (field name → FieldSpec)'),
}

define(
    name        = 'api_routes',
    path        = '/api-routes',
    methods     = ['GET'],
    description = 'Returns the full API document — every registered route with its name, path, methods, description, and input/output schema.',
    output      = {
        'routes': FieldSpec('array', 'All registered route specs', items=FieldSpec('object', properties=_route_spec_properties)),
        'total':  FieldSpec('integer', 'Count of registered routes', example=8),
    },
)


@bp.route('/api-routes', methods=['GET'])
def api_routes():
    return jsonify({
        'routes': [_route_to_dict(r) for r in REGISTRY],
        'total':  len(REGISTRY),
    })
