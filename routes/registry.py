"""
Route registry — single source of truth for all API documentation.

Usage
-----
Call ``define(...)`` once per route (at module import time in each blueprint
file) to register its spec.  The /api-routes endpoint reads REGISTRY and
returns the full document.

Example
-------
    from routes.registry import define

    define(
        name        = 'my_route',
        path        = '/my-route',
        methods     = ['GET'],
        description = 'Does something useful.',
        output      = {'result': 'str'},
    )

    @bp.route('/my-route')
    def my_route():
        ...
"""
from dataclasses import dataclass, field, asdict
from flask import Blueprint, jsonify


@dataclass
class RouteSpec:
    name:        str
    path:        str
    methods:     list
    description: str
    input:       dict = field(default_factory=dict)
    output:      dict = field(default_factory=dict)


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

define(
    name        = 'api_routes',
    path        = '/api-routes',
    methods     = ['GET'],
    description = 'Returns the full API document — every registered route with its name, path, methods, description, and input/output schema.',
    output      = {
        'routes': 'list[RouteSpec] — all registered routes',
        'total':  'int — count of registered routes',
    },
)


@bp.route('/api-routes', methods=['GET'])
def api_routes():
    return jsonify({
        'routes': [asdict(r) for r in REGISTRY],
        'total':  len(REGISTRY),
    })
