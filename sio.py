"""
SocketIO singleton.

Kept in its own module so both appv2.py and routes/socket_events.py can
import it without creating a circular dependency.

appv2.py calls socketio.init_app(app, ...) at startup.
routes/socket_events.py decorates handlers with @socketio.on(...).
"""
from flask_socketio import SocketIO

socketio = SocketIO()
