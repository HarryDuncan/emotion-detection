"""
Shared mutable state for the emotion-detection app.

Imported by appv2.py (which writes to it) and by the route blueprints
(which read from it).  Keeping state here avoids circular imports and
eliminates the need for `global` statements scattered across modules.
"""
import threading

# ---------------------------------------------------------------------------
# Camera frame state
# Written exclusively by _camera_reader_loop in appv2.py.
# Read by streaming generators in routes/video.py.
# ---------------------------------------------------------------------------
frame_condition      = threading.Condition()
latest_frame         = None
latest_frame_flipped = None
frame_timestamp      = 0.0
frame_seq            = 0   # monotonically increments on every new frame

# ---------------------------------------------------------------------------
# Emotion inference state
# Written by _emotion_inference_loop in appv2.py.
# Read by the video_dominant_emotion generator in routes/video.py.
# ---------------------------------------------------------------------------
latest_emotion_result = {'face_detected': False, 'faces': []}
emotion_result_lock   = threading.Lock()

# ---------------------------------------------------------------------------
# System initialisation status
# Written by initialize_system / check_tensorflow_gpu in appv2.py.
# Read by /health and /status in routes/core.py.
# ---------------------------------------------------------------------------
initialization_status = {
    'camera_ready':           False,
    'tensorflow_ready':       False,
    'tensorflow_gpu':         False,
    'tensorflow_gpu_devices': [],
    'emotion_models_loaded':  False,
    'initializing':           True,
}

# ---------------------------------------------------------------------------
# Lifecycle
# Set by _shutdown to signal all background threads to exit.
# ---------------------------------------------------------------------------
stop_event = threading.Event()
