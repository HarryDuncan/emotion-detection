import os

# Suppress TensorFlow and ABSL logs before any import that touches TF.
# TF_CPP_MIN_LOG_LEVEL: 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR (only FATAL)
# GLOG_minloglevel: 0=INFO 1=WARNING 2=ERROR 3=FATAL — silences W0000 gpu_timer spam
# Hard-assign (not setdefault) so container env vars cannot re-enable noisy levels.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel']      = '3'
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"
# Silence OpenCV libjpeg warnings from MJPG frames received over USBIPD.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import time
import threading
import signal

from flask import Flask
from flask_cors import CORS

from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG
from emotion_detection.emotion_detector import EmotionDetector
from gpu_check import verify_gpu
import state as _state
from sio import socketio

# Route blueprints — importing these modules executes all define() calls,
# which populates routes.registry.REGISTRY before the first request arrives.
from routes.registry     import bp as registry_bp
from routes.core         import bp as core_bp
from routes.video        import bp as video_bp
from routes.detection    import bp as detection_bp
from routes.ws           import sock as _ws_sock   # raw WebSocket /ws endpoint
import routes.socket_events as _socket_events      # registers @socketio.on handlers

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

app.register_blueprint(registry_bp)
app.register_blueprint(core_bp)
app.register_blueprint(video_bp)
app.register_blueprint(detection_bp)
_ws_sock.init_app(app)   # registers /ws raw WebSocket route

socketio.init_app(
    app,
    cors_allowed_origins='*',
    async_mode='threading',   # compatible with our threading.Condition state
)

# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------

# load_models_on_init=False — models are loaded inside initialize_system()
# (child process only) so Flask's reloader parent never triggers a GPU load.
emotion_detector = EmotionDetector(load_models_on_init=False)

# Camera inputs — one or two pipelines.  When two are configured, the reader
# thread auto-switches between them every CAMERA_SWITCH_INTERVAL seconds.
# Both pipelines run in parallel so the switch is a zero-downtime pointer swap.
_camera_configs = [
    {
        **DEFAULT_CAMERA_CONFIG,
        "camera_gst_pipeline": os.environ.get("CAMERA_GST_PIPELINE") or None,
        "camera_url":   None,
        "camera_index": int(os.environ.get("CAMERA_INDEX", 0)),
    },
]
_pipeline_2 = os.environ.get("CAMERA_GST_PIPELINE_2") or None
if _pipeline_2:
    _camera_configs.append({
        **DEFAULT_CAMERA_CONFIG,
        "camera_gst_pipeline": _pipeline_2,
        "camera_url":   None,
        "camera_index": int(os.environ.get("CAMERA_INDEX_2", 1)),
    })

camera_inputs = [CameraInput(cfg) for cfg in _camera_configs]
_state.camera_input     = camera_inputs[0]   # primary — exposed to route blueprints
_state.emotion_detector = emotion_detector

CAMERA_SWITCH_INTERVAL = float(os.environ.get("CAMERA_SWITCH_INTERVAL", "20"))

# ---------------------------------------------------------------------------
# Camera reader thread
# The only thread that ever calls camera_input.read_latest().
# Writes new frames into state and signals all waiting generators.
#
# When two cameras are configured, the loop auto-switches between them on a
# timer.  Both GStreamer pipelines run continuously — the inactive one keeps
# decoding with max-buffers=1/drop=true, so its buffer always holds the
# freshest frame.  Switching is just a pointer swap: no pipeline restart,
# no cold-start delay.
# ---------------------------------------------------------------------------

EMOTION_FPS = 30

def _camera_reader_loop():
    multi       = len(camera_inputs) > 1
    active_idx  = 0
    active      = camera_inputs[active_idx]
    last_switch = time.time()

    if multi:
        print(f"[camera] Reader thread started — {len(camera_inputs)} cameras, "
              f"switching every {CAMERA_SWITCH_INTERVAL}s")
    else:
        print("[camera] Reader thread started")

    while not _state.stop_event.is_set():
        # --- auto-switch ---
        if multi and CAMERA_SWITCH_INTERVAL > 0:
            now = time.time()
            if now - last_switch >= CAMERA_SWITCH_INTERVAL:
                active_idx = (active_idx + 1) % len(camera_inputs)
                active     = camera_inputs[active_idx]
                last_switch = now
                emotion_detector._tracks.clear()
                print(f"[camera] Switched to camera {active_idx}")

        if not active.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = active.read_latest()
        if ret and frame is not None:
            flipped = cv2.flip(frame, 1)
            with _state.frame_condition:
                _state.latest_frame         = frame
                _state.latest_frame_flipped = flipped
                _state.frame_timestamp      = time.time()
                _state.frame_seq           += 1
                _state.frame_condition.notify_all()

    with _state.frame_condition:
        _state.latest_frame         = None
        _state.latest_frame_flipped = None
        _state.frame_condition.notify_all()
    print("[camera] Reader thread stopped")


# ---------------------------------------------------------------------------
# Emotion inference thread
# Runs at EMOTION_FPS, reads the latest frame from state, writes results back.
# ---------------------------------------------------------------------------

def _emotion_inference_loop():
    interval  = 1.0 / EMOTION_FPS
    dbg_count = 0
    print("[emotion] Inference thread started (idle — waiting for active client)")
    while not _state.stop_event.is_set():
        # Idle cheaply when nothing is consuming emotion results.
        needs_inference = (
            _state.emotion_active_clients > 0
            or _state.emotion_explicitly_enabled
        )
        if not needs_inference or not _state.initialization_status.get('emotion_models_loaded'):
            time.sleep(0.1)
            continue

        with _state.frame_condition:
            frame = _state.latest_frame_flipped
        if frame is not None:
            result = emotion_detector.detect_emotions_from_frame(frame, silent=True)
            h, w = frame.shape[:2]
            result['frame_width']  = w
            result['frame_height'] = h
            with _state.emotion_result_lock:
                _state.latest_emotion_result.clear()
                _state.latest_emotion_result.update(result)
            # Signal the Socket.IO broadcaster that a new result is available.
            with _state.emotion_condition:
                _state.emotion_condition.notify_all()
            dbg_count += 1
            if dbg_count % 30 == 1:
                n   = len(result.get('faces', []))
                err = result.get('error', '')
                print(f"[emotion] #{dbg_count} face_detected={result.get('face_detected')} faces={n} clients={_state.emotion_active_clients} err={err!r}")

        time.sleep(interval)
    print("[emotion] Inference thread stopped")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _check_tensorflow_gpu():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("TensorFlow GPU memory growth enabled.")
            except RuntimeError as e:
                print(f"Could not set memory growth: {e}")
        _state.initialization_status['tensorflow_ready']       = True
        _state.initialization_status['tensorflow_gpu']         = len(gpus) > 0
        _state.initialization_status['tensorflow_gpu_devices'] = [g.name for g in gpus]
        print(f"TensorFlow GPU: {[g.name for g in gpus] or 'none (CPU)'}")
        return True
    except Exception as e:
        print(f"TensorFlow check failed: {e}")
        _state.initialization_status['tensorflow_ready']       = False
        _state.initialization_status['tensorflow_gpu']         = False
        _state.initialization_status['tensorflow_gpu_devices'] = []
        return False


def initialize_system():
    """
    Two-phase startup:

    Phase 1 — Camera (immediate)
        Open the capture device and start the camera-reader thread so frames
        are available to /video_feed right away.

    Phase 2 — Models (blocking, ~seconds)
        Load TensorFlow + emotion models.  The inference thread starts after
        this completes but idles at 0.1 s sleep until a client connects to
        /video_dominant_emotion or POST /start_detection is called.
    """
    _state.stop_event.clear()

    # ------------------------------------------------------------------
    # Phase 1: cameras
    # ------------------------------------------------------------------
    any_opened = False
    for i, cam in enumerate(camera_inputs):
        try:
            cam._initialize_camera()
            if cam.isOpened():
                print(f"[init] Camera {i} ready")
                any_opened = True
            else:
                print(f"[init] Camera {i} — not available")
        except Exception as e:
            print(f"[init] Camera {i} init failed: {e}")

    _state.initialization_status['camera_ready'] = any_opened
    if not any_opened:
        print("[init] No cameras available — video feed will show placeholder.")

    # Camera reader runs immediately; /video_feed is usable from this point.
    threading.Thread(target=_camera_reader_loop, daemon=True, name="camera-reader").start()

    # ------------------------------------------------------------------
    # Phase 2: models (inference stays idle until a route activates it)
    # ------------------------------------------------------------------
    _check_tensorflow_gpu()

    try:
        emotion_detector.load_models()
        _state.initialization_status['emotion_models_loaded'] = emotion_detector.models_loaded
        print("[init] Emotion models ready — inference will start on demand.")
    except Exception as e:
        print(f"[init] Emotion models not loaded: {e}")
        _state.initialization_status['emotion_models_loaded'] = False

    _state.initialization_status['initializing'] = False

    # Inference thread starts here but immediately enters its idle branch
    # (emotion_active_clients == 0 and emotion_explicitly_enabled == False).
    threading.Thread(target=_emotion_inference_loop, daemon=True, name="emotion-inference").start()

    # Socket.IO broadcaster — watches emotion_condition and emits to /emotion clients.
    _socket_events.start_broadcaster()

    print("[init] Initialization complete — camera streaming, inference idle.")
    return True


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _shutdown(signum=None, frame=None):
    print(f"[shutdown] Stopping background threads (signal={signum})...")
    _state.stop_event.set()
    time.sleep(0.15)
    for cam in camera_inputs:
        try:
            cam.release()
        except Exception:
            pass
    try:
        emotion_detector.cleanup()
    except Exception:
        pass
    print("[shutdown] Cleanup complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        import atexit
        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        print("Starting initialization...")
        initialize_system()

    port = int(os.environ.get('PORT', 5005))
    socketio.run(app, debug=True, port=port, host='0.0.0.0', allow_unsafe_werkzeug=True)
