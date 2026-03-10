import os

# Suppress TensorFlow and ABSL logs before any import that touches TF.
# TF_CPP_MIN_LOG_LEVEL: 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR (only FATAL)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# GLOG_minloglevel suppresses the W0000/I0000 ABSL-format lines (gpu_timer spam etc.)
# 0=INFO 1=WARNING 2=ERROR 3=FATAL
os.environ.setdefault("GLOG_minloglevel", "2")
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

# Route blueprints — importing these modules executes all define() calls,
# which populates routes.registry.REGISTRY before the first request arrives.
from routes.registry  import bp as registry_bp
from routes.core      import bp as core_bp
from routes.video     import bp as video_bp
from routes.detection import bp as detection_bp

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

app.register_blueprint(registry_bp)
app.register_blueprint(core_bp)
app.register_blueprint(video_bp)
app.register_blueprint(detection_bp)

# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------

emotion_detector = EmotionDetector(load_models_on_init=True)

_camera_config = {
    **DEFAULT_CAMERA_CONFIG,
    "camera_url":   None,
    "camera_index": int(os.environ.get("CAMERA_INDEX", 0)),
}
camera_input = CameraInput(_camera_config)

# ---------------------------------------------------------------------------
# Camera reader thread
# The only thread that ever calls camera_input.read_latest().
# Writes new frames into state and signals all waiting generators.
# ---------------------------------------------------------------------------

EMOTION_FPS = 30   # RetinaFace inference is slow; 8 fps is plenty

def _camera_reader_loop():
    print("[camera] Reader thread started")
    read_count = 0
    while not _state.stop_event.is_set():
        if not camera_input.isOpened():
            time.sleep(0.1)
            continue
        t_before = time.time()
        ret, frame = camera_input.read_latest()
        t_read = time.time()
        if ret and frame is not None:
            flipped = cv2.flip(frame, 1)
            with _state.frame_condition:
                _state.latest_frame         = frame
                _state.latest_frame_flipped = flipped
                _state.frame_timestamp      = t_read
                _state.frame_seq           += 1
                _state.frame_condition.notify_all()
            read_count += 1
            if read_count % 60 == 0:
                read_ms = (t_read - t_before) * 1000
                print(f"[camera] frame #{read_count}  read={read_ms:.1f}ms")
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
    interval   = 1.0 / EMOTION_FPS
    dbg_count  = 0
    print("[emotion] Inference thread started")
    while not _state.stop_event.is_set():
        if _state.initialization_status.get('emotion_models_loaded'):
            with _state.frame_condition:
                frame = _state.latest_frame_flipped
            if frame is not None:
                result = emotion_detector.detect_emotions_from_frame(frame, silent=False)
                with _state.emotion_result_lock:
                    _state.latest_emotion_result.clear()
                    _state.latest_emotion_result.update(result)
                dbg_count += 1
                if dbg_count % 30 == 1:
                    n   = len(result.get('faces', []))
                    err = result.get('error', '')
                    print(f"[emotion] #{dbg_count} face_detected={result.get('face_detected')} faces={n} err={err!r}")
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
    """Initialise camera, TensorFlow, and emotion models then start threads."""
    _state.stop_event.clear()

    try:
        camera_input._initialize_camera()
        if camera_input.isOpened():
            print(f"Camera source: device index {_camera_config.get('camera_index', 0)}")
            _state.initialization_status['camera_ready'] = True
        else:
            print("No camera available — app will run without video feed.")
            _state.initialization_status['camera_ready'] = False
    except Exception as e:
        print(f"Camera init failed: {e}")
        _state.initialization_status['camera_ready'] = False

    _check_tensorflow_gpu()

    try:
        emotion_detector.load_models()
        _state.initialization_status['emotion_models_loaded'] = emotion_detector.models_loaded
    except Exception as e:
        print(f"Emotion models not loaded: {e}")
        _state.initialization_status['emotion_models_loaded'] = False

    _state.initialization_status['initializing'] = False

    threading.Thread(target=_camera_reader_loop,    daemon=True, name="camera-reader").start()
    threading.Thread(target=_emotion_inference_loop, daemon=True, name="emotion-inference").start()

    print("Initialization complete")
    return True


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _shutdown(signum=None, frame=None):
    print(f"[shutdown] Stopping background threads (signal={signum})...")
    _state.stop_event.set()
    time.sleep(0.15)
    try:
        camera_input.release()
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
    app.run(debug=True, threaded=True, port=port, host='0.0.0.0')
