import os

# Suppress TensorFlow and ABSL logs before any import that touches TF.
# TF_CPP_MIN_LOG_LEVEL: 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR (only FATAL)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# GLOG_minloglevel suppresses the W0000/I0000 ABSL-format lines (gpu_timer spam etc.)
# 0=INFO 1=WARNING 2=ERROR 3=FATAL
os.environ.setdefault("GLOG_minloglevel", "2")
# Suppress oneDNN and other misc TF noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
import json
import time
import threading
import signal
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG
from emotion_detection.emotion_detector import EmotionDetector
from gpu_check import verify_gpu

app = Flask(__name__)
CORS(app)

# Models are pre-loaded at startup so the first real frame is not slow
emotion_detector = EmotionDetector(load_models_on_init=True)

# ---------------------------------------------------------------------------
# Stop event — set on shutdown/reload so all background threads exit cleanly.
# Flask's reloader sends SIGTERM to the child process; we catch it here.
# ---------------------------------------------------------------------------
_stop_event = threading.Event()

# ---------------------------------------------------------------------------
# Shared latest frame — written by _camera_reader_loop, read by all routes.
# This ensures only ONE thread ever calls camera_input.read(), eliminating
# the concurrent VideoCapture access that causes 'fctx->async_lock' crashes.
# ---------------------------------------------------------------------------
_latest_frame = None
_latest_frame_flipped = None
_frame_lock = threading.Lock()

_frame_timestamp = 0.0  # time.time() when _latest_frame was last written

def _camera_reader_loop():
    """Single thread that owns all camera reads. Exits when _stop_event is set."""
    global _latest_frame, _latest_frame_flipped, _frame_timestamp
    print("[camera] Reader thread started (drain mode — no sleep)")
    read_count = 0
    log_interval = 60  # print stats every N frames
    while not _stop_event.is_set():
        if not camera_input.isOpened():
            time.sleep(0.1)
            continue
        t_before = time.time()
        ret, frame = camera_input.read_latest()
        t_read = time.time()
        if ret and frame is not None:
            flipped = cv2.flip(frame, 1)
            with _frame_lock:
                _latest_frame = frame
                _latest_frame_flipped = flipped
                _frame_timestamp = t_read
            read_count += 1
            if read_count % log_interval == 0:
                read_ms = (t_read - t_before) * 1000
                print(f"[camera] frame #{read_count}  read={read_ms:.1f}ms")
    with _frame_lock:
        _latest_frame = None
        _latest_frame_flipped = None
    print("[camera] Reader thread stopped")

# ---------------------------------------------------------------------------
# Background inference thread
# Runs emotion detection at up to EMOTION_FPS independently of the stream FPS.
# Reads from the shared frame cache — never calls camera_input directly.
# ---------------------------------------------------------------------------
EMOTION_FPS = 10
_latest_emotion_result = {'face_detected': False}
_emotion_result_lock = threading.Lock()

def _emotion_inference_loop():
    """Runs emotion inference. Exits when _stop_event is set."""
    interval = 1.0 / EMOTION_FPS
    print("[emotion] Inference thread started")
    while not _stop_event.is_set():
        if initialization_status.get('emotion_models_loaded'):
            with _frame_lock:
                frame = _latest_frame_flipped
            if frame is not None:
                result = emotion_detector.detect_emotions_from_frame(frame, silent=True)
                with _emotion_result_lock:
                    _latest_emotion_result.clear()
                    _latest_emotion_result.update(result)
        time.sleep(interval)
    print("[emotion] Inference thread stopped")

# Initialize camera: use local cv2 device index 0 by default.
# Override with CAMERA_INDEX env var if needed.
_camera_config = {**DEFAULT_CAMERA_CONFIG}
_camera_config["camera_url"] = None
_camera_config["camera_index"] = int(os.environ.get("CAMERA_INDEX", 0))
camera_input = CameraInput(_camera_config)

# Initialization status
initialization_status = {
    'camera_ready': False,
    'tensorflow_ready': False,
    'tensorflow_gpu': False,
    'tensorflow_gpu_devices': [],
    'emotion_models_loaded': False,
    'initializing': True
}


def check_tensorflow_gpu():
    """Load TensorFlow, enable memory growth, and verify it sees GPU(s)."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        # --- NEW: Enable Memory Growth ---
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Successfully enabled TensorFlow GPU memory growth.")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"⚠️ Warning: Could not set memory growth: {e}")
        # ---------------------------------

        initialization_status['tensorflow_ready'] = True
        initialization_status['tensorflow_gpu'] = len(gpus) > 0
        initialization_status['tensorflow_gpu_devices'] = [g.name for g in gpus]
        
        if gpus:
            print(f"TensorFlow GPU available: {[g.name for g in gpus]}")
        else:
            print("TensorFlow running on CPU (no GPU devices found)")
        return True
    
    except Exception as e:
        print(f"TensorFlow check failed: {e}")
        initialization_status['tensorflow_ready'] = False
        initialization_status['tensorflow_gpu'] = False
        initialization_status['tensorflow_gpu_devices'] = []
        return False

def initialize_system():
    """Initialize camera and TensorFlow. Safe to call on each Flask reload."""
    global initialization_status

    # Reset the stop flag so threads start fresh on every reload
    _stop_event.clear()

    # Check camera (non-fatal: no camera is ok in WSL/headless)
    try:
        camera_input._initialize_camera()
        if camera_input.isOpened():
            print(f"Camera source: device index {_camera_config.get('camera_index', 0)}")
            initialization_status['camera_ready'] = True
        else:
            print("No camera source available (app will run without video feed).")
            initialization_status['camera_ready'] = False
    except Exception as e:
        print(f"Camera init failed: {e}. App will run without video feed.")
        initialization_status['camera_ready'] = False

    # Load and check TensorFlow / GPU
    check_tensorflow_gpu()

    # Load emotion models for /video_dominant_emotion
    try:
        emotion_detector.load_models()
        initialization_status['emotion_models_loaded'] = emotion_detector.models_loaded
    except Exception as e:
        print(f"Emotion models not loaded: {e}")
        initialization_status['emotion_models_loaded'] = False

    initialization_status['initializing'] = False

    # Single camera reader — the only thread that calls camera_input.read()
    threading.Thread(target=_camera_reader_loop, daemon=True, name="camera-reader").start()

    # Emotion inference reads from the shared frame cache, never from camera directly
    threading.Thread(target=_emotion_inference_loop, daemon=True, name="emotion-inference").start()

    print("Initialization complete")
    return True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def get_status():
    """Get initialization status (camera, TensorFlow, GPU)."""
    return jsonify(initialization_status)


@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Stub: emotion detection disabled in v2."""
    return jsonify({
        'status': 'disabled',
        'message': 'Emotion detection is disabled in appv2. Use app.py for emotion detection.'
    })


@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stub: emotion detection disabled in v2."""
    return jsonify({
        'status': 'disabled',
        'message': 'Emotion detection is disabled in appv2.'
    })


@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    """Stub: no emotion data in v2."""
    return jsonify({
        'emotions': {},
        'scores_out_of_10': {},
        'timestamp': None,
        'message': 'Emotion detection disabled in appv2.'
    })

def _no_camera_frame_bytes():
    """Single JPEG frame with 'No camera' text for headless/WSL."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, "No camera", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


# Lower quality = faster encode + less data over WSL2 network bridge = less lag.
# 70 is visually fine for live preview; raise to 90 if quality matters more than latency.
STREAM_JPEG_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY", 70))
_jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]

VIDEO_FEED_LOG_INTERVAL = 30  # log every N frames (~1s at 30fps)

@app.route('/video_feed')
def video_feed():
    """Video streaming route - plain feed, no emotion overlay."""
    no_camera_frame = _no_camera_frame_bytes()
    def generate():
        frame_count = 0
        while True:
            t0 = time.time()
            with _frame_lock:
                frame = _latest_frame_flipped
                frame_age = t0 - _frame_timestamp if _frame_timestamp else 0
            if frame is None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                time.sleep(0.5)
                continue
            t1 = time.time()
            ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
            t2 = time.time()
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            t3 = time.time()
            frame_count += 1
            if frame_count % VIDEO_FEED_LOG_INTERVAL == 0:
                print(
                    f"[video_feed] #{frame_count}"
                    f"  frame_age={frame_age*1000:.0f}ms"   # how old was the frame when we grabbed it
                    f"  lock={((t1-t0)*1000):.1f}ms"        # time to acquire _frame_lock
                    f"  encode={((t2-t1)*1000):.1f}ms"      # JPEG encode time
                    f"  yield={((t3-t2)*1000):.1f}ms"       # time to hand frame to Flask/network
                )
            time.sleep(0.033)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_dominant_emotion')
def video_dominant_emotion():
    """Video stream at full FPS with emotion overlay drawn from the background inference thread."""
    def generate():
        no_camera_frame = _no_camera_frame_bytes()
        while True:
            with _frame_lock:
                frame = _latest_frame_flipped
            if frame is None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                time.sleep(0.5)
                continue

            # Copy so we can draw on it without affecting the shared reference
            frame = frame.copy()

            with _emotion_result_lock:
                result = dict(_latest_emotion_result)

            if result.get('face_detected'):
                x, y, w, h = result['face_bbox']
                color = result['emotion_color_bgr']
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.01)  # ~30 FPS cap on the output side

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def _shutdown(signum=None, frame=None):
    """Signal the background threads to stop and release the camera.
    Called on SIGTERM (Flask reloader kills the child with SIGTERM) and atexit.
    """
    print(f"[shutdown] Stopping background threads (signal={signum})...")
    _stop_event.set()
    # Give threads a moment to notice the stop flag before releasing the capture
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


if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # This branch runs in the reloader child process.
        # Register cleanup for both graceful shutdown (atexit) and
        # SIGTERM (sent by Flask reloader when it kills the old child on reload).
        import atexit
        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        print("Starting initialization...")
        initialize_system()

    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, threaded=True, port=port, host='0.0.0.0')
