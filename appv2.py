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
# Silence OpenCV's libjpeg warnings ("Corrupt JPEG data: extraneous bytes…")
# that fire on every MJPG frame received over USBIPD. The frames are still
# decoded successfully; the messages are purely noise.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

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
#
# _frame_condition acts as both the mutex and the signal: the camera reader
# acquires it, updates the frame, increments _frame_seq, then calls
# notify_all().  Streaming generators wait with wait_for() keyed on the
# sequence number, so they wake *exactly* when a new frame arrives — no
# polling, no fixed sleep().
# ---------------------------------------------------------------------------
_frame_condition = threading.Condition()
_latest_frame = None
_latest_frame_flipped = None
_frame_timestamp = 0.0
_frame_seq = 0  # monotonically incrementing; generators track their last-seen value


def _camera_reader_loop():
    """Single thread that owns all camera reads. Exits when _stop_event is set."""
    global _latest_frame, _latest_frame_flipped, _frame_timestamp, _frame_seq
    print("[camera] Reader thread started")
    read_count = 0
    log_interval = 60
    while not _stop_event.is_set():
        if not camera_input.isOpened():
            time.sleep(0.1)
            continue
        t_before = time.time()
        ret, frame = camera_input.read_latest()
        t_read = time.time()
        if ret and frame is not None:
            flipped = cv2.flip(frame, 1)
            with _frame_condition:
                _latest_frame = frame
                _latest_frame_flipped = flipped
                _frame_timestamp = t_read
                _frame_seq += 1
                _frame_condition.notify_all()
            read_count += 1
            if read_count % log_interval == 0:
                read_ms = (t_read - t_before) * 1000
                print(f"[camera] frame #{read_count}  read={read_ms:.1f}ms")
    with _frame_condition:
        _latest_frame = None
        _latest_frame_flipped = None
        _frame_condition.notify_all()
    print("[camera] Reader thread stopped")

# ---------------------------------------------------------------------------
# Background inference thread
# Runs emotion detection at up to EMOTION_FPS independently of the stream FPS.
# Reads from the shared frame cache — never calls camera_input directly.
# ---------------------------------------------------------------------------
EMOTION_FPS = 8  # RetinaFace inference is slow; 8fps is plenty
_latest_emotion_result = {'face_detected': False, 'faces': []}
_emotion_result_lock = threading.Lock()

def _emotion_inference_loop():
    """Runs emotion inference at EMOTION_FPS. Exits when _stop_event is set."""
    interval = 1.0 / EMOTION_FPS
    print("[emotion] Inference thread started")
    _dbg_count = 0
    while not _stop_event.is_set():
        if initialization_status.get('emotion_models_loaded'):
            with _frame_condition:
                frame = _latest_frame_flipped
            if frame is not None:
                result = emotion_detector.detect_emotions_from_frame(frame, silent=False)
                with _emotion_result_lock:
                    _latest_emotion_result.clear()
                    _latest_emotion_result.update(result)
                _dbg_count += 1
                if _dbg_count % 30 == 1:
                    n = len(result.get('faces', []))
                    err = result.get('error', '')
                    print(f"[emotion] #{_dbg_count} face_detected={result.get('face_detected')} faces={n} err={err!r}")
        time.sleep(interval)
    print("[emotion] Inference thread stopped")

# Initialize camera: use local cv2 device index 0 by default.
# Override with CAMERA_INDEX env var if needed.
_camera_config = {
    **DEFAULT_CAMERA_CONFIG,
    "camera_url": None,
    "camera_index": int(os.environ.get("CAMERA_INDEX", 0)),
}
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
    """Video streaming route — plain feed, no emotion overlay."""
    no_camera_frame = _no_camera_frame_bytes()

    def generate():
        frame_count = 0
        last_seq = -1
        while True:
            # Block until the camera reader publishes a frame we haven't seen yet.
            # timeout=1.0 handles the case where the camera drops a frame or
            # hasn't opened yet — we wake up, emit the placeholder, and retry.
            with _frame_condition:
                _frame_condition.wait_for(lambda: _frame_seq != last_seq, timeout=1.0)
                frame = _latest_frame_flipped
                last_seq = _frame_seq
                frame_age = time.time() - _frame_timestamp if _frame_timestamp else 0

            if frame is None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                continue

            t_enc = time.time()
            ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_count += 1
            if frame_count % VIDEO_FEED_LOG_INTERVAL == 0:
                encode_ms = (time.time() - t_enc) * 1000
                print(f"[video_feed] #{frame_count}  frame_age={frame_age*1000:.0f}ms  encode={encode_ms:.1f}ms")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_dominant_emotion')
def video_dominant_emotion():
    """Video stream at camera FPS with per-frame emotion overlay."""
    no_camera_frame = _no_camera_frame_bytes()

    def generate():
        last_seq = -1
        while True:
            with _frame_condition:
                _frame_condition.wait_for(lambda: _frame_seq != last_seq, timeout=1.0)
                # Copy inside the lock so the camera thread can't overwrite while we draw
                frame = _latest_frame_flipped.copy() if _latest_frame_flipped is not None else None
                last_seq = _frame_seq

            if frame is None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                continue

            with _emotion_result_lock:
                result = dict(_latest_emotion_result)

            for face in result.get('faces', []):
                x, y, w, h = face['face_bbox']
                color = face['emotion_color_bgr']
                label = face.get('dominant_emotion', '')
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                ty = max(y - 6, th + 4)
                cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 2), color, cv2.FILLED)
                luma = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
                text_color = (0, 0, 0) if luma > 140 else (255, 255, 255)
                cv2.putText(frame, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
