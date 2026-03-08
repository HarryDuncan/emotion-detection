from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
import requests
import json
import time
import os
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG
from emotion_detection.emotion_detector import EmotionDetector

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --- NEW: Enable massive CUDA caching ---
os.environ["CUDA_CACHE_DISABLE"] = "0"
# Set cache size to 2GB so it has plenty of room to store the 5080 binaries
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"

app = Flask(__name__)
CORS(app)
cache_dir = os.path.expanduser("~/.nv/ComputeCache")
os.environ["CUDA_CACHE_PATH"] = cache_dir
# Emotion detector for /video_dominant_emotion (models loaded on startup)
emotion_detector = EmotionDetector()

# Initialize camera: CAMERA_URL for remote feed; default to Windows interaction_node stream.
# From WSL, if localhost fails to reach Windows, set CAMERA_URL=http://<Windows_IP>:5000/interaction_node/video_feed
_default_stream_url = "http://172.29.224.1:5000/interaction_node/video_feed"
_camera_config = {**DEFAULT_CAMERA_CONFIG}
_env_url = os.environ.get("CAMERA_URL")
if _env_url is not None:
    # Explicit: use this URL, or None if set to "" (use local device)
    _camera_config["camera_url"] = _env_url.strip() or None
else:
    _camera_config["camera_url"] = os.environ.get("CAMERA_STREAM_URL", _default_stream_url)
if os.environ.get("CAMERA_INDEX") is not None:
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

def log_cuda_cache_status():
    """Logs the existence and size of the CUDA cache to verify JIT progress."""
    if os.path.exists(cache_dir):
        # Calculate the total size of the cache directory
        size_bytes = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                         for dirpath, _, filenames in os.walk(cache_dir) 
                         for filename in filenames)
        size_mb = size_bytes / (1024 * 1024)
        print(f"ℹ️ CUDA Cache Status: Found at {cache_dir}")
        print(f"ℹ️ CUDA Cache Size: {size_mb:.2f} MB")
        
        if size_mb > 50:
            print("✅ Cache looks populated. TensorFlow should load quickly!")
        else:
            print("⏳ Cache is small. JIT compilation might still be running or hasn't started.")
    else:
        print("⚠️ No CUDA Cache found. The RTX 5080 will need to JIT compile the models (this may take 30+ minutes).")
log_cuda_cache_status()

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
    """Initialize camera and TensorFlow."""
    global initialization_status

    # Check camera (non-fatal: no camera is ok in WSL/headless)
    try:
        camera_input._initialize_camera()
        if camera_input.isOpened():
            source = _camera_config.get('camera_url') or f"device index {_camera_config.get('camera_index', 0)}"
            print(f"Camera source: {source}")
            ret, frame = camera_input.read_flipped()
            if ret and frame is not None:
                print(f"Camera reading frames OK (shape {frame.shape})")
                initialization_status['camera_ready'] = True
            else:
                print("Camera opened but cannot read frames")
                initialization_status['camera_ready'] = False
        else:
            print("No camera at index (app will run without video feed).")
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


# Log video_feed frame read every N frames to confirm camera/port input
VIDEO_FEED_LOG_INTERVAL = 100  # log every N frames

@app.route('/video_feed')
def video_feed():
    """Video streaming route - plain feed, no emotion overlay."""
    def generate():
        no_camera_frame = _no_camera_frame_bytes() if not initialization_status.get('camera_ready') else None
        frame_count = 0
        while True:
            if not camera_input.isOpened():
                if no_camera_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                time.sleep(0.5)
                continue
            ret, frame = camera_input.read_flipped()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            frame_count += 1
            if frame_count % VIDEO_FEED_LOG_INTERVAL == 1:
                print(f"[video_feed] reading frames from camera input OK (frame #{frame_count}, shape {frame.shape})")
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_dominant_emotion')
def video_dominant_emotion():
    """Video stream with a colored rectangle around the face based on dominant emotion."""
    def generate():
        no_camera_frame = _no_camera_frame_bytes() if not initialization_status.get('camera_ready') else None
        while True:
            if not camera_input.isOpened():
                if no_camera_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                time.sleep(0.5)
                continue
            ret, frame = camera_input.read_flipped()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            if initialization_status.get('emotion_models_loaded'):
                result = emotion_detector.detect_emotions_from_frame(frame, silent=True)
                if result and result.get('face_detected'):
                    x, y, w, h = result['face_bbox']
                    color = result['emotion_color_bgr']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("Starting initialization...")
        initialize_system()

        import atexit

        def cleanup():
            print("Cleaning up...")
            try:
                camera_input.release()
            except Exception:
                pass
            try:
                emotion_detector.cleanup()
            except Exception:
                pass
            print("Cleanup complete")

        atexit.register(cleanup)

    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, threaded=True, port=port, host='0.0.0.0')
