from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
import requests
import json
import time
import os
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG

app = Flask(__name__)
CORS(app)

# Initialize camera: CAMERA_URL for remote feed (e.g. http://192.168.1.10:5000/video_feed), else CAMERA_INDEX
_camera_config = {**DEFAULT_CAMERA_CONFIG}
if os.environ.get('CAMERA_URL'):
    _camera_config['camera_url'] = os.environ.get('CAMERA_URL').strip()
if os.environ.get('CAMERA_INDEX') is not None:
    _camera_config['camera_index'] = int(os.environ.get('CAMERA_INDEX', 0))
camera_input = CameraInput(_camera_config)

# Initialization status
initialization_status = {
    'camera_ready': False,
    'tensorflow_ready': False,
    'tensorflow_gpu': False,
    'tensorflow_gpu_devices': [],
    'initializing': True
}


def check_tensorflow_gpu():
    """Load TensorFlow and verify it sees GPU(s)."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
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
            print("Cleanup complete")

        atexit.register(cleanup)

    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, threaded=True, port=port, host='0.0.0.0')
