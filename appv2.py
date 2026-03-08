from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import requests
import json
import time
import os
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG

app = Flask(__name__)
CORS(app)

# Initialize camera
camera_input = CameraInput(DEFAULT_CAMERA_CONFIG)

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

    # Check camera
    if camera_input.isOpened():
        ret, frame = camera_input.read_flipped()
        if ret and frame is not None:
            print("Camera got frames")
            initialization_status['camera_ready'] = True
        else:
            print("Camera opened but cannot read frames")
            initialization_status['camera_ready'] = False
    else:
        print("Error: Could not open camera")
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


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat with Ollama (no emotion context in v2)."""
    user_message = request.json.get('message', '')

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "machine",
        "messages": [{"role": "user", "content": user_message}],
        "stream": True
    }

    def generate():
        try:
            response = requests.post(url, json=payload, stream=True)

            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_data = json.loads(line)
                            if "message" in json_data and "content" in json_data["message"]:
                                content = json_data["message"]["content"]
                                yield f"data: {json.dumps({'content': content, 'done': json_data.get('done', False)})}\n\n"
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"data: {json.dumps({'error': f'Error: {response.status_code}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/video_feed')
def video_feed():
    """Video streaming route - plain feed, no emotion overlay."""
    def generate():
        while True:
            ret, frame = camera_input.read_flipped()

            if not ret or frame is None:
                time.sleep(0.1)
                continue

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
        camera_input._initialize_camera()
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
    app.run(debug=True, threaded=True, port=port)
