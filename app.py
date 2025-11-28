from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import requests
import json
import time
import threading
from collections import deque
import numpy as np
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG
from emotion_detection.emotion_detector import EmotionDetector
from emotion_detection.utils import convert_numpy_types
from prompt_formatting.format_prompt import format_dominant_emotion
from prompt_formatting.structure import QuizResponse
app = Flask(__name__)
CORS(app)

# Initialize classes
camera_input = CameraInput(DEFAULT_CAMERA_CONFIG)
emotion_detector = EmotionDetector()
json_schema = QuizResponse.model_json_schema()
# Flag to control continuous emotion detection
emotion_detection_running = False
emotion_detection_thread = None

# Initialization status
initialization_status = {
    'camera_ready': False,
    'models_loaded': False,
    'initializing': True
}

# Global variables for emotion detection
emotion_data = {
    'emotions': {},
    'scores_out_of_10': {},
    'timestamp': None
}
emotion_history = deque(maxlen=10)

def initialize_system():
    """Initialize camera and models synchronously"""
    global initialization_status
    
    # Check camera
    if camera_input.isOpened():
        # Test by reading a frame
        ret, frame = camera_input.read_flipped()
        if ret and frame is not None:
            print("Camera got framse")
            initialization_status['camera_ready'] = True
        else:
            print("Camera opened but cannot read frames")
            initialization_status['camera_ready'] = False
    else:
        print("Error: Could not open camera")
        initialization_status['camera_ready'] = False
    
    # Check models
    if emotion_detector.models_loaded:
        print("Models loaded successfully")
        initialization_status['models_loaded'] = True
    else:
        print("Warning: Models not loaded")
        initialization_status['models_loaded'] = False
    
    initialization_status['initializing'] = False
    print("Initialization complete")
    return True

def convert_to_out_of_10(emotions_dict):
    """Convert emotion scores from 0-100 to 0-10 scale"""
    # Convert numpy types first, then convert to out of 10
    converted = convert_numpy_types(emotions_dict)
    return {emotion: round(float(score) / 10, 1) for emotion, score in converted.items()}

def format_emotion_string(emotions_dict, scores_out_of_10):
    """Format emotions for the prompt"""
    emotion_list = []
    for emotion, score_10 in scores_out_of_10.items():
        emotion_list.append(f"{emotion}: {score_10}/10")
    return ", ".join(emotion_list)


def continuous_emotion_detection():
    """Continuously detect emotions in the background"""
    global emotion_data, emotion_history, emotion_detection_running
    
    print("Starting continuous emotion detection...")
    
    while emotion_detection_running:
        # Get frame from camera_input
        ret, frame = camera_input.read_flipped()
        
        if not ret or frame is None:
            time.sleep(0.1)
            continue
        
        # Use the emotion detector class
        result = emotion_detector.detect_emotions_from_frame(frame, silent=True)
        
        if result and result['face_detected']:
            emotions = result['emotions']
            
            # Convert numpy types to native Python types
            emotions = convert_numpy_types(emotions)
            
            # Store in history
            emotion_history.append({
                'emotions': emotions,
                'timestamp': time.time()
            })
            
            # Update current emotion data
            emotion_data['emotions'] = emotions
            emotion_data['scores_out_of_10'] = convert_to_out_of_10(emotions)
            emotion_data['timestamp'] = time.time()
        
        time.sleep(0.2)  # Process ~5 frames per second for continuous detection
    
    print("Continuous emotion detection stopped")

def start_continuous_emotion_detection():
    """Start continuous emotion detection in background thread"""
    global emotion_detection_running, emotion_detection_thread
    
    if not emotion_detection_running:
        emotion_detection_running = True
        emotion_detection_thread = threading.Thread(target=continuous_emotion_detection, daemon=True)
        emotion_detection_thread.start()
        return True
    return False

def stop_continuous_emotion_detection():
    """Stop continuous emotion detection"""
    global emotion_detection_running
    emotion_detection_running = False

def get_latest_emotions():
    """Get the latest emotion data"""
    data = emotion_data.copy()
    # Convert all numpy types to native Python types
    data['emotions'] = convert_numpy_types(data['emotions'])
    data['scores_out_of_10'] = convert_numpy_types(data['scores_out_of_10'])
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    """Get initialization status"""
    return jsonify(initialization_status)

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start continuous emotion detection"""
    if start_continuous_emotion_detection():
        return jsonify({'status': 'started', 'message': 'Continuous emotion detection started'})
    else:
        return jsonify({'status': 'already_running', 'message': 'Emotion detection already running'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop continuous emotion detection"""
    stop_continuous_emotion_detection()
    return jsonify({'status': 'stopped', 'message': 'Emotion detection stopped'})

@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    """Get current emotion data"""
    data = get_latest_emotions()
    return jsonify(data)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat with Ollama, including emotion data"""
    user_message = request.json.get('message', '')
    
    # Get latest emotion data
    emotion_info = get_latest_emotions()
    
    # Format emotion string
    if emotion_info['emotions']:
        emotion_string = format_dominant_emotion(emotion_info['emotions'])
        print(emotion_string)
        emotion_context = f"\n\nCurrent emotional state: {emotion_string}."
    else:
        emotion_context = "\n\nNote: No emotion data available yet. Please start emotion detection first."
    
    # Create the prompt with emotion context
    full_prompt = f"{user_message}{emotion_context}\n\nWhat do you know about these emotions? Please ask me a question about them."
    
    # Prepare Ollama request
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "machine",
        "messages": [{"role": "user", "content": full_prompt}],
        "schema": json_schema,
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
    """Video streaming route - reads directly from camera"""
    def generate():
        while True:
            # Get frame from camera_input
            ret, frame = camera_input.read_flipped()
            
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            # Detect emotions using emotion_detector class
            result = emotion_detector.detect_emotions_from_frame(frame, silent=True)
            
            if result and result['face_detected']:
                x, y, w, h = result['face_bbox']
                color = result['emotion_color_bgr']
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import os
    
    # Only initialize in the actual Flask process, not the reloader parent process
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # Initialize system (camera and models) synchronously before starting Flask
        print("Starting initialization...")
        emotion_detector.load_models()
        camera_input._initialize_camera()
        initialize_system()
        
        # Start continuous emotion detection automatically
        start_continuous_emotion_detection()
        
        # Cleanup on exit
        import atexit
        def cleanup():
            print("Cleaning up...")
            stop_continuous_emotion_detection()
            try:
                camera_input.release()
            except:
                pass
            try:
                emotion_detector.cleanup()
            except:
                pass
            print("Cleanup complete")
        
        atexit.register(cleanup)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, threaded=True, port=port)
