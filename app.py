from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
# import requests  # Commented out - chat functionality removed
import json  # Needed for SocketIO JSON parsing
import time
import threading
import logging
from collections import deque
import numpy as np
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG
from emotion_detection.emotion_detector import EmotionDetector
from emotion_detection.utils import convert_numpy_types
from emotion_transforms import emotions_to_color
from constants import TEXTURE_STREAM
# from prompt_formatting.format_prompt import format_dominant_emotion  # Commented out - chat functionality removed
# from prompt_formatting.structure import QuizResponse  # Commented out - chat functionality removed

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize classes
camera_input = CameraInput(DEFAULT_CAMERA_CONFIG)
emotion_detector = EmotionDetector(fast_mode=True)
# json_schema = QuizResponse.model_json_schema()  # Commented out - chat functionality removed

# Flag to control continuous emotion detection
emotion_detection_running = False
emotion_detection_thread = None

# SocketIO connection tracking
active_socketio_connections = set()
model_output_streaming_active = False
model_output_streaming_thread = None

# Process status constants
PROCESS_STATUS = {
    'UNINITIALIZED': 'UNINITIALIZED',
    'INITIALIZED': 'INITIALIZED',
    'RUNNING': 'RUNNING',
    'PAUSED': 'PAUSED',
    'STOPPED': 'STOPPED'
}

# State management
initialized = False
process_status = PROCESS_STATUS['UNINITIALIZED']

# Algorithm configuration
algorithm_config = {
    'type': 'full',  # 'full' or 'dominant_emotion'
    'config': {}  # Additional config options
}

# Initialization status
initialization_status = {
    'camera_ready': False,
    'models_loaded': False,
    'initializing': False
}

# Global variables for emotion detection
emotion_data = {
    'emotions': {},
    'scores_out_of_10': {},
    'timestamp': None
}
emotion_history = deque(maxlen=10)

# Global flag for texture streaming
texture_stream_enabled = False

# Global variables for video feed
latest_frame_data = {
    'frame': None,
    'detection_result': None,
    'timestamp': None,
    'lock': threading.Lock()  # Thread-safe access
}

def initialize_system():
    """Initialize camera and models synchronously"""
    global initialization_status
    
    # Check camera
    if camera_input.isOpened():
        # Test by reading a frame
        ret, frame = camera_input.read_flipped()
        if ret and frame is not None:
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

def annotate_frame_with_model_data(frame, detection_result):
    """
    Generic function to annotate frame with model detection data.
    Works with any model that returns detection results.
    
    Args:
        frame: BGR frame from OpenCV
        detection_result: Detection result dict from the model
    
    Returns:
        Annotated frame
    """
    if detection_result is None:
        return frame
    
    # Make a copy to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Draw bounding box if face is detected (emotion detection specific)
    if detection_result.get('face_detected', False):
        face_bbox = detection_result.get('face_bbox')
        dominant_emotion = detection_result.get('dominant_emotion')
        color = detection_result.get('emotion_color_bgr', (0, 255, 0))
        
        if face_bbox:
            x, y, w, h = face_bbox
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Add text label with dominant emotion
            if dominant_emotion:
                # Position text above the bounding box, or at top-left if bbox is at top
                text_y = max(y - 10, 30) if y > 30 else y + h + 25
                cv2.putText(annotated_frame, dominant_emotion.upper(), (x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif dominant_emotion:
            # If no bbox but we have emotion, show it at top-left
            cv2.putText(annotated_frame, f"Emotion: {dominant_emotion.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add other model-specific annotations here
    # This function can be extended for other models
    
    return annotated_frame

def continuous_emotion_detection():
    """Continuously detect emotions in the background"""
    global emotion_data, emotion_history, emotion_detection_running, model_output_streaming_active, latest_frame_data, algorithm_config
    
    print("Starting continuous emotion detection...")
    
    while emotion_detection_running:
        # Get frame from camera_input
        ret, frame = camera_input.read_flipped()
        
        if not ret or frame is None:
            time.sleep(0.1)
            continue
        
        # Use the emotion detector class
        result = emotion_detector.detect_emotions_from_frame(frame, silent=False)
        
        # Store frame and detection result for video feed (thread-safe)
        with latest_frame_data['lock']:
            latest_frame_data['frame'] = frame.copy()
            latest_frame_data['detection_result'] = result
            latest_frame_data['timestamp'] = time.time()
        
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
        else:
            # No face detected - clear emotion data
            emotion_data['emotions'] = {}
            emotion_data['scores_out_of_10'] = {}
            emotion_data['timestamp'] = time.time()
        
        # Emit emotion data via SocketIO if streaming is active and clients are connected
        # Format output based on algorithm config (dominant_emotion or full)
        if model_output_streaming_active and len(active_socketio_connections) > 0:
            try:
                # Get formatted emotion data based on algorithm config
                # This will return dominant_emotion format if config is 'dominant_emotion'
                # or full emotions if config is 'full'
                formatted_data = get_latest_emotions()

                # Format dominant_emotion to color and strength
                if algorithm_config['type'] == 'dominant_emotion':
                    if formatted_data.get('dominant_emotion') and formatted_data['dominant_emotion']:
                        dominant = formatted_data['dominant_emotion']
                        emotion_key = dominant['key']
                        emotion_value = dominant['value']
                        
                        # Get RGB color for the emotion
                        emotion_color = emotions_to_color({emotion_key: emotion_value})
                        
                        # Convert tuple to list for JSON serialization
                        emotion_color_list = list(emotion_color)
                        

                        # Return as JSON array with key, type, and value
                        parameter_key = algorithm_config.get('parameter_key')
                        output_type = algorithm_config.get('output_type', 'COLOR')
                        
                        if parameter_key and parameter_key.strip():  # Check if parameter_key exists and is not empty
                            output_data = [{
                                'key': f"{parameter_key}",
                                'type': output_type,
                                'value': emotion_color_list
                            }]
                           
                        else:
                            # Fallback: return just the color array if no parameter_key
                            output_data = emotion_color_list
                          
                        socketio.emit('model_output_data', output_data)
                    else:
                        # No dominant emotion detected
                        if algorithm_config.get('parameter_key'):
                            socketio.emit('model_output_data', [{
                                'key': f"{algorithm_config['parameter_key']}",
                                'type': algorithm_config.get('output_type', 'COLOR'),
                                'value': [0, 0, 0]
                            }])
                        else:
                            socketio.emit('model_output_data', [0, 0, 0])
                else:
                    # Full emotions mode - return as before
                    socketio.emit('model_output_data', formatted_data)
            except Exception as e:
                logging.error(f"Error emitting emotion data: {e}")
        
        time.sleep(0.2)  # Process ~5 frames per second for continuous detection
    
    print("Continuous emotion detection stopped")

def start_continuous_emotion_detection():
    """Start continuous emotion detection in background thread"""
    global emotion_detection_running, emotion_detection_thread, process_status
    
    if not emotion_detection_running:
        emotion_detection_running = True
        process_status = PROCESS_STATUS['RUNNING']
        emotion_detection_thread = threading.Thread(target=continuous_emotion_detection, daemon=True)
        emotion_detection_thread.start()
        return True
    return False

def stop_continuous_emotion_detection():
    """Stop continuous emotion detection and release camera"""
    global emotion_detection_running, process_status, model_output_streaming_active, initialized, latest_frame_data
    emotion_detection_running = False
    model_output_streaming_active = False  # Also stop streaming when detection stops
    process_status = PROCESS_STATUS['STOPPED']
    
    # Clear frame data
    with latest_frame_data['lock']:
        latest_frame_data['frame'] = None
        latest_frame_data['detection_result'] = None
        latest_frame_data['timestamp'] = None
    
    # Release camera
    try:
        if camera_input.isOpened():
            camera_input.release()
            print("Camera released")
    except Exception as e:
        logging.error(f"Error releasing camera: {e}")

def get_latest_emotions():
    """Get the latest emotion data based on algorithm config"""
    global algorithm_config
    
    data = emotion_data.copy()
    # Convert all numpy types to native Python types
    data['emotions'] = convert_numpy_types(data['emotions'])
    data['scores_out_of_10'] = convert_numpy_types(data['scores_out_of_10'])
    
    # Apply algorithm config to format output
    if algorithm_config['type'] == 'dominant_emotion':
        # Return only dominant emotion key and value
        if data['emotions']:
            # Find dominant emotion (highest score)
            dominant_key = max(data['emotions'].items(), key=lambda x: x[1])[0]
            dominant_value = data['emotions'][dominant_key]
            return {
                'dominant_emotion': {
                    'key': dominant_key,
                    'value': float(dominant_value)
                },
                'timestamp': data['timestamp']
            }
        else:
            return {
                'dominant_emotion': None,
                'timestamp': data['timestamp']
            }
    else:
        # Return full emotions (default)
        return data

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize camera and models with optional algorithm config"""
    global initialized, initialization_status, process_status, algorithm_config, texture_stream_enabled
    
    try:
        print("Starting initialization...")
        
        # Get algorithm config from request (optional)
        request_data = request.get_json() or {}
        print(request_data)
        
        # Get algorithm type from request_data or model_config
        model_config = request_data.get('model_config', {})
        algo_type = request_data.get('algorithm_type')
        if not algo_type:
            algo_type = model_config.get('algorithmType', 'dominant_emotion')
        
        algo_config = request_data.get('algorithm_config', {})
        
        # Extract parameter key and output type from output_config (top level) if algorithm type is dominant_emotion
        parameter_key = None
        output_type = None
        texture_stream_enabled = False  # Reset to False initially
        
        if algo_type == 'dominant_emotion':
            output_config = request_data.get('output_config', {})
            output_schema = output_config.get('outputSchema', [])
            print(f'output_config: {output_config}')
            print(f'output_schema: {output_schema}')
            
            if output_schema and len(output_schema) > 0:
                # Check all schemas for texture stream config
                for schema in output_schema:
                    if schema.get('id') == TEXTURE_STREAM['id']:
                        texture_stream_enabled = True
                        print(f'Texture stream enabled: {texture_stream_enabled}')
                        break
                
                # Find the first schema that is NOT the texture stream config
                for schema in output_schema:
                    if schema.get('id') != TEXTURE_STREAM['id']:
                        parameter_key = schema.get('parameterKey')
                        output_type = schema.get('outputType')
                        print(f'Extracted parameter_key: {parameter_key}, output_type: {output_type}')
                        break
        
        # Update global texture_stream_enabled variable
        globals()['texture_stream_enabled'] = texture_stream_enabled
        
        # Validate and set algorithm config
        valid_types = ['full', 'dominant_emotion']
        if algo_type not in valid_types:
            return jsonify({
                'status': 'error',
                'message': f'Invalid algorithm_type. Must be one of: {valid_types}',
                'processStatus': process_status,
                'optionalMessageData': None
            }), 400
        
        algorithm_config = {
            'type': algo_type,
            'config': algo_config,
            'parameter_key': parameter_key,
            'output_type': output_type
        }
        
        print(f"Algorithm config set to: {algo_type}")
        
        # Load emotion detection models
        emotion_detector.load_models()
        
        # Initialize camera
        camera_input._initialize_camera()
        
        # Verify initialization
        initialize_system()
        
        # Check if initialization was successful
        if initialization_status['camera_ready'] and initialization_status['models_loaded']:
            initialized = True
            process_status = PROCESS_STATUS['INITIALIZED']
            return jsonify({
                'status': 'success',
                'message': 'System initialized successfully',
                'processStatus': process_status,
                'optionalMessageData': {
                    'camera_ready': initialization_status['camera_ready'],
                    'models_loaded': initialization_status['models_loaded'],
                    'algorithm_type': algorithm_config['type'],
                    'algorithm_config': algorithm_config['config']
                }
            })
        else:
            process_status = PROCESS_STATUS['UNINITIALIZED']
            return jsonify({
                'status': 'error',
                'message': 'Initialization failed',
                'processStatus': process_status,
                'optionalMessageData': {
                    'camera_ready': initialization_status['camera_ready'],
                    'models_loaded': initialization_status['models_loaded']
                }
            }), 500
    except Exception as e:
        print(f"Error during initialization: {e}")
        process_status = PROCESS_STATUS['UNINITIALIZED']
        return jsonify({
            'status': 'error',
            'message': f'Initialization error: {str(e)}',
            'processStatus': process_status,
            'optionalMessageData': None
        }), 500

@app.route('/run', methods=['POST'])
def run():
    """Start continuous emotion detection"""
    global initialized, process_status
    
    if not initialized:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized. Please call /initialize first.',
            'processStatus': process_status,
            'optionalMessageData': None
        }), 400
    
    if start_continuous_emotion_detection():
        return jsonify({
            'status': 'success',
            'message': 'Continuous emotion detection started',
            'processStatus': process_status,
            'optionalMessageData': None
        })
    else:
        # Already running, but status should still be RUNNING
        if process_status != PROCESS_STATUS['RUNNING']:
            process_status = PROCESS_STATUS['RUNNING']
        return jsonify({
            'status': 'error',
            'message': 'Emotion detection already running',
            'processStatus': process_status,
            'optionalMessageData': None
        })

@app.route('/stop', methods=['POST'])
def stop():
    """Stop continuous emotion detection and release camera"""
    global process_status, initialized
    stop_continuous_emotion_detection()
    
    # Mark as uninitialized since camera is released
    initialized = False
    
    return jsonify({
        'status': 'success',
        'message': 'Emotion detection stopped and camera released',
        'processStatus': process_status,
        'optionalMessageData': None
    })
@app.route('/video_feed', methods=['GET'])
def video_feed():
    """Stream video frames from the camera with model annotations"""
    global process_status
    
    def generate():
        """Generator function that continuously yields video frames"""
        while True:
            try:
                # Check if camera is still open and emotion detection is running
                if not emotion_detection_running or not camera_input.isOpened():
                    break
                
                # Get the latest frame and detection result (thread-safe)
                with latest_frame_data['lock']:
                    frame = latest_frame_data['frame']
                    detection_result = latest_frame_data['detection_result']
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Annotate frame with model data (generic function)
                annotated_frame = annotate_frame_with_model_data(frame, detection_result)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Control frame rate (~30 FPS)
                time.sleep(0.03)
                
            except Exception as e:
                logging.error(f"Error in video feed generator: {e}")
                # Break on error to stop the generator
                break
    
    try:
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error starting video stream: {str(e)}',
            'processStatus': process_status,
            'optionalMessageData': None
        }), 500

# SocketIO event handlers for model output streaming
@socketio.on('connect')
def handle_connect():
    logging.info("=== WEBSOCKET CLIENT CONNECTED ===")
    active_socketio_connections.add(request.sid)
    emit('connected', {
        'status': 'success',
        'message': 'WebSocket connection established',
        'processStatus': process_status,
        'optionalMessageData': {
            'timestamp': time.time()
        }
    })

@socketio.on('disconnect')
def handle_disconnect():
    logging.info("=== WEBSOCKET CLIENT DISCONNECTED ===")
    active_socketio_connections.discard(request.sid)
    stop_continuous_emotion_detection()
    model_output_streaming_active = False


@socketio.on_error_default
def default_error_handler(e):
    logging.error(f"Socket.IO error: {e}")
    emit('error', {
        'status': 'error',
        'message': f'Socket.IO error: {str(e)}',
        'processStatus': process_status,
        'optionalMessageData': None
    })

@socketio.on('start_model_output')
def handle_start_model_output():
    global model_output_streaming_active, model_output_streaming_thread
    logging.info("=== START MODEL OUTPUT REQUESTED ===")
    
    try:
        if not initialized:
            emit('model_output_error', {
                'status': 'error',
                'message': 'System not initialized. Please call /initialize first.',
                'processStatus': process_status,
                'optionalMessageData': None
            })
            return
        
        if not emotion_detection_running:
            emit('model_output_error', {
                'status': 'error',
                'message': 'Emotion detection is not running. Please call /run first.',
                'processStatus': process_status,
                'optionalMessageData': None
            })
            return
        
        if model_output_streaming_active:
            emit('model_output_started', {
                'status': 'success',
                'message': 'Model output streaming already active',
                'processStatus': process_status,
                'optionalMessageData': None
            })
            return
        
        # Start streaming model output
        model_output_streaming_active = True
        
        emit('model_output_started', {
            'status': 'success',
            'message': 'Model output streaming started',
            'processStatus': process_status,
            'optionalMessageData': {
                'timestamp': time.time()
            }
        })
        
    except Exception as e:
        logging.error(f"Error starting model output: {e}")
        emit('model_output_error', {
            'status': 'error',
            'message': f'Failed to start model output: {str(e)}',
            'processStatus': process_status,
            'optionalMessageData': None
        })

@socketio.on('stop_model_output')
def handle_stop_model_output():
    global model_output_streaming_active
    logging.info("=== STOP MODEL OUTPUT REQUESTED ===")
    model_output_streaming_active = False
    emit('model_output_stopped', {
        'status': 'success',
        'message': 'Model output streaming stopped',
        'processStatus': process_status,
        'optionalMessageData': None
    })

# Commented out routes - chat functionality removed
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/status', methods=['GET'])
# def get_status():
#     """Get initialization status"""
#     return jsonify(initialization_status)
#
# @app.route('/get_emotions', methods=['GET'])
# def get_emotions():
#     """Get current emotion data"""
#     data = get_latest_emotions()
#     return jsonify(data)
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chat with Ollama, including emotion data"""
#     user_message = request.json.get('message', '')
#     
#     # Get latest emotion data
#     emotion_info = get_latest_emotions()
#     
#     # Format emotion string
#     if emotion_info['emotions']:
#         emotion_string = format_dominant_emotion(emotion_info['emotions'])
#         print(emotion_string)
#         emotion_context = f"\n\nCurrent emotional state: {emotion_string}."
#     else:
#         emotion_context = "\n\nNote: No emotion data available yet. Please start emotion detection first."
#     
#     # Create the prompt with emotion context
#     full_prompt = f"{user_message}{emotion_context}\n\nWhat do you know about these emotions? Please ask me a question about them."
#     
#     # Prepare Ollama request
#     url = "http://localhost:11434/api/chat"
#     payload = {
#         "model": "machine",
#         "messages": [{"role": "user", "content": full_prompt}],
#         "schema": json_schema,
#         "stream": True
#     }
#     
#     def generate():
#         try:
#             response = requests.post(url, json=payload, stream=True)
#             
#             if response.status_code == 200:
#                 for line in response.iter_lines(decode_unicode=True):
#                     if line:
#                         try:
#                             json_data = json.loads(line)
#                             if "message" in json_data and "content" in json_data["message"]:
#                                 content = json_data["message"]["content"]
#                                 yield f"data: {json.dumps({'content': content, 'done': json_data.get('done', False)})}\n\n"
#                         except json.JSONDecodeError:
#                             continue
#             else:
#                 yield f"data: {json.dumps({'error': f'Error: {response.status_code}'})}\n\n"
#         except Exception as e:
#             yield f"data: {json.dumps({'error': str(e)})}\n\n"
#     
#     return Response(stream_with_context(generate()), mimetype='text/event-stream')
#
# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route - reads directly from camera"""
#     def generate():
#         while True:
#             # Get frame from camera_input
#             ret, frame = camera_input.read_flipped()
#             
#             if not ret or frame is None:
#                 time.sleep(0.1)
#                 continue
#             
#             # Detect emotions using emotion_detector class
#             result = emotion_detector.detect_emotions_from_frame(frame, silent=True)
#             
#             if result and result['face_detected']:
#                 x, y, w, h = result['face_bbox']
#                 color = result['emotion_color_bgr']
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             
#             # Encode frame as JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if ret:
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#             
#             time.sleep(0.03)  # ~30 FPS
#     
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import os
    import atexit
    
    # Cleanup on exit
    def cleanup():
        global initialized, process_status
        if initialized:
            print("Cleaning up...")
            stop_continuous_emotion_detection()
            process_status = PROCESS_STATUS['STOPPED']
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
    socketio.run(app, debug=True, port=port, allow_unsafe_werkzeug=True)
