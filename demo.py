
import time
import socket
import os
import warnings
import logging
import cv2

# Suppress warnings before importing TensorFlow
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow C++ logs (0=all, 1=info, 2=warnings, 3=errors)

from detector.emotion_detector import EmotionDetector
from detector.utils import convert_numpy_types
from emotion_transforms import emotions_to_color

# Try to import TensorFlow for GPU checking
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    # Set TensorFlow logger to ERROR level to suppress warnings
    tf.get_logger().setLevel(logging.ERROR)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

def format_color_display(rgb_color):
    """Format RGB color for display"""
    r, g, b = rgb_color
    return f"RGB({r}, {g}, {b})"

def display_emotion_data(emotions, dominant_emotion, emotion_color_rgb, emotion_color_bgr):
    """Display all emotion data in a formatted way"""

    if not emotions:
        print("No face detected or no emotion data available")
     
        return
    
    # Display dominant emotion
    if dominant_emotion:
        print(f"\nDominant Emotion: {dominant_emotion.upper()}")
        if dominant_emotion in emotions:
            print(f"  Score: {emotions[dominant_emotion]:.2f}%")
    
    # Display all emotions sorted by score
    print("\nAll Emotions (sorted by score):")
    print("-" * 60)
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_emotions:
        bar_length = int(score / 2)  # Scale bar to 50 chars max
        bar = "█" * bar_length
        print(f"  {emotion.upper():12s}: {score:6.2f}% [{bar}]")
    
    # # Display colors
    # print("\nEmotion Colors:")
    # print("-" * 60)
    # print(f"  RGB Color: {format_color_display(emotion_color_rgb)}")
    # print(f"  BGR Color: {format_color_display(emotion_color_bgr)}")
    
    # # Display individual emotion colors
    # print("\nIndividual Emotion Colors:")
    # print("-" * 60)
    emotion_colors_map = {
        'neutral': (255, 255, 255),
        'happy': (255, 215, 0),
        'sad': (0, 0, 255),
        'angry': (255, 0, 0),
        'surprise': (255, 20, 147),
        'fear': (128, 0, 128),
        'disgust': (0, 128, 0),
    }
    
    for emotion in sorted_emotions:
        emotion_name = emotion[0]
        if emotion_name in emotion_colors_map:
            color = emotion_colors_map[emotion_name]
           
    
    print("="*60 + "\n")

def annotate_frame(frame, detection_result):
    """Annotate frame with emotion detection results"""
    if detection_result is None or not detection_result.get('face_detected', False):
        # Display "No face detected" message
        cv2.putText(frame, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    # Draw bounding box
    face_bbox = detection_result.get('face_bbox')
    dominant_emotion = detection_result.get('dominant_emotion')
    emotion_color_bgr = detection_result.get('emotion_color_bgr', (0, 255, 0))
    
    if face_bbox:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color_bgr, 2)
        
        # Add dominant emotion label
        if dominant_emotion:
            text_y = max(y - 10, 30)
            cv2.putText(frame, dominant_emotion.upper(), (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color_bgr, 2)
    
    # Add color overlay on face region
    if face_bbox and emotion_color_bgr:
        x, y, w, h = face_bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size > 0:
            overlay_alpha = 0.35
            color_overlay = frame[y1:y2, x1:x2].copy()
            color_overlay[:] = emotion_color_bgr
            colored_face_roi = cv2.addWeighted(face_roi, 1.0 - overlay_alpha,
                                               color_overlay, overlay_alpha, 0)
            frame[y1:y2, x1:x2] = colored_face_roi
    
    # Add emotion scores on the frame
    emotions = detection_result.get('emotions', {})
    if emotions:
        y_offset = 50
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]  # Top 5
        for i, (emotion, score) in enumerate(sorted_emotions):
            text = f"{emotion}: {score:.1f}%"
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def check_and_configure_gpu():
    """Check GPU availability and configure TensorFlow for GPU usage"""
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - cannot check GPU status")
        print("="*60 + "\n")
        return False
    
    try:
        # 1. Get the list of GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ GPU(s) available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Get GPU details if available
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                if gpu_details:
                    device_name = gpu_details.get('device_name', 'Unknown')
                    print(f"  Device: {device_name}")
            except:
                pass
            
            print(f"TensorFlow version: {tf.__version__}")
            
            # 2. Enable "Memory Growth" for all GPUs
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU Memory Growth Enabled")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"⚠ Warning: Could not set GPU memory growth: {e}")
            
            print("DeepFace should use GPU automatically")
            print("="*60 + "\n")
            return True
        else:
            print("✗ No GPU detected - running on CPU")
            print("TensorFlow will use CPU for emotion detection")
            print("Note: GPU support requires:")
            print("  - TensorFlow with GPU support")
            print("  - CUDA and cuDNN installed")
            print("  - Compatible NVIDIA GPU")
            print("="*60 + "\n")
            return False
    
    except Exception as e:
        print(f"⚠ Error checking GPU: {e}")
        print("Will attempt to run on CPU")
        print("="*60 + "\n")
        return False

def get_windows_host_ip():
    return "172.29.224.1"
    """Get the Windows host IP address from WSL2"""
    try:
        # In WSL2, the Windows host IP is in /etc/resolv.conf
        if os.path.exists('/etc/resolv.conf'):
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        ip = line.split()[1]
                        # Skip localhost
                        if ip != '127.0.0.1':
                            return ip
    except Exception as e:
        print(f"Could not read /etc/resolv.conf: {e}")
    
    # Fallback: try common WSL2 host IPs
    # WSL2 typically uses 172.x.x.1 for Windows host
    try:
        # Try to connect to Windows host via common IPs
        test_ips = ['172.16.0.1', '172.17.0.1', '172.18.0.1', '172.19.0.1', '172.20.0.1']
        for ip in test_ips:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(0.1)
                sock.connect((ip, 1995))
                sock.close()
                return ip
            except:
                continue
    except:
        pass
    
    return None

def main():
    """Main function to run emotion detection demo"""
    print("Initializing emotion detection demo...")
    
    # Check and configure GPU
    check_and_configure_gpu()
    
    # Initialize camera from webcam server
    # Webcam server should be running on port 1995
    # Try multiple connection methods for WSL2 compatibility
    webcam_server_urls = []
    
    # First try localhost (if running in same environment)
    webcam_server_urls.append("http://172.29.224.1:1995/video")
    webcam_server_urls.append("http://localhost:1995/video")
    
    # If in WSL2, try Windows host IP
    windows_host_ip = get_windows_host_ip()
    if windows_host_ip:
        webcam_server_urls.append(f"http://{windows_host_ip}:1995/video")
        print(f"Detected Windows host IP: {windows_host_ip}")
    
    camera = None
    connected_url = None
    
    print("Attempting to connect to webcam server...")
    for url in webcam_server_urls:
        print(f"  Trying: {url}")
        camera = cv2.VideoCapture(url)
        if camera.isOpened():
            # Try to read a frame to verify connection
            ret, frame = camera.read()
            if ret and frame is not None:
                connected_url = url
                print(f"✓ Successfully connected to webcam server at {url}")
                break
            else:
                camera.release()
                camera = None
    
    if not camera or not camera.isOpened():
        print("\n" + "="*60)
        print("ERROR: Could not connect to webcam server!")
        print("="*60)
        print("Tried the following URLs:")
        for url in webcam_server_urls:
            print(f"  - {url}")
        print("\nMake sure:")
        print("  1. webcam_server.py is running (on Windows or WSL2)")
        print("  2. The server is accessible from this environment")
        if windows_host_ip:
            print(f"  3. If running on Windows, try: http://{windows_host_ip}:1995/video")
        print("="*60)
        return
    
    print(f"Using webcam server: {connected_url}")
    
    # Initialize emotion detector
    print("Loading emotion detection models...")
    emotion_detector = EmotionDetector()
    emotion_detector.load_models()
    
    if not emotion_detector.models_loaded:
        print("Error: Could not load emotion detection models")
        camera.release()
        return
    
    print("Starting emotion detection...")
    print("Press 'q' to quit\n")
    
    last_display_time = 0
    display_interval = 0.5  # Update console display every 0.5 seconds
    
    try:
        while True:
            # Read frame from webcam server stream
            ret, frame = camera.read()
            
            if not ret or frame is None:
                print("Warning: Could not read frame from webcam server")
                time.sleep(0.1)
                continue
            
            # Detect emotions
            result = emotion_detector.detect_emotions_from_frame(frame, silent=False)
            
            # Annotate frame with detection results
            annotated_frame = annotate_frame(frame.copy(), result)
            
           
           
            
            # Display emotion data to console (throttled)
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                if result and result.get('face_detected'):
                    emotions = convert_numpy_types(result.get('emotions', {}))
                    dominant_emotion = result.get('dominant_emotion')
                    emotion_color_rgb = result.get('emotion_color_rgb', (0, 0, 0))
                    emotion_color_bgr = result.get('emotion_color_bgr', (0, 0, 0))
                    
                    display_emotion_data(emotions, dominant_emotion, emotion_color_rgb, emotion_color_bgr)
                else:
                    print("\n" + "="*60)
                    print("No face detected")
                    print("="*60 + "\n")
                
                last_display_time = current_time
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to control frame rate
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        if camera is not None:
            camera.release()
        emotion_detector.cleanup()
        cv2.destroyAllWindows()
        print("Demo ended")

if __name__ == "__main__":
    main()

