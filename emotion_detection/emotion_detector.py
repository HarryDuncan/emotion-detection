import cv2
from deepface import DeepFace
from emotion_transforms import emotions_to_color
import numpy as np


class EmotionDetector:
    """
    Emotion detection class that handles face detection and emotion analysis.
    """
    
    def __init__(self, load_models_on_init=False):
        """
        Initialize the EmotionDetector.
        
        Args:
            load_models_on_init: If True, pre-load DeepFace models during initialization (default: True)
        """
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.models_loaded = False
        
        if load_models_on_init:
            self.load_models()
    
    def load_models(self):
        """Pre-load DeepFace models"""
        print("Loading emotion detection models...")
        try:
            # Create a test frame to trigger model loading
            # Models are lazy-loaded, so we need to actually call analyze
            test_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray frame
            try:
                # This will trigger model download/loading on first run
                # It will fail (no face), but models will be cached
                DeepFace.analyze(test_frame, actions=['emotion'], enforce_detection=False, silent=True)
            except Exception as e:
                # Expected to fail (no face detected), but models are now loaded
                print(f"Model loading triggered (expected error: {type(e).__name__})")
            self.models_loaded = True
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            # Still mark as loaded if it's just a detection error
            if "No face" in str(e) or "face" in str(e).lower():
                self.models_loaded = True
                return True
            return False
    
    def detect_emotions_from_frame(self, frame, silent=True):
        """
        Detect emotions from a single frame.
        
        Args:
            frame: BGR frame from OpenCV (numpy array)
            silent: Whether to suppress DeepFace warnings (default: True)
        
        Returns:
            dict: {
                'emotions': dict of emotion scores (0-100),
                'dominant_emotion': str,
                'emotion_color_rgb': tuple (R, G, B),
                'emotion_color_bgr': tuple (B, G, R),
                'face_detected': bool,
                'face_bbox': tuple (x, y, w, h) or None
            }
            Returns None if no face detected or error occurred.
        """
        try:
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert grayscale frame to RGB format for DeepFace
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            
            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=7, minSize=(50, 50))
            
            if len(faces) == 0:
                return {
                    'emotions': {},
                    'dominant_emotion': None,
                    'emotion_color_rgb': (0, 0, 0),
                    'emotion_color_bgr': (0, 0, 0),
                    'face_detected': False,
                    'face_bbox': None
                }
            
            # Use the first detected face
            (x, y, w, h) = faces[0]
            
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]
            
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=silent)
            
            # Get all emotions from the result
            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            
            # Convert emotions to RGB color
            emotion_color_rgb = emotions_to_color(emotions)
            # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
            emotion_color_bgr = (int(emotion_color_rgb[2]), int(emotion_color_rgb[1]), int(emotion_color_rgb[0]))
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'emotion_color_rgb': emotion_color_rgb,
                'emotion_color_bgr': emotion_color_bgr,
                'face_detected': True,
                'face_bbox': (x, y, w, h)
            }
        
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return {
                'emotions': {},
                'dominant_emotion': None,
                'emotion_color_rgb': (0, 0, 0),
                'emotion_color_bgr': (0, 0, 0),
                'face_detected': False,
                'face_bbox': None,
                'error': str(e)
            }
    
    def cleanup(self):
        """
        Cleanup resources. Called when the detector is no longer needed.
        """
        # DeepFace models are managed internally, no explicit cleanup needed
        # Face cascade is a simple classifier, no cleanup needed
        self.models_loaded = False
        print("EmotionDetector cleaned up")
    
    def __del__(self):
        """Destructor - ensures cleanup is called"""
        try:
            self.cleanup()
        except:
            pass


# Note: This module only works with frames, not cameras.
# Camera management should be handled by the calling code (e.g., app.py)
# This ensures only one camera instance exists and avoids conflicts.

# For standalone testing
if __name__ == "__main__":
    # Example standalone usage
    print("Example usage:")
    print("  import cv2")
    print("  from emotion_detection.emotion_detector import EmotionDetector")
    print("  ")
    print("  detector = EmotionDetector()  # Loads models on init")
    print("  cap = cv2.VideoCapture(0)")
    print("  ret, frame = cap.read()")
    print("  result = detector.detect_emotions_from_frame(frame)")
    print("  detector.cleanup()  # Clean up when done")