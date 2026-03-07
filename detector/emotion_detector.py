import cv2
from deepface import DeepFace
from emotion_transforms import emotions_to_color
import numpy as np


class EmotionDetector:
    """
    Emotion detection class that handles face detection and emotion analysis.
    """
    
    def __init__(self, load_models_on_init=False, enable_preprocessing=True, enable_alignment=False, fast_mode=False):
        """
        Initialize the EmotionDetector.
        
        Args:
            load_models_on_init: If True, pre-load DeepFace models during initialization (default: False)
            enable_preprocessing: If True, apply CLAHE preprocessing for better contrast (default: True)
            enable_alignment: If True, enable face alignment (adds overhead but improves accuracy) (default: False)
            fast_mode: If True, enables fast mode (disables preprocessing, alignment, and frame skipping) (default: False)
        """
        self.models_loaded = False
        
        # Handle fast_mode - overrides other settings
        if fast_mode:
            self.enable_preprocessing = False
            self.enable_alignment = False
            self.frame_skip = 1  # Skip every 2nd frame (process every other frame)
        else:
            self.enable_preprocessing = enable_preprocessing
            self.enable_alignment = enable_alignment
            self.frame_skip = 0  # Process all frames by default
        
        # Cache CLAHE object if preprocessing is enabled
        self.clahe = None
        if self.enable_preprocessing:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Frame counter for skipping logic
        self.frame_counter = 0
        
        # Store last result for frame skipping
        self.last_result = None
        
        if load_models_on_init:
            self.load_models()
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better emotion detection using CLAHE histogram equalization.
        
        Args:
            frame: BGR frame from OpenCV (numpy array)
        
        Returns:
            Preprocessed RGB frame (or original if preprocessing disabled)
        """
        # Early return if preprocessing is disabled
        if not self.enable_preprocessing:
            # Still convert BGR to RGB for DeepFace
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel (lightness) using cached object
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert LAB back to RGB
        rgb_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return rgb_frame
    
    def _should_process_frame(self, frame_skip=None):
        """
        Determine if current frame should be processed based on frame_skip setting.
        
        Args:
            frame_skip: Frame skip value to use (if None, uses self.frame_skip)
        
        Returns:
            bool: True if frame should be processed, False if it should be skipped
        """
        skip_value = frame_skip if frame_skip is not None else self.frame_skip
        
        if skip_value == 0:
            return True
        
        # Process frame if counter is divisible by (frame_skip + 1)
        # frame_skip=1 means process every 2nd frame (process frame 0, skip 1, process 2, skip 3...)
        should_process = (self.frame_counter % (skip_value + 1)) == 0
        self.frame_counter += 1
        return should_process
    
    def _downscale_frame_if_needed(self, frame, max_width=640):
        """
        Downscale frame if it's larger than max_width to improve processing speed.
        
        Args:
            frame: Input frame
            max_width: Maximum width before downscaling (default: 640)
        
        Returns:
            tuple: (scaled_frame, scale_factor) where scale_factor is used to scale bbox back
        """
        height, width = frame.shape[:2]
        if width > max_width:
            scale_factor = max_width / width
            new_width = max_width
            new_height = int(height * scale_factor)
            scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            return scaled_frame, scale_factor
        return frame, 1.0
    
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
                # Try mediapipe first, fallback to mtcnn
                try:
                    DeepFace.analyze(
                        test_frame, 
                        actions=['emotion'], 
                        enforce_detection=False, 
                        silent=True,
                        detector_backend='mediapipe',
                        align=self.enable_alignment
                    )
                except:
                    # Fallback to mtcnn if mediapipe is not available
                    DeepFace.analyze(
                        test_frame, 
                        actions=['emotion'], 
                        enforce_detection=False, 
                        silent=True,
                        detector_backend='mtcnn',
                        align=self.enable_alignment
                    )
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
    
    def detect_emotions_from_frame(self, frame, silent=True, detector_backend='mediapipe', 
                                    enable_preprocessing=None, enable_alignment=None, frame_skip=None):
        """
        Detect emotions from a single frame.
        
        Args:
            frame: BGR frame from OpenCV (numpy array)
            silent: Whether to suppress DeepFace warnings (default: True)
            detector_backend: Face detector backend ('mediapipe', 'mtcnn', etc.) (default: 'mediapipe')
            enable_preprocessing: Override instance preprocessing setting (default: None = use instance setting)
            enable_alignment: Override instance alignment setting (default: None = use instance setting)
            frame_skip: Override instance frame_skip setting (default: None = use instance setting)
        
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
            # Use instance settings unless overridden
            use_preprocessing = enable_preprocessing if enable_preprocessing is not None else self.enable_preprocessing
            use_alignment = enable_alignment if enable_alignment is not None else self.enable_alignment
            use_frame_skip = frame_skip if frame_skip is not None else self.frame_skip
            
            # Check if we should skip this frame
            if use_frame_skip > 0:
                if not self._should_process_frame(use_frame_skip):
                    # Return last result if available
                    if self.last_result is not None:
                        return self.last_result
                    # Otherwise return no face detected
                    return {
                        'emotions': {},
                        'dominant_emotion': None,
                        'emotion_color_rgb': (0, 0, 0),
                        'emotion_color_bgr': (0, 0, 0),
                        'face_detected': False,
                        'face_bbox': None
                    }
            
            # Downscale frame if it's too large (for speed)
            scaled_frame, scale_factor = self._downscale_frame_if_needed(frame)
            
            # Preprocess frame for better contrast (if enabled)
            if use_preprocessing:
                preprocessed_frame = self.preprocess_frame(scaled_frame)
            else:
                # Still convert BGR to RGB for DeepFace
                preprocessed_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            
            # Try mediapipe first, fallback to mtcnn if mediapipe fails
            result = None
            used_backend = detector_backend
            
            try:
                # Use DeepFace's built-in face detection with optional alignment
                result = DeepFace.analyze(
                    preprocessed_frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=silent,
                    detector_backend=detector_backend,
                    align=use_alignment
                )
            except Exception as e:
                # Fallback to mtcnn if mediapipe is not available
                if detector_backend == 'mediapipe':
                    try:
                        used_backend = 'mtcnn'
                        result = DeepFace.analyze(
                            preprocessed_frame,
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=silent,
                            detector_backend='mtcnn',
                            align=use_alignment
                        )
                    except Exception as fallback_error:
                        # If both fail, return no face detected
                        print(f"Error in emotion detection with both backends: {fallback_error}")
                        no_face_result = {
                            'emotions': {},
                            'dominant_emotion': None,
                            'emotion_color_rgb': (0, 0, 0),
                            'emotion_color_bgr': (0, 0, 0),
                            'face_detected': False,
                            'face_bbox': None
                        }
                        self.last_result = no_face_result
                        return no_face_result
                else:
                    # If mtcnn was requested and failed, return no face detected
                    print(f"Error in emotion detection: {e}")
                    no_face_result = {
                        'emotions': {},
                        'dominant_emotion': None,
                        'emotion_color_rgb': (0, 0, 0),
                        'emotion_color_bgr': (0, 0, 0),
                        'face_detected': False,
                        'face_bbox': None
                    }
                    self.last_result = no_face_result
                    return no_face_result
            
            # Check if result is valid and contains emotion data
            if not result or len(result) == 0:
                no_face_result = {
                    'emotions': {},
                    'dominant_emotion': None,
                    'emotion_color_rgb': (0, 0, 0),
                    'emotion_color_bgr': (0, 0, 0),
                    'face_detected': False,
                    'face_bbox': None
                }
                self.last_result = no_face_result
                return no_face_result
            
            # Get all emotions from the result
            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            
            # Extract face bounding box from DeepFace result if available
            face_bbox = None
            # DeepFace returns 'region' not 'facial_areas'
            if 'region' in result[0] and result[0]['region']:
                # DeepFace returns region as a dict with 'x', 'y', 'w', 'h'
                region = result[0]['region']
                if isinstance(region, dict):
                    # Scale bbox back if frame was downscaled
                    x = int(region.get('x', 0) / scale_factor)
                    y = int(region.get('y', 0) / scale_factor)
                    w = int(region.get('w', 0) / scale_factor)
                    h = int(region.get('h', 0) / scale_factor)
                    face_bbox = (x, y, w, h)
            
            # Convert emotions to RGB color
            emotion_color_rgb = emotions_to_color(emotions)
            # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
            emotion_color_bgr = (int(emotion_color_rgb[2]), int(emotion_color_rgb[1]), int(emotion_color_rgb[0]))
            
            result_dict = {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'emotion_color_rgb': emotion_color_rgb,
                'emotion_color_bgr': emotion_color_bgr,
                'face_detected': True,
                'face_bbox': face_bbox
            }
            
            # Store result for frame skipping
            self.last_result = result_dict
            
            return result_dict
        
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            error_result = {
                'emotions': {},
                'dominant_emotion': None,
                'emotion_color_rgb': (0, 0, 0),
                'emotion_color_bgr': (0, 0, 0),
                'face_detected': False,
                'face_bbox': None,
                'error': str(e)
            }
            self.last_result = error_result
            return error_result
    
    def cleanup(self):
        """
        Cleanup resources. Called when the detector is no longer needed.
        """
        # DeepFace models are managed internally, no explicit cleanup needed
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