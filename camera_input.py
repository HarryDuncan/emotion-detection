import cv2
import logging
import time


DEFAULT_CAMERA_CONFIG = {
    'camera_fps': 30,
    'camera_fps_adaptive': True,
    'camera_fps_min': 15,
    'camera_fps_max': 60,
    'camera_index': 0,
    'camera_flip_horizontal': True,  
}

class CameraInput:
    def __init__(self, input_config):
        logging.info(f"CameraInput initialized with config: {input_config}")
        self.video_capture = None
       
        self.input_config = {**DEFAULT_CAMERA_CONFIG, **(input_config or {})}
        
        # Frame rate limiting variables
        self.target_fps = self.input_config.get('camera_fps', 30)
        self.last_frame = None
        self.last_frame_valid = False
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.target_fps  # Time between frames in seconds
        
        logging.info(f"Frame rate limiting enabled: target FPS {self.target_fps}, interval {self.frame_interval:.3f}s")

    def _initialize_camera(self):
        print("Initializing camera")
        if self.video_capture is not None:
            self.video_capture.release()
        else:
            self.video_capture = cv2.VideoCapture(0)

        if not self.video_capture.isOpened():
            print("Failed to open video capture.")
            return False
        
        # Log actual camera properties for verification
        actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Camera hardware FPS: {actual_fps}, Target FPS: {self.target_fps}")
        
        print("Camera initialized successfully")
    
    def _read_with_frame_rate_limiting(self):
        """Read frames with time-based rate limiting"""
        current_time = time.time()
        
        # Check if enough time has passed since last frame
        if current_time - self.last_frame_time >= self.frame_interval:
            # Time to read a new frame
            ret, frame = self.video_capture.read()
            logging.debug(f"Reading new frame: ret={ret}, frame_shape={frame.shape if ret and frame is not None else 'None'}")
            if ret:
                self.last_frame = frame
                self.last_frame_valid = True
                self.last_frame_time = current_time
                return ret, frame
            else:
                # If reading failed but we have a previous frame, return it
                if self.last_frame_valid:
                    logging.debug("Using previous frame due to read failure")
                    return True, self.last_frame
                logging.warning("No frame available and no previous frame")
                return False, None
        else:
            # Not enough time has passed, return the last valid frame
            if self.last_frame_valid:
                logging.debug("Returning cached frame (rate limiting)")
                return True, self.last_frame
            else:
                # No previous frame available, read a new one
                ret, frame = self.video_capture.read()
                logging.debug(f"Reading frame (no cache): ret={ret}, frame_shape={frame.shape if ret and frame is not None else 'None'}")
                if ret:
                    self.last_frame = frame
                    self.last_frame_valid = True
                    self.last_frame_time = current_time
                return ret, frame
    
    def get_frame_width(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def get_frame_height(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def isOpened(self):
        return self.video_capture.isOpened()
    
    def read(self):
        """Read a frame with frame rate limiting"""
        return self._read_with_frame_rate_limiting()
    
    def read_flipped(self):
        """Read a frame with frame rate limiting and return it flipped horizontally for mirror effect if configured"""
        ret, frame = self._read_with_frame_rate_limiting()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)  # Flip horizontally
        return ret, frame
    
    def release(self):
        self.video_capture.release()
    
    def get_video_capture(self):
        return self.video_capture