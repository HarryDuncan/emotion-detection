import cv2
import logging
import os
import threading
import time


DEFAULT_CAMERA_CONFIG = {
    'camera_fps': 30,
    'camera_fps_adaptive': True,
    'camera_fps_min': 15,
    'camera_fps_max': 60,
    'camera_index': 0,
    # Use a URL to read from a remote video feed (e.g. Windows machine serving /video_feed)
    # Example: "http://192.168.1.10:5000/video_feed" — if set, overrides camera_index
    'camera_url': None,
    'camera_flip_horizontal': True,
}

# ============================================================
# Windows ffmpeg commands (run on Windows, NOT in the container)
# ============================================================
#
# RECOMMENDED — MJPEG (~33ms/frame, no accumulating lag):
#   ffmpeg -f dshow -framerate 30 -i video="c922 Pro Stream Webcam" -vcodec mjpeg -q:v 5 -f mpegts udp://127.0.0.1:1235
#
# Each MJPEG frame is an independent JPEG — no inter-frame dependencies,
# no GOP buffer, no decoder lookahead. read() returns every ~33ms reliably.
# -q:v 5 = high quality (2=best, 31=worst).
#
# H.264 (higher compression but ~60ms/frame reads due to GOP/decoder buffer):
#   ffmpeg -f dshow -framerate 30 -i video="c922 Pro Stream Webcam" \
#     -vcodec libx264 -preset ultrafast -tune zerolatency -crf 23 \
#     -x264-params "keyint=15:min-keyint=15:scenecut=0:repeat_headers=1" \
#     -f mpegts "udp://127.0.0.1:1235?pkt_size=1316"
# ============================================================


class CameraInput:
    def __init__(self, input_config):
        logging.info(f"CameraInput initialized with config: {input_config}")
        self.video_capture = None
        self._capture_lock = threading.Lock()  # Protects video_capture from concurrent reads
       
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
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None

        camera_url = self.input_config.get('camera_url')
        camera_index = self.input_config.get('camera_index', 0)

        if camera_url:
            # Read from remote video feed (e.g. http://192.168.1.10:5000/video_feed)
            source = camera_url
            print(f"Opening video feed from URL: {camera_url}")
        else:
            source = camera_index
            print(f"Opening camera device index: {camera_index}")

        try:
            is_udp = isinstance(source, str) and source.startswith("udp://")
            if is_udp:
                # Low-latency FFmpeg options for UDP/H.264 mpegts streams.
                # fflags=nobuffer+flush_packets : no demuxer buffering
                # flags=low_delay               : disable B-frame reorder delay
                # probesize=32 / analyzeduration=0 : skip lengthy stream analysis
                # max_delay=0                   : no packet reorder buffer
                # threads=1 / thread_type=slice : single-threaded decode —
                #   prevents 'Assertion fctx->async_lock failed' crashes
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "fflags;nobuffer+flush_packets"
                    "|flags;low_delay"
                    "|probesize;32"
                    "|analyzeduration;0"
                    "|max_delay;0"
                    "|thread_type;slice"
                    "|threads;1"
                )
                self.video_capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.video_capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 440)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
        except Exception as e:
            print(f"Could not open video source {source!r}: {e}")
            return False

        if not self.video_capture.isOpened():
            print("Failed to open video capture (source: {}).".format(source))
            self.video_capture.release()
            self.video_capture = None
            return False

        # Log properties (FPS may be 0 or unreliable for stream URLs)
        actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Video source opened. FPS: {actual_fps}, Target FPS: {self.target_fps}")

        print("Camera initialized successfully")
        return True
    
    def _read_with_frame_rate_limiting(self):
        """Read frames with time-based rate limiting (thread-safe)."""
        if self.video_capture is None:
            return False, None
        current_time = time.time()

        with self._capture_lock:
            # Check if enough time has passed since last frame
            if current_time - self.last_frame_time >= self.frame_interval:
                ret, frame = self.video_capture.read()
                logging.debug(f"Reading new frame: ret={ret}, frame_shape={frame.shape if ret and frame is not None else 'None'}")
                if ret:
                    self.last_frame = frame
                    self.last_frame_valid = True
                    self.last_frame_time = current_time
                    return ret, frame
                else:
                    if self.last_frame_valid:
                        logging.debug("Using previous frame due to read failure")
                        return True, self.last_frame
                    logging.warning("No frame available and no previous frame")
                    return False, None
            else:
                if self.last_frame_valid:
                    logging.debug("Returning cached frame (rate limiting)")
                    return True, self.last_frame
                else:
                    ret, frame = self.video_capture.read()
                    logging.debug(f"Reading frame (no cache): ret={ret}, frame_shape={frame.shape if ret and frame is not None else 'None'}")
                    if ret:
                        self.last_frame = frame
                        self.last_frame_valid = True
                        self.last_frame_time = current_time
                    return ret, frame
    
    def get_frame_width(self):
        if self.video_capture is None:
            return 0
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        if self.video_capture is None:
            return 0
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def isOpened(self):
        return self.video_capture is not None and self.video_capture.isOpened()
    
    def read_latest(self):
        """Read one frame directly from the capture with no rate limiting or caching.

        Does NOT hold _capture_lock during the blocking read() call so other
        threads (isOpened checks, release) are not blocked while waiting for
        the next frame from the network.  The dedicated camera-reader thread
        is the ONLY caller, so there is no concurrent read() race.
        """
        cap = self.video_capture
        if cap is None:
            return False, None
        return cap.read()

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
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None
    
    def get_video_capture(self):
        return self.video_capture