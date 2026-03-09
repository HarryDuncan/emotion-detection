import cv2
import os


DEFAULT_CAMERA_CONFIG = {
    'camera_fps': 30,
    'camera_index': 0,
    'camera_url': None,      # If set, overrides camera_index (e.g. "udp://@0.0.0.0:1235")
    'camera_width': 440,
    'camera_height': 380,
}

# ============================================================
# Windows ffmpeg commands (run on Windows, NOT in the container)
# ============================================================
#
# RECOMMENDED — MJPEG (~33ms/frame, no accumulating lag):
#   ffmpeg -f dshow -framerate 30 -i video="c922 Pro Stream Webcam" \
#     -vcodec mjpeg -q:v 5 -f mpegts udp://127.0.0.1:1235
#
# H.264 (higher compression but ~60ms/frame reads due to GOP/decoder buffer):
#   ffmpeg -f dshow -framerate 30 -i video="c922 Pro Stream Webcam" \
#     -vcodec libx264 -preset ultrafast -tune zerolatency -crf 23 \
#     -x264-params "keyint=15:min-keyint=15:scenecut=0:repeat_headers=1" \
#     -f mpegts "udp://127.0.0.1:1235?pkt_size=1316"
# ============================================================


class CameraInput:
    """
    Thin wrapper around cv2.VideoCapture.

    Owned exclusively by the camera-reader thread in appv2.py — only one
    thread ever calls read_latest(), so no internal locking is needed.
    """

    def __init__(self, input_config=None):
        self.video_capture = None
        self.config = {**DEFAULT_CAMERA_CONFIG, **(input_config or {})}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_camera(self):
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None

        camera_url = self.config.get('camera_url')
        camera_index = self.config.get('camera_index', 0)
        fps = self.config.get('camera_fps', 30)
        width = self.config.get('camera_width', 440)
        height = self.config.get('camera_height', 380)

        if camera_url:
            self._open_url(camera_url)
        else:
            self._open_device(camera_index, width, height, fps)

        if not self.isOpened():
            print("Failed to open video capture.")
            return False

        actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera ready: {w}x{h} @ {actual_fps:.0f}fps  codec={fourcc_str!r}")
        return True

    def _open_device(self, index, width, height, fps):
        print(f"Opening camera device index {index}")
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

        # MJPG: each frame is an independent JPEG — low USB bandwidth, fast
        # delivery.  Must be set before width/height/fps or the driver may
        # silently ignore it and fall back to a raw format.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Buffer of 1 keeps us on the most-recent frame and avoids the driver
        # queue building up latency when the consumer is briefly slower.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if cap.isOpened():
            self.video_capture = cap
        else:
            cap.release()

    def _open_url(self, url):
        print(f"Opening video feed from URL: {url}")
        is_udp = url.startswith("udp://")
        if is_udp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "fflags;nobuffer+flush_packets"
                "|flags;low_delay"
                "|probesize;32"
                "|analyzeduration;0"
                "|max_delay;0"
                "|thread_type;slice"
                "|threads;1"
            )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self.video_capture = cap
        else:
            cap.release()

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def read_latest(self):
        """
        Read the next frame from the capture device.

        Blocking call — returns when the driver delivers a complete frame.
        The camera-reader thread in appv2.py is the ONLY caller; no locking
        is needed here.

        Frames with obvious decode failures (zero-size or empty array) are
        discarded and reported as ret=False so the caller never publishes a
        garbled frame to the stream.
        """
        if self.video_capture is None:
            return False, None
        ret, frame = self.video_capture.read()
        if not ret or frame is None or frame.size == 0:
            return False, None
        return True, frame

    def isOpened(self):
        return self.video_capture is not None and self.video_capture.isOpened()

    def release(self):
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None
