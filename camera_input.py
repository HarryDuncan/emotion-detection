import os
import numpy as np

# GStreamer is imported LAZILY inside _initialize_camera() to avoid loading
# libgobject-2.0 before TensorFlow does.  If gi/Gst are loaded at module
# import time they initialise GLib's GObject type system, which then conflicts
# with TensorFlow's bundled GLib when CUDA is brought up → SIGSEGV (exit 139).
_Gst = None  # populated on first call to _initialize_camera()


DEFAULT_CAMERA_CONFIG = {
    'camera_fps': 15,  # 15fps keeps USB bandwidth low over USBIPD — reduces tearing from packet drops
    'camera_index': 0,
    'camera_url': None,      # If set, overrides camera_index (e.g. "udp://@0.0.0.0:1235")
    'camera_width': 640,
    'camera_height': 480,
}


class CameraInput:
    """
    Camera capture backed by GStreamer.

    The MJPG path:
        v4l2src → image/jpeg caps → jpegdec → videoconvert → BGR appsink

    This avoids the V4L2 raw-YUYV path that caused 256–448ms reads, and the
    corrupt-JPEG warnings produced by OpenCV's libjpeg when frames arrived
    over USBIPD with dropped packets.  GStreamer's jpegdec handles partial
    JPEG data more gracefully and delivers fully decoded BGR frames directly.

    The appsink is configured for pull mode (emit-signals=false) with a
    max-buffer queue of 1 and drop=true, so pull-sample() always returns the
    most recent complete frame and naturally blocks at the camera's FPS.
    """

    def __init__(self, input_config=None):
        self._pipeline = None
        self._sink = None
        self._opened = False
        self.config = {**DEFAULT_CAMERA_CONFIG, **(input_config or {})}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gst():
        """Import and initialise GStreamer on the first call (lazy load)."""
        global _Gst
        if _Gst is None:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst as _GstModule
            _GstModule.init(None)
            _Gst = _GstModule
        return _Gst

    def _initialize_camera(self):
        self._teardown()

        Gst = self._ensure_gst()

        camera_url = self.config.get('camera_url')
        index  = self.config.get('camera_index', 0)
        fps    = self.config.get('camera_fps', 30)
        width  = self.config.get('camera_width', 640)
        height = self.config.get('camera_height', 480)

        if camera_url:
            pipeline_str = self._url_pipeline(camera_url)
        else:
            pipeline_str = self._device_pipeline(index, width, height, fps)

        print(f"[camera] GStreamer pipeline: {pipeline_str}")
        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
            self._sink = self._pipeline.get_by_name('sink')

            change = self._pipeline.set_state(Gst.State.PLAYING)
            if change == Gst.StateChangeReturn.FAILURE:
                print("[camera] GStreamer pipeline failed to start")
                self._teardown()
                return False

            # Block until the pipeline is actually PLAYING (up to 5 s)
            _, state, _ = self._pipeline.get_state(timeout=5 * Gst.SECOND)
            if state != Gst.State.PLAYING:
                print(f"[camera] GStreamer pipeline stalled in state {state.value_nick}")
                self._teardown()
                return False

            self._opened = True
            print(f"[camera] GStreamer ready: {width}x{height} @ {fps}fps  device=/dev/video{index}")
            return True

        except Exception as e:
            print(f"[camera] GStreamer init error: {e}")
            self._teardown()
            return False

    # ------------------------------------------------------------------
    # Pipeline strings
    # ------------------------------------------------------------------

    def _device_pipeline(self, index, width, height, fps):
        """MJPG from a local V4L2 webcam.

        The queue after v4l2src decouples USB interrupt transfers from the
        decode/convert stages, preventing the driver from stalling while
        downstream elements are busy — which is the other cause of tearing
        over USBIPD.  leaky=downstream drops the oldest queued frame if the
        queue fills up, keeping latency low instead of blocking the source.
        """
        device = f"/dev/video{index}"
        return (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"jpegdec ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    def _url_pipeline(self, url):
        """UDP MPEG-TS or HTTP MJPEG stream."""
        if url.startswith("udp://"):
            port = url.rsplit(":", 1)[-1] if ":" in url else "1235"
            return (
                f"udpsrc port={port} ! "
                f"tsdemux ! h264parse ! avdec_h264 max-threads=1 ! "
                f"videoconvert ! "
                f"video/x-raw,format=BGR ! "
                f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
            )
        return (
            f"souphttpsrc location={url} ! "
            f"multipartdemux ! jpegdec ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def read_latest(self):
        """
        Block until the next frame arrives, then return it as a BGR numpy array.

        The appsink's max-buffers=1 / drop=true configuration means:
        - If the consumer (camera-reader thread) is briefly slow, old frames
          are discarded and this call returns the most recent one.
        - The call blocks naturally at the camera's capture FPS — no
          time.sleep() needed in the caller.
        """
        if not self._opened or self._sink is None:
            return False, None

        sample = self._sink.emit('pull-sample')
        if sample is None:
            return False, None

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        s    = caps.get_structure(0)
        h    = s.get_value('height')
        w    = s.get_value('width')

        ok, map_info = buf.map(_Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            arr    = np.frombuffer(map_info.data, dtype=np.uint8)
            stride = map_info.size // h  # actual bytes per row (may include padding)
            if stride != w * 3:
                # Row padding present — strip it before reshaping to avoid tearing.
                frame = np.ascontiguousarray(
                    arr.reshape(h, stride)[:, : w * 3].reshape(h, w, 3)
                )
            else:
                frame = arr.reshape(h, w, 3).copy()
        except Exception:
            return False, None
        finally:
            buf.unmap(map_info)

        if frame.size == 0:
            return False, None
        return True, frame

    def isOpened(self):
        return self._opened

    def release(self):
        self._teardown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _teardown(self):
        if self._pipeline is not None:
            if _Gst is not None:
                self._pipeline.set_state(_Gst.State.NULL)
            self._pipeline = None
        self._sink = None
        self._opened = False
