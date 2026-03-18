"""
Shared frame utilities — background removal and annotation.

Used by both the MJPEG HTTP route and the /ws/video WebSocket endpoint so that
rendering/processing code lives in exactly one place.

Background removal
------------------
Uses the MediaPipe Tasks API (ImageSegmenter) with the selfie-segmenter
landscape model (~1 MB TFLite).  The model is downloaded once on first use
and cached next to this file.

GPU path:  TFLite GPU delegate (OpenCL / OpenGL ES via EGL).
           Set EGL_PLATFORM=device in the environment for headless NVIDIA GPU.
CPU path:  automatic fallback if GPU delegate init fails.

NOTE: mediapipe must be imported before matplotlib/DeepFace to avoid a fatal
pybind11 ft2font conflict.  The module-level import below ensures this.
"""
import os
import threading
import urllib.request

import cv2
import mediapipe as mp                          # must stay at module level
import numpy as np
from mediapipe.tasks import python as _mp_py
from mediapipe.tasks.python import vision as _mp_vision

_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'image_segmenter/selfie_segmenter_landscape/float16/latest/'
    'selfie_segmenter_landscape.tflite'
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'selfie_segmenter.tflite')


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f'[bg] Downloading selfie segmenter model → {_MODEL_PATH} …')
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print('[bg] Model download complete.')
    return _MODEL_PATH


def _build_segmenter() -> _mp_vision.ImageSegmenter:
    """Try GPU delegate; fall back to CPU if unavailable."""
    model_path = _ensure_model()
    for label, delegate in [
        ('GPU', _mp_py.BaseOptions.Delegate.GPU),
        ('CPU', _mp_py.BaseOptions.Delegate.CPU),
    ]:
        try:
            opts = _mp_vision.ImageSegmenterOptions(
                base_options=_mp_py.BaseOptions(
                    model_asset_path=model_path,
                    delegate=delegate,
                ),
                output_confidence_masks=True,
            )
            seg = _mp_vision.ImageSegmenter.create_from_options(opts)
            print(f'[bg] MediaPipe ImageSegmenter ready on {label}')
            return seg
        except Exception as exc:
            print(f'[bg] {label} delegate failed ({exc}), trying next…')
    raise RuntimeError('[bg] Could not initialise MediaPipe segmenter on any delegate')


# ---------------------------------------------------------------------------
# Background remover — lazy singleton
# ---------------------------------------------------------------------------

_remover_instance: 'BackgroundRemover | None' = None
_remover_lock = threading.Lock()


class BackgroundRemover:
    """
    MediaPipe selfie-segmenter background remover.

    ``remove(frame)`` returns a BGR frame with the background replaced by
    ``bg_color`` (default solid black).

    ``threshold`` (0.0–1.0) sets the minimum foreground confidence required
    to keep a pixel.  0.5 is a good default; raise it for a tighter mask.
    """

    def __init__(self, threshold: float = 0.5):
        self._seg      = _build_segmenter()
        self.threshold = threshold

    def remove(self, frame: np.ndarray, bg_color: tuple = (0, 0, 0)) -> np.ndarray:
        """Return a copy of *frame* with background replaced by *bg_color*."""
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._seg.segment(mp_image)
        # confidence_masks[0] = person foreground confidence (0.0–1.0).
        # squeeze() ensures 2D regardless of whether MediaPipe returns (H,W) or (H,W,1).
        conf     = np.squeeze(result.confidence_masks[0].numpy_view())
        mask     = (conf > self.threshold).astype(np.uint8)[:, :, np.newaxis]
        bg       = np.full_like(frame, bg_color)
        return (frame * mask + bg * (1 - mask)).astype(np.uint8)

    def close(self):
        self._seg.close()


def get_background_remover() -> BackgroundRemover:
    """Return the process-wide BackgroundRemover, creating it on first call."""
    global _remover_instance
    if _remover_instance is None:
        with _remover_lock:
            if _remover_instance is None:
                _remover_instance = BackgroundRemover()
    return _remover_instance


def annotate_frame(frame, result: dict) -> None:
    """Draw per-face emotion bounding boxes and labels onto *frame* in-place.

    Parameters
    ----------
    frame:
        BGR numpy array — modified in-place.
    result:
        The latest emotion result dict from ``_state.latest_emotion_result``.
        Expected keys per face entry: ``face_bbox``, ``emotion_color_bgr``,
        ``dominant_emotion``.
    """
    for face in result.get('faces', []):
        x, y, w, h = face['face_bbox']
        color       = face['emotion_color_bgr']
        label       = face.get('dominant_emotion', '')

        # Face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty          = max(y - 6, th + 4)
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 2), color, cv2.FILLED)

        luma        = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color  = (0, 0, 0) if luma > 140 else (255, 255, 255)
        cv2.putText(frame, label, (x + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
