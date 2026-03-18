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

from emotion_transforms import EMOTION_COLORS_RGB as _EMOTION_COLORS_RGB

# BGR version of the per-emotion palette (used for OpenCV drawing)
_EMOTION_COLORS_BGR: dict[str, tuple] = {
    name: (rgb[2], rgb[1], rgb[0])
    for name, rgb in _EMOTION_COLORS_RGB.items()
}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICK = 1

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


def _top3_lines(face: dict) -> list[tuple[str, tuple]]:
    """Return [(label_str, bgr_color), ...] for the top-3 emotions by score.

    Each label is formatted as ``"happy  45%"`` and carries the individual
    emotion's color so rows can be visually distinguished at a glance.
    """
    emotions     = face.get('emotions') or {}
    default_bgr  = face.get('emotion_color_bgr', (128, 128, 128))
    top3         = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [
        (f"{name}  {score:.0f}%", _EMOTION_COLORS_BGR.get(name.lower(), default_bgr))
        for name, score in top3
    ]


def _text_color(bgr: tuple, *, alpha: int | None = None) -> tuple:
    """Black or white text depending on background luminance; append alpha if given."""
    b, g, r = bgr[:3]
    luma    = 0.299 * r + 0.587 * g + 0.114 * b
    base    = (0, 0, 0) if luma > 140 else (255, 255, 255)
    return (*base, alpha) if alpha is not None else base


def _draw_top3(canvas: np.ndarray, face: dict) -> None:
    """Draw bounding box and top-3 emotion labels onto *canvas* in-place.

    Handles both 3-channel BGR frames and 4-channel BGRA transparent canvases.
    Each of the three label rows uses the individual emotion's palette color so
    the blend is visible at a glance. Rows are stacked immediately above the
    bounding box, ordered 1st (top) → 3rd (bottom).
    """
    n_ch       = canvas.shape[2]
    alpha      = 255 if n_ch == 4 else None   # None → omit alpha component

    def _c(bgr):
        return (*bgr, alpha) if alpha is not None else bgr

    x, y, w, h  = face['face_bbox']
    bbox_color  = face['emotion_color_bgr']

    cv2.rectangle(canvas, (x, y), (x + w, y + h), _c(bbox_color), 2)

    lines     = _top3_lines(face)
    cursor_y  = y - 2           # baseline just above the top edge of the bbox

    for label, bgr in reversed(lines):   # draw 3rd → 1st so 1st ends up on top
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        top = max(cursor_y - th - 4, 0)
        cv2.rectangle(canvas, (x, top), (x + tw + 6, cursor_y + 2),
                      _c(bgr), cv2.FILLED)
        cv2.putText(canvas, label, (x + 3, cursor_y - 1),
                    _FONT, _FONT_SCALE, _text_color(bgr, alpha=alpha),
                    _FONT_THICK, cv2.LINE_AA)
        cursor_y = top - 2      # 2 px gap between rows


def annotate_data_layer(result: dict, height: int, width: int) -> np.ndarray:
    """Return a transparent BGRA canvas with only the annotation layer drawn.

    All pixels start fully transparent (alpha=0). Drawn shapes (bounding boxes,
    label backgrounds, text) are fully opaque (alpha=255). Encode the returned
    array as PNG to preserve the alpha channel — JPEG cannot represent transparency.

    Each face shows the top-3 emotions by score, each row colored with its own
    emotion palette color, stacked above the bounding box.

    Parameters
    ----------
    result:
        The latest emotion result dict from ``_state.latest_emotion_result``.
    height, width:
        Canvas dimensions — should match the video frame for correct compositing.
    """
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for face in result.get('faces', []):
        _draw_top3(canvas, face)
    return canvas


def annotate_frame(frame: np.ndarray, result: dict) -> None:
    """Draw per-face top-3 emotion labels and bounding boxes onto *frame* in-place.

    Parameters
    ----------
    frame:
        BGR numpy array — modified in-place.
    result:
        The latest emotion result dict from ``_state.latest_emotion_result``.
    """
    for face in result.get('faces', []):
        _draw_top3(frame, face)
