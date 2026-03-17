"""
Shared frame utilities — background removal and annotation.

Used by both the MJPEG HTTP route and the /ws/video WebSocket endpoint so that
rendering/processing code lives in exactly one place.

Background removal
------------------
Uses OpenCV MOG2 (Mixture of Gaussians) background subtraction.
Best for static/near-static cameras — learns the background over ~500 frames
then reliably isolates the foreground subject.
No extra dependencies; runs entirely on CPU via opencv-python-headless.
"""
import threading

import cv2
import numpy as np

# Morphological kernel for mask cleanup (removes speckling / fills small holes).
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ---------------------------------------------------------------------------
# Background remover — lazy singleton
# ---------------------------------------------------------------------------

_remover_instance: 'BackgroundRemover | None' = None
_remover_lock = threading.Lock()


class BackgroundRemover:
    """
    MOG2 background subtractor.

    The model warms up over the first ``history`` frames; during warm-up the
    mask will include more noise.  Once stable it cleanly isolates the
    foreground subject.

    Parameters
    ----------
    history:
        Number of frames used to build the background model (default 500).
    var_threshold:
        Mahalanobis distance threshold — lower = more sensitive to change,
        higher = only large changes count as foreground (default 40).
    learning_rate:
        How fast the background model updates each frame.
        0.0 = frozen, 1.0 = full reset every frame, -1 = auto (default 0.005).
    bg_color:
        BGR tuple for the replacement background (default black).
    """

    def __init__(
        self,
        history:       int   = 500,
        var_threshold: float = 40,
        learning_rate: float = 0.005,
        bg_color:      tuple = (0, 0, 0),
    ):
        self._sub           = cv2.createBackgroundSubtractorMOG2(
                                  history       = history,
                                  varThreshold  = var_threshold,
                                  detectShadows = True,
                              )
        self.learning_rate  = learning_rate
        self.bg_color       = bg_color
        print('[bg] MOG2 background subtractor initialised')

    def remove(self, frame: np.ndarray, bg_color: tuple | None = None) -> np.ndarray:
        """Return a copy of *frame* with background replaced by *bg_color*."""
        color = bg_color if bg_color is not None else self.bg_color

        raw_mask = self._sub.apply(frame, learningRate=self.learning_rate)

        # MOG2 marks shadows as 127 — treat them as background.
        fg_mask = (raw_mask == 255).astype(np.uint8)

        # Clean up noise: remove small specks, close gaps inside the subject.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

        mask = fg_mask[:, :, np.newaxis]
        bg   = np.full_like(frame, color)
        return (frame * mask + bg * (1 - mask)).astype(np.uint8)


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
