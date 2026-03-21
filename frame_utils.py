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
from PIL import Image, ImageDraw, ImageFont
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

DATA_LAYER_PAD = 200

# HUD fonts (PIL TrueType) — loaded lazily on first draw
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
_DATATYPE = os.path.join(_FONT_DIR, 'Datatype_Expanded-Regular.ttf')
_hud_fonts: dict | None = None


def _get_hud_fonts() -> dict:
    global _hud_fonts
    if _hud_fonts is None:
        try:
            _hud_fonts = {
                'label': ImageFont.truetype(_DATATYPE, 16),
                'value': ImageFont.truetype(_DATATYPE, 16),
                'mono':  ImageFont.truetype(_DATATYPE, 16),
                'text':  ImageFont.truetype(_DATATYPE, 15),
            }
        except OSError:
            fallback = ImageFont.load_default()
            _hud_fonts = {k: fallback for k in ('label', 'value', 'mono', 'text')}
            print('[hud] Datatype font not found — using PIL default')
    return _hud_fonts


def _wrap_text_pil(draw: ImageDraw.ImageDraw, text: str,
                   font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Word-wrap *text* so each line fits within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ''
    for word in words:
        test = f'{current} {word}'.strip()
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return '\n'.join(lines)

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


def colorize_frame(frame: np.ndarray, bgr: tuple,
                   strength: float = 0.25) -> np.ndarray:
    """Apply a colour wash over *frame* using an alpha blend.

    *bgr* is the tint colour in BGR order.  *strength* (0.0–1.0) controls
    how dominant the tint is — 0.0 returns the original frame, 1.0 replaces
    it entirely with the solid colour.  0.25 gives a noticeable but
    non-destructive tint.
    """
    overlay = np.full_like(frame, bgr, dtype=np.uint8)
    return cv2.addWeighted(frame, 1.0 - strength, overlay, strength, 0)


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


def _draw_top3(canvas: np.ndarray, face: dict, *, y_offset: int = 0) -> None:
    """Draw bounding box and top-3 emotion labels onto *canvas* in-place.

    Handles both 3-channel BGR frames and 4-channel BGRA transparent canvases.
    Each of the three label rows uses the individual emotion's palette color so
    the blend is visible at a glance. Rows are stacked immediately above the
    bounding box, ordered 1st (top) → 3rd (bottom).

    *y_offset* shifts all drawing downward (used by the expanded data-layer
    canvas to centre the tracking area below the HUD strip).
    """
    n_ch       = canvas.shape[2]
    alpha      = 255 if n_ch == 4 else None   # None → omit alpha component

    def _c(bgr):
        return (*bgr, alpha) if alpha is not None else bgr

    x, y, w, h  = face['face_bbox']
    y += y_offset
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


def _draw_hud(canvas: np.ndarray, pad: int, width: int,
              elapsed_s: float, status: str, subjects: int,
              llm_text: str = '') -> None:
    """Draw compact status bar + LLM text area into the top *pad* pixels."""
    fonts   = _get_hud_fonts()
    margin  = 20

    hud  = Image.new('RGBA', (width, pad), (15, 15, 15, 180))
    draw = ImageDraw.Draw(hud)

    # ── compact status bar (top ~30 px) ──────────────────────────────────
    total_s    = int(elapsed_s)
    ms         = int((elapsed_s - total_s) * 1000)
    hrs, rem   = divmod(total_s, 3600)
    mins, secs = divmod(rem, 60)
    timer_val  = (f'{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}'
                  if hrs else f'{mins:02d}:{secs:02d}.{ms:03d}')

    dim   = (100, 100, 100, 255)
    bright = (220, 220, 220, 255)
    bar_y = 8
    font  = fonts['label']

    sections = [
        (f'TIMER  {timer_val}',            0.0),
        (f'STATUS  {status}',              0.33),
        (f'SUBJECTS DETECTED  {subjects}', 0.66),
    ]
    for text, frac in sections:
        x = int(width * frac) + margin
        draw.text((x, bar_y), text, fill=bright, font=font)

    # separator
    sep_y = 34
    draw.line([(margin, sep_y), (width - margin, sep_y)], fill=dim, width=1)

    # ── LLM text area ───────────────────────────────────────────────────
    if llm_text:
        text_font  = fonts['text']
        max_w      = width - 2 * margin
        wrapped    = _wrap_text_pil(draw, llm_text, text_font, max_w)
        draw.multiline_text((margin, sep_y + 8), wrapped,
                            fill=(200, 200, 200, 255), font=text_font,
                            spacing=5)

    # bottom accent
    draw.line([(0, pad - 1), (width - 1, pad - 1)],
              fill=(80, 80, 80, 255), width=1)

    # RGBA → BGRA into the canvas slice
    rgba = np.array(hud)
    canvas[:pad, :, 0] = rgba[:, :, 2]
    canvas[:pad, :, 1] = rgba[:, :, 1]
    canvas[:pad, :, 2] = rgba[:, :, 0]
    canvas[:pad, :, 3] = rgba[:, :, 3]


def _draw_stats_panel(canvas: np.ndarray, pad: int, width: int,
                      total_h: int, stats: dict) -> None:
    """Draw long-running session statistics into the bottom *pad* pixels.

    Stats dict keys:
        top3            — list of (emotion_name, count) for the top 3 emotions
        change_count    — total number of dominant-emotion transitions
        emotion_counts  — {emotion_name: times_experienced}
    """
    fonts   = _get_hud_fonts()
    margin  = 20

    panel = Image.new('RGBA', (width, pad), (15, 15, 15, 180))
    draw  = ImageDraw.Draw(panel)

    dim    = (100, 100, 100, 255)
    bright = (220, 220, 220, 255)
    accent = (180, 180, 180, 255)
    font   = fonts['label']
    val_font = fonts['value']

    # top accent line
    draw.line([(0, 0), (width - 1, 0)], fill=(80, 80, 80, 255), width=1)

    # ── header bar ────────────────────────────────────────────────────────
    bar_y = 10
    draw.text((margin, bar_y), 'SESSION STATISTICS', fill=bright, font=font)

    change_count = stats.get('change_count', 0)
    changes_txt = f'EMOTION CHANGES  {change_count}'
    cw = draw.textlength(changes_txt, font=font)
    draw.text((width - margin - cw, bar_y), changes_txt,
              fill=bright, font=font)

    sep_y = 32
    draw.line([(margin, sep_y), (width - margin, sep_y)], fill=dim, width=1)

    # ── top 3 emotions (left column) ──────────────────────────────────────
    col1_x = margin
    row_y  = sep_y + 10
    draw.text((col1_x, row_y), 'TOP EMOTIONS', fill=dim, font=font)
    row_y += 22

    top3 = stats.get('top3', [])
    for rank, (emo, count) in enumerate(top3, 1):
        rgb = _EMOTION_COLORS_RGB.get(emo.lower(), (180, 180, 180))
        color_rgba = (*rgb, 255)

        pill_w = 8
        pill_h = 8
        pill_y = row_y + 4
        draw.rectangle([(col1_x, pill_y), (col1_x + pill_w, pill_y + pill_h)],
                       fill=color_rgba)

        label = f'{rank}.  {emo.upper()}  ×{count}'
        draw.text((col1_x + pill_w + 8, row_y), label,
                  fill=bright, font=val_font)
        row_y += 22

    # ── per-emotion experience counts (right column) ──────────────────────
    emotion_counts = stats.get('emotion_counts', {})
    if emotion_counts:
        col2_x = width // 2 + margin
        row_y2 = sep_y + 10
        draw.text((col2_x, row_y2), 'TIMES EXPERIENCED', fill=dim, font=font)
        row_y2 += 22

        sorted_emos = sorted(emotion_counts.items(),
                             key=lambda kv: kv[1], reverse=True)
        for emo, count in sorted_emos:
            if count == 0:
                continue
            rgb = _EMOTION_COLORS_RGB.get(emo.lower(), (180, 180, 180))
            color_rgba = (*rgb, 255)

            pill_w = 8
            pill_h = 8
            pill_y = row_y2 + 4
            draw.rectangle(
                [(col2_x, pill_y), (col2_x + pill_w, pill_y + pill_h)],
                fill=color_rgba)

            label = f'{emo.upper()}  ×{count}'
            draw.text((col2_x + pill_w + 8, row_y2), label,
                      fill=accent, font=val_font)
            row_y2 += 20

    # RGBA → BGRA into the canvas bottom slice
    rgba = np.array(panel)
    y_start = total_h - pad
    canvas[y_start:, :, 0] = rgba[:, :, 2]
    canvas[y_start:, :, 1] = rgba[:, :, 1]
    canvas[y_start:, :, 2] = rgba[:, :, 0]
    canvas[y_start:, :, 3] = rgba[:, :, 3]


def annotate_data_layer(result: dict, height: int, width: int, *,
                        elapsed_s: float = 0.0,
                        llm_text: str = '',
                        stats: dict | None = None) -> np.ndarray:
    """Return a transparent BGRA canvas with HUD strip and annotation layer.

    The canvas is *height + 2 × DATA_LAYER_PAD* tall: a 200 px HUD strip on
    top, the original-frame-sized tracking area in the centre, and 200 px of
    transparent space at the bottom.  Bounding boxes and emotion labels are
    drawn in the centre region; the HUD strip shows a compact status bar and
    the current LLM narrative (typewriter-animated by the caller).

    Parameters
    ----------
    result:
        The latest emotion result dict from ``_state.latest_emotion_result``.
    height, width:
        *Original* video frame dimensions (before padding).
    elapsed_s:
        Seconds since the data-layer stream started — drives the timer display.
    llm_text:
        Visible portion of the LLM narrative (caller truncates for typewriter).
    stats:
        Session statistics dict with keys ``top3``, ``change_count``,
        ``emotion_counts``.  When provided the bottom panel renders them.
    """
    pad     = DATA_LAYER_PAD
    total_h = height + 2 * pad
    canvas  = np.zeros((total_h, width, 4), dtype=np.uint8)

    faces        = result.get('faces', [])
    face_detected = result.get('face_detected', False)
    status       = 'ACTIVE' if face_detected else 'SCANNING'

    _draw_hud(canvas, pad, width, elapsed_s, status, len(faces),
              llm_text=llm_text)

    for face in faces:
        _draw_top3(canvas, face, y_offset=pad)

    if stats is not None:
        _draw_stats_panel(canvas, pad, width, total_h, stats)

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
