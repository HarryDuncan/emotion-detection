import numpy as np

# Canonical per-emotion RGB colors. Neutral is white so it sits at the
# bright end of the gamut rather than collapsing the blend toward black.
EMOTION_COLORS_RGB: dict[str, tuple] = {
    'neutral':  (255, 255, 255),  # White
    'happy':    (255, 255,   0),  # Yellow
    'sad':      (  0,   0, 255),  # Blue
    'angry':    (255,   0,   0),  # Red
    'surprise': (255, 165,   0),  # Orange
    'fear':     (128,   0, 128),  # Purple
    'disgust':  (  0, 128,   0),  # Green
}


def emotions_to_color(emotions_dict):
    """
    Convert emotion confidence scores to a weighted RGB color.

    Args:
        emotions_dict: Dictionary with emotion names as keys and confidence
                       scores (0-100) as values,
                       e.g. {'happy': 45.2, 'sad': 30.1, 'neutral': 15.0, ...}

    Returns:
        tuple: (R, G, B) color values in range 0-255
    """
    weighted_rgb = np.array([0.0, 0.0, 0.0])

    for emotion, confidence in emotions_dict.items():
        weight      = confidence / 100.0
        emotion_rgb = np.array(EMOTION_COLORS_RGB.get(emotion.lower(), (255, 255, 255)))
        weighted_rgb += emotion_rgb * weight

    final_rgb = np.clip(weighted_rgb, 0, 255).astype(int)
    return (int(final_rgb[0]), int(final_rgb[1]), int(final_rgb[2]))


def emotions_to_color_normalized(emotions_dict):
    """
    Convert emotions to color with normalized weighting.
    Useful if confidences don't sum to exactly 100.
    """
    weighted_rgb     = np.array([0.0, 0.0, 0.0])
    total_confidence = sum(emotions_dict.values())

    if total_confidence == 0:
        return (255, 255, 255)

    for emotion, confidence in emotions_dict.items():
        weight      = confidence / total_confidence
        emotion_rgb = np.array(EMOTION_COLORS_RGB.get(emotion.lower(), (255, 255, 255)))
        weighted_rgb += emotion_rgb * weight

    final_rgb = np.clip(weighted_rgb, 0, 255).astype(int)
    return (int(final_rgb[0]), int(final_rgb[1]), int(final_rgb[2]))

