import numpy as np

def emotions_to_color(emotions_dict):
    """
    Convert emotion confidence scores to a weighted RGB color.
    
    Args:
        emotions_dict: Dictionary with emotion names as keys and confidence scores (0-100) as values
                     e.g., {'happy': 45.2, 'sad': 30.1, 'neutral': 15.0, ...}
    
    Returns:
        tuple: (R, G, B) color values in range 0-255
    """
    # Define color mapping for each emotion (RGB values 0-255)
    # Neutral is at origin (0, 0, 0) - black
    emotion_colors = {
        'neutral': (0, 0, 0),        # Black (origin)
        'happy': (255, 255, 0),      # Yellow
        'sad': (0, 0, 255),          # Blue
        'angry': (255, 0, 0),        # Red
        'surprise': (255, 165, 0),   # Orange
        'fear': (128, 0, 128),       # Purple
        'disgust': (0, 128, 0),      # Green
    }
    
    # Initialize weighted color vector
    weighted_rgb = np.array([0.0, 0.0, 0.0])
    
    # Normalize confidences and compute weighted sum
    for emotion, confidence in emotions_dict.items():
        # Normalize confidence from 0-100 to 0-1
        weight = confidence / 100.0
        
        # Get emotion color (default to neutral if emotion not in mapping)
        emotion_rgb = np.array(emotion_colors.get(emotion.lower(), (0, 0, 0)))
        
        # Add weighted color vector
        weighted_rgb += emotion_rgb * weight
    
    # Clamp values to valid RGB range (0-255) and convert to integers
    final_rgb = np.clip(weighted_rgb, 0, 255).astype(int)
    
    # Convert numpy integers to native Python integers for OpenCV compatibility
    return (int(final_rgb[0]), int(final_rgb[1]), int(final_rgb[2]))


def emotions_to_color_normalized(emotions_dict):
    """
    Convert emotions to color with normalized weighting.
    Useful if confidences don't sum to exactly 100.
    """
    emotion_colors = {
        'neutral': (0, 0, 0),
        'happy': (255, 255, 0),
        'sad': (0, 0, 255),
        'angry': (255, 0, 0),
        'surprise': (255, 165, 0),
        'fear': (128, 0, 128),
        'disgust': (0, 128, 0),
    }
    
    weighted_rgb = np.array([0.0, 0.0, 0.0])
    total_confidence = sum(emotions_dict.values())
    
    # Avoid division by zero
    if total_confidence == 0:
        return (0, 0, 0)
    
    for emotion, confidence in emotions_dict.items():
        # Normalize by total confidence so weights sum to 1
        weight = confidence / total_confidence
        
        emotion_rgb = np.array(emotion_colors.get(emotion.lower(), (0, 0, 0)))
        weighted_rgb += emotion_rgb * weight
    
    final_rgb = np.clip(weighted_rgb, 0, 255).astype(int)
    # Convert numpy integers to native Python integers for OpenCV compatibility
    return (int(final_rgb[0]), int(final_rgb[1]), int(final_rgb[2]))

