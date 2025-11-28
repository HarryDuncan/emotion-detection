def format_dominant_emotion(emotions_dict):
    """
    Format the dominant emotion as "I am feeling [emotion]"
    
    Args:
        emotions_dict: Dictionary with emotion names as keys and scores (0-100) as values
    
    Returns:
        str: "I am feeling [dominant_emotion]" or empty string if no emotions
    """
    if not emotions_dict:
        return ""
    
    # Find the emotion with the highest score
    dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])[0]
    
    # Capitalize the emotion name
    emotion_name = dominant_emotion.capitalize()
    
    return f"I am feeling {emotion_name}"