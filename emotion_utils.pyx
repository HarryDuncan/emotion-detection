# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Cython-optimized emotion calculation utilities.
Compile with: python setup.py build_ext --inplace
"""

import cython
cimport cython
from libc.math cimport fabs

cdef class EmotionAverager:
    """
    Cython-optimized class for calculating emotion averages.
    """
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef dict calculate_averages(self, list window_entries, double cutoff_time):
        """
        Calculate average emotion scores over a time window.
        
        Args:
            window_entries: List of emotion history entries (dicts with 'timestamp' and 'emotions')
            cutoff_time: Timestamp cutoff (entries before this are filtered out)
            
        Returns:
            dict: Averaged emotion scores
        """
        cdef:
            dict averaged_emotions = {}
            dict entry
            dict entry_emotions
            str emotion_key
            double score
            int count
            double total
            double entry_time
        
        # Filter entries within time window and collect all emotion keys
        cdef set all_emotion_keys = set()
        cdef list valid_entries = []
        
        for entry in window_entries:
            entry_time = entry.get('timestamp', 0.0)
            if entry_time >= cutoff_time:
                entry_emotions = entry.get('emotions', {})
                if entry_emotions:
                    valid_entries.append(entry_emotions)
                    all_emotion_keys.update(entry_emotions.keys())
        
        if not valid_entries:
            return {}
        
        # Calculate averages for each emotion
        for emotion_key in all_emotion_keys:
            total = 0.0
            count = 0
            
            for entry_emotions in valid_entries:
                score = entry_emotions.get(emotion_key, 0.0)
                if score > 0.0:  # Only count non-zero scores
                    total += score
                    count += 1
            
            if count > 0:
                averaged_emotions[emotion_key] = total / count
        
        return averaged_emotions


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dict calculate_emotion_averages_cython(list emotion_history, double window_seconds):
    """
    Cython-optimized version of calculate_emotion_averages.
    
    Args:
        emotion_history: List of emotion history entries (dicts with 'timestamp' and 'emotions')
        window_seconds: Time window in seconds
        
    Returns:
        dict: Averaged emotion scores
    """
    cdef:
        double current_time
        double cutoff_time
        EmotionAverager averager
    
    if not emotion_history:
        return {}
    
    # Import time module
    import time
    current_time = time.time()
    cutoff_time = current_time - window_seconds
    
    # Check if we have any recent entries
    if emotion_history:
        latest = emotion_history[-1]
        if latest.get('timestamp', 0.0) < cutoff_time:
            # No data in window, return latest available
            latest_emotions = latest.get('emotions', {})
            if latest_emotions:
                return latest_emotions.copy()
            return {}
    
    averager = EmotionAverager()
    return averager.calculate_averages(emotion_history, cutoff_time)

