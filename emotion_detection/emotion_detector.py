import cv2
from deepface import DeepFace
from emotion_transforms import emotions_to_color
import numpy as np
from collections import deque

# Number of consecutive frames to average per tracked face.
# Higher = smoother labels, slightly more lag on real expression changes.
_SMOOTH_WINDOW = 4

# Minimum IoU to consider a new detection as the same face as an existing track.
_IOU_MATCH_THRESHOLD = 0.25

# Frames a track can go unmatched before it is discarded.
_MAX_TRACK_AGE = 2


def _iou(a, b):
    """Intersection-over-Union for two bboxes given as (x, y, w, h)."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class _FaceTrack:
    """Tracks a single face and smooths its emotion scores over a rolling window."""

    def __init__(self, bbox, emotions):
        self.bbox = bbox
        self.history: deque = deque(maxlen=_SMOOTH_WINDOW)
        self.history.append(emotions)
        self.age = 0

    def update(self, bbox, emotions):
        self.bbox = bbox
        self.history.append(emotions)
        self.age = 0

    @property
    def smoothed_emotions(self):
        keys = self.history[0].keys()
        return {k: float(np.mean([h[k] for h in self.history])) for k in keys}


class EmotionDetector:
    """
    Multi-face emotion detector with temporal smoothing.

    Uses RetinaFace (via DeepFace) for face detection — significantly more
    accurate than Haar cascades, handles angles and partial occlusion well.
    Emotions are averaged over a rolling window per tracked face to reduce
    frame-to-frame jitter.
    """

    def __init__(self, load_models_on_init=False):
        self.models_loaded = False
        self._tracks: list = []
        self._logged_keys = False

        if load_models_on_init:
            self.load_models()

    def load_models(self):
        """Warm up DeepFace / TensorFlow so the first real frame is fast."""
        print("Loading emotion detection models...")
        try:
            test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            try:
                DeepFace.analyze(
                    test_frame,
                    actions=['emotion'],
                    detector_backend='retinaface',
                    model_name='Facenet512',
                    enforce_detection=False,
                    silent=True,
                )
            except Exception as e:
                print(f"Model warmup triggered (expected: {type(e).__name__})")
            self.models_loaded = True
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            if "face" in str(e).lower():
                self.models_loaded = True
                return True
            return False

    def detect_emotions_from_frame(self, frame, silent=True):
        """
        Detect emotions for every face visible in the frame.

        Args:
            frame: BGR frame from OpenCV (numpy array)
            silent: Suppress DeepFace log output

        Returns:
            dict: {
                'face_detected': bool,
                'faces': [
                    {
                        'emotions':          dict  — smoothed scores (0-100),
                        'dominant_emotion':  str,
                        'emotion_color_rgb': tuple (R, G, B),
                        'emotion_color_bgr': tuple (B, G, R),
                        'face_bbox':         tuple (x, y, w, h),
                    },
                    ...   one entry per detected face
                ]
            }
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            raw = DeepFace.analyze(
                rgb_frame,
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=False,
                align=True,
                silent=silent,
            )

            # DeepFace always returns a list; guard against unexpected formats.
            if not raw:
                self._age_tracks(matched=set())
                return {'face_detected': False, 'faces': []}
            if isinstance(raw, dict):
                raw = [raw]

            # Debug: log keys and facial_area on first call to identify key names.
            if not getattr(self, '_logged_keys', False):
                self._logged_keys = True
                sample = raw[0] if raw else {}
                fa_sample = sample.get('facial_area') or sample.get('region') or {}
                print(f"[emotion_detector] DeepFace result keys: {list(sample.keys())}")
                print(f"[emotion_detector] facial_area/region: {fa_sample}")
                print(f"[emotion_detector] face_confidence: {sample.get('face_confidence', 'N/A')}")

            # --- build (bbox, emotions) pairs from DeepFace output ---
            detections = []
            for r in raw:
                # Support both 'facial_area' (newer DeepFace) and 'region' (older)
                fa = r.get('facial_area') or r.get('region') or {}
                bbox = (
                    int(fa.get('x', 0)),
                    int(fa.get('y', 0)),
                    int(fa.get('w', 0)),
                    int(fa.get('h', 0)),
                )
                # Filter out zero-confidence "no face found" placeholders that
                # DeepFace inserts when enforce_detection=False finds nothing.
                confidence = float(r.get('face_confidence', 1.0))
                if confidence < 0.5:
                    continue
                # Also skip degenerate bboxes
                if bbox[2] < 20 or bbox[3] < 20:
                    continue
                detections.append((bbox, r['emotion']))

            if not detections:
                self._age_tracks(matched=set())
                return {'face_detected': False, 'faces': []}

            # --- match each detection to an existing track via IoU ---
            matched_indices: set = set()
            new_tracks = []

            for bbox, emotions in detections:
                best_idx, best_iou = -1, _IOU_MATCH_THRESHOLD
                for i, track in enumerate(self._tracks):
                    if i in matched_indices:
                        continue
                    score = _iou(track.bbox, bbox)
                    if score > best_iou:
                        best_iou = score
                        best_idx = i

                if best_idx >= 0:
                    self._tracks[best_idx].update(bbox, emotions)
                    matched_indices.add(best_idx)
                else:
                    new_track = _FaceTrack(bbox, emotions)
                    new_tracks.append(new_track)

            # Age out unmatched tracks; append new ones.
            self._age_tracks(matched=matched_indices)
            self._tracks.extend(new_tracks)
            # Include new tracks in the output set
            new_track_indices = set(range(len(self._tracks) - len(new_tracks), len(self._tracks)))
            all_active = matched_indices | new_track_indices

            # --- build output ---
            faces_out = []
            for idx in sorted(all_active):
                if idx >= len(self._tracks):
                    continue
                track = self._tracks[idx]
                smoothed = track.smoothed_emotions
                dominant = max(smoothed, key=smoothed.get)
                color_rgb = emotions_to_color(smoothed)
                color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                faces_out.append({
                    'emotions': smoothed,
                    'dominant_emotion': dominant,
                    'emotion_color_rgb': color_rgb,
                    'emotion_color_bgr': color_bgr,
                    'face_bbox': track.bbox,
                })

            return {
                'face_detected': len(faces_out) > 0,
                'faces': faces_out,
            }

        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return {'face_detected': False, 'faces': [], 'error': str(e)}

    def _age_tracks(self, matched: set):
        """Increment age of unmatched tracks and prune stale ones."""
        for i, track in enumerate(self._tracks):
            if i not in matched:
                track.age += 1
        self._tracks = [t for t in self._tracks if t.age <= _MAX_TRACK_AGE]

    def cleanup(self):
        self._tracks.clear()
        self.models_loaded = False
        print("EmotionDetector cleaned up")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# Note: This module only works with frames, not cameras.
# Camera management is handled by the calling code (appv2.py).

if __name__ == "__main__":
    print("Example usage:")
    print("  from emotion_detection.emotion_detector import EmotionDetector")
    print("  import cv2")
    print("  detector = EmotionDetector(load_models_on_init=True)")
    print("  cap = cv2.VideoCapture(0)")
    print("  ret, frame = cap.read()")
    print("  result = detector.detect_emotions_from_frame(frame)")
    print("  for face in result['faces']:")
    print("      print(face['dominant_emotion'], face['face_bbox'])")
    print("  detector.cleanup()")
