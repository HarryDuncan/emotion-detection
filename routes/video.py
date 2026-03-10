"""
Video streaming routes.

Endpoints
---------
GET /video_feed              — raw MJPEG stream, no annotation
GET /video_dominant_emotion  — MJPEG stream with per-face emotion overlay
"""
import os
import time

import cv2
import numpy as np
from flask import Blueprint, Response

import state as _state
from routes.registry import define

bp = Blueprint('video', __name__)

STREAM_JPEG_QUALITY   = int(os.environ.get('STREAM_JPEG_QUALITY', 70))
_jpeg_params          = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
_VIDEO_LOG_INTERVAL   = 30   # log every N frames (~1 s at 30 fps)

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

define(
    name        = 'video_feed',
    path        = '/video_feed',
    methods     = ['GET'],
    description = 'Raw MJPEG video stream from the camera with no annotation.',
    output      = {'stream': 'multipart/x-mixed-replace — JPEG frames at camera FPS'},
)

define(
    name        = 'video_dominant_emotion',
    path        = '/video_dominant_emotion',
    methods     = ['GET'],
    description = (
        'MJPEG video stream with per-face bounding boxes and dominant emotion '
        'labels overlaid, sourced from the background inference thread.'
    ),
    output      = {'stream': 'multipart/x-mixed-replace — annotated JPEG frames at camera FPS'},
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_camera_frame_bytes() -> bytes:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, 'No camera', (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@bp.route('/video_feed')
def video_feed():
    no_camera_frame = _no_camera_frame_bytes()

    def generate():
        frame_count = 0
        last_seq    = -1
        while True:
            with _state.frame_condition:
                _state.frame_condition.wait_for(
                    lambda: _state.frame_seq != last_seq, timeout=1.0
                )
                frame      = _state.latest_frame_flipped
                last_seq   = _state.frame_seq
                frame_age  = time.time() - _state.frame_timestamp if _state.frame_timestamp else 0

            if frame is None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                continue

            t_enc = time.time()
            ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_count += 1
            if frame_count % _VIDEO_LOG_INTERVAL == 0:
                encode_ms = (time.time() - t_enc) * 1000
                print(f'[video_feed] #{frame_count}  frame_age={frame_age*1000:.0f}ms  encode={encode_ms:.1f}ms')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/video_dominant_emotion')
def video_dominant_emotion():
    no_camera_frame = _no_camera_frame_bytes()

    def generate():
        last_seq = -1
        while True:
            with _state.frame_condition:
                _state.frame_condition.wait_for(
                    lambda: _state.frame_seq != last_seq, timeout=1.0
                )
                frame    = (_state.latest_frame_flipped.copy()
                            if _state.latest_frame_flipped is not None else None)
                last_seq = _state.frame_seq

            if frame is None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + no_camera_frame + b'\r\n')
                continue

            with _state.emotion_result_lock:
                result = dict(_state.latest_emotion_result)

            for face in result.get('faces', []):
                x, y, w, h = face['face_bbox']
                color       = face['emotion_color_bgr']
                label       = face.get('dominant_emotion', '')
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                ty          = max(y - 6, th + 4)
                cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 2), color, cv2.FILLED)
                luma        = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
                text_color  = (0, 0, 0) if luma > 140 else (255, 255, 255)
                cv2.putText(frame, label, (x + 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
