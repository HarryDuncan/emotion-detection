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
from frame_utils import annotate_data_layer, annotate_frame
from routes.registry import FieldSpec, define

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
    output      = {
        'stream': FieldSpec(
            'stream',
            'multipart/x-mixed-replace boundary=frame — '
            'continuous JPEG frames delivered at camera FPS (default 15 fps). '
            'Displays a grey placeholder frame when no camera is available.',
        ),
    },
)

define(
    name        = 'video_dominant_emotion',
    path        = '/video_dominant_emotion',
    methods     = ['GET'],
    description = (
        'MJPEG video stream with per-face bounding boxes and dominant emotion '
        'labels overlaid. Connecting to this endpoint automatically activates '
        'the background inference loop; disconnecting idles it when no other '
        'clients remain and /start_detection has not been called.'
    ),
    output      = {
        'stream': FieldSpec(
            'stream',
            'multipart/x-mixed-replace boundary=frame — '
            'annotated JPEG frames at camera FPS. '
            'Each face is outlined in a color matching its dominant emotion, '
            'with the emotion label rendered inside the bounding box.',
        ),
    },
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
        # Tell the inference loop a client is watching — it will start running.
        with _state.emotion_client_lock:
            _state.emotion_active_clients += 1
        print(f"[emotion] Client connected   active_clients={_state.emotion_active_clients}")

        last_seq = -1
        try:
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

                annotate_frame(frame, result)

                ret, buffer = cv2.imencode('.jpg', frame, _jpeg_params)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        finally:
            # Client disconnected — decrement and idle the inference loop if nobody remains.
            with _state.emotion_client_lock:
                _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
            print(f"[emotion] Client disconnected active_clients={_state.emotion_active_clients}")
            if _state.emotion_active_clients == 0 and not _state.emotion_explicitly_enabled:
                with _state.emotion_result_lock:
                    _state.latest_emotion_result.clear()
                    _state.latest_emotion_result.update({'face_detected': False, 'faces': []})

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


define(
    name        = 'video_data_layer',
    path        = '/video_data_layer',
    methods     = ['GET'],
    description = (
        'MJPEG-style stream of transparent PNG frames containing only the '
        'annotation layer (bounding boxes and emotion labels). No camera image '
        'is included — composite this over /video_feed in the browser to overlay '
        'annotations without baking them into the video. '
        'Activates the inference loop while a client is connected.'
    ),
    output      = {
        'stream': FieldSpec(
            'stream',
            'multipart/x-mixed-replace boundary=frame — '
            'RGBA PNG frames at camera FPS. Each frame is fully transparent '
            'except for the drawn annotation shapes.',
        ),
    },
)

@bp.route('/video_data_layer')
def video_data_layer():
    def generate():
        with _state.emotion_client_lock:
            _state.emotion_active_clients += 1
        print(f"[data_layer] Client connected   active_clients={_state.emotion_active_clients}")

        last_seq = -1
        try:
            while True:
                with _state.frame_condition:
                    _state.frame_condition.wait_for(
                        lambda: _state.frame_seq != last_seq, timeout=1.0
                    )
                    frame    = _state.latest_frame_flipped
                    last_seq = _state.frame_seq

                h, w = frame.shape[:2] if frame is not None else (480, 640)

                with _state.emotion_result_lock:
                    result = dict(_state.latest_emotion_result)

                canvas = annotate_data_layer(result, h, w)

                ret, buffer = cv2.imencode('.png', canvas)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/png\r\n\r\n' + buffer.tobytes() + b'\r\n')

        finally:
            with _state.emotion_client_lock:
                _state.emotion_active_clients = max(0, _state.emotion_active_clients - 1)
            print(f"[data_layer] Client disconnected active_clients={_state.emotion_active_clients}")
            if _state.emotion_active_clients == 0 and not _state.emotion_explicitly_enabled:
                with _state.emotion_result_lock:
                    _state.latest_emotion_result.clear()
                    _state.latest_emotion_result.update({'face_detected': False, 'faces': []})

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
