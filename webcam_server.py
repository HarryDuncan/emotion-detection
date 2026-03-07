from flask import Flask, Response
import cv2
import atexit
import sys

app = Flask(__name__)

# Try to open camera
print("Attempting to open webcam...")
camera = None

# Try multiple camera indices
for camera_index in [0, 1, 2]:
    print(f"Trying camera index {camera_index}...")
    test_camera = cv2.VideoCapture(camera_index)
    
    if test_camera.isOpened():
        # Try to read a frame to verify it's working
        ret, frame = test_camera.read()
        if ret and frame is not None:
            camera = test_camera
            print(f"✓ Successfully opened camera at index {camera_index}")
            print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
            break
        else:
            print(f"  Camera {camera_index} opened but cannot read frames")
            test_camera.release()
    else:
        print(f"  Camera index {camera_index} not available")
        if test_camera:
            test_camera.release()

if camera is None or not camera.isOpened():
    print("\n" + "="*60)
    print("ERROR: Could not open webcam!")
    print("="*60)
    print("Possible issues:")
    print("  1. Webcam is being used by another application")
    print("  2. Webcam drivers are not installed")
    print("  3. Webcam permissions are not granted")
    print("  4. Webcam is not connected")
    print("\nTroubleshooting:")
    print("  - Close other applications using the webcam (Zoom, Teams, etc.)")
    print("  - Check Windows Camera app to verify webcam works")
    print("  - Try running as administrator")
    print("="*60)
    sys.exit(1)

# Set camera properties for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

print(f"Webcam opened successfully!")
print(f"Camera resolution: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Camera FPS: {camera.get(cv2.CAP_PROP_FPS)}")

@atexit.register
def cleanup():
    print("\nReleasing camera...")
    if camera is not None:
        camera.release()
    print("Camera released.")

def generate_frames():
    frame_count = 0
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print(f"Warning: Failed to read frame from camera")
                break
            
            if frame is None:
                print(f"Warning: Frame is None")
                continue
                
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Streaming frame {frame_count}...")
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("Warning: Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

@app.route('/')
def index():
    return """
    <html>
    <head><title>Webcam Server</title></head>
    <body>
        <h1>Webcam Server is Running!</h1>
        <p>Video stream is available at: <a href="/video">/video</a></p>
        <p>To test, open this URL in your browser or use it with OpenCV:</p>
        <code>http://localhost:1995/video</code>
        <hr>
        <h2>Video Stream:</h2>
        <img src="/video" style="max-width: 100%; height: auto;">
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {
        'status': 'running',
        'camera_opened': camera.isOpened() if camera else False,
        'port': 1995
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Webcam Server")
    print("="*60)
    print(f"Server will be available at: http://0.0.0.0:1995")
    print(f"Video stream URL: http://localhost:1995/video")
    print(f"Status endpoint: http://localhost:1995/status")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        app.run(host='0.0.0.0', port=1995, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        cleanup()