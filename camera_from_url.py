#!/usr/bin/env python3
"""
Read from a remote video feed URL (e.g. Windows machine serving /video_feed) and display.
Usage:
  CAMERA_URL=http://192.168.1.10:5000/video_feed python camera_from_url.py
  # or set win_ip below and run
"""
import os
import cv2
from camera_input import CameraInput, DEFAULT_CAMERA_CONFIG

# Set your Windows IP, or use env CAMERA_URL
win_ip = os.environ.get("CAMERA_WIN_IP", "172.29.224.1")
port = os.environ.get("CAMERA_PORT", "5000")
default_url = f"http://{win_ip}:{port}/interaction_node/video_feed"

config = {**DEFAULT_CAMERA_CONFIG, "camera_url": os.environ.get("CAMERA_URL", default_url)}
cap_source = CameraInput(config)
cap_source._initialize_camera()

if not cap_source.isOpened():
    print("Could not open video feed. Check CAMERA_URL or CAMERA_WIN_IP/CAMERA_PORT.")
    exit(1)

print("Reading from", config["camera_url"], "- press 'q' to quit.")

while True:
    ret, frame = cap_source.read()
    if not ret or frame is None:
        break
    cv2.imshow("WSL Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap_source.release()
cv2.destroyAllWindows()
