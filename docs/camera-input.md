# Camera Input

The emotion-detection app receives video via GStreamer RTP H.264 streams sent
from the host machine (Windows) over UDP. This guide covers single-camera and
dual-camera setups.

## Finding camera device names (Windows)

You need the device index or name to tell GStreamer which camera to use.

### Option 1 — GStreamer device monitor (recommended)

```powershell
gst-device-monitor-1.0 Video/Source
```

Look for lines like:

```
Device found:

    name  : HD Webcam
    class : Video/Source
    ...
    properties:
        device.index = 0
        ...

    name  : USB Camera
    class : Video/Source
    ...
    properties:
        device.index = 1
```

### Option 2 — FFmpeg

```powershell
ffmpeg -list_devices true -f dshow -i dummy
```

Lists DirectShow device names. Note: the index order may differ from
GStreamer's `ksvideosrc` device-index — always verify with option 1.

### Option 3 — PowerShell

```powershell
Get-PnpDevice -Class Camera | Format-Table -Property FriendlyName, Status
```

Shows connected camera hardware but doesn't give GStreamer indices.

---

## Single-camera setup

### 1. Get the WSL2 IP

Run inside WSL:

```bash
ip addr show eth0 | grep 'inet '
```

### 2. Start the GStreamer sender on Windows

```powershell
gst-launch-1.0 ksvideosrc device-index=0 ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! rtph264pay config-interval=-1 ! udpsink host=<WSL2_IP> port=5001
```

### 3. docker-compose.yml

```yaml
environment:
  - CAMERA_GST_PIPELINE=udpsrc port=5001 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink
```

---

## Dual-camera setup (auto-switching)

Both cameras are connected to the same Windows machine. Each camera gets its
own GStreamer sender process and its own UDP port. Both pipelines run
continuously inside the container — switching is a zero-downtime pointer swap.

### 1. Start two GStreamer senders on Windows

Open two terminals:

**Camera A (port 5001):**

```powershell
gst-launch-1.0 ksvideosrc device-index=0 ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! rtph264pay config-interval=-1 ! udpsink host=<WSL2_IP> port=5001
```

**Camera B (port 5002):**

```powershell
gst-launch-1.0 ksvideosrc device-index=1 ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! rtph264pay config-interval=-1 ! udpsink host=<WSL2_IP> port=5002
```

Replace `<WSL2_IP>` with the address from `ip addr show eth0 | grep 'inet '`.

### 2. docker-compose.yml

```yaml
environment:
  # Camera A
  - CAMERA_GST_PIPELINE=udpsrc port=5001 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink
  # Camera B
  - CAMERA_GST_PIPELINE_2=udpsrc port=5002 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink
  # Seconds between automatic camera switches (default 20, 0 = no switching)
  - CAMERA_SWITCH_INTERVAL=20
```

### How it works

- Both GStreamer decode pipelines run in parallel from startup.
- The camera reader thread pulls frames from the **active** camera only.
- Every `CAMERA_SWITCH_INTERVAL` seconds the active camera flips.
- The inactive pipeline keeps decoding with `max-buffers=1 drop=true`, so
  only the latest frame is buffered. The first frame after a switch is
  immediately available — no cold-start delay.
- Face tracking state is cleared on switch to avoid ghost detections from the
  previous camera.

### Fallback

If `CAMERA_GST_PIPELINE_2` is not set, the system runs in single-camera mode
exactly as before. `CAMERA_SWITCH_INTERVAL` is ignored.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GStreamer pipeline stalled in state ...` | No UDP data arriving | Check WSL2 IP, Windows firewall, sender running |
| Black/frozen frame after switch | Inactive pipeline lost sync | Shouldn't happen — both pipelines are always running. Check sender is still alive. |
| Wrong camera on device-index N | GStreamer index doesn't match Windows order | Run `gst-device-monitor-1.0 Video/Source` and use the correct index |
| `udpsrc: Could not open resource for reading` | Port already in use | Change the port number or kill the conflicting process |
