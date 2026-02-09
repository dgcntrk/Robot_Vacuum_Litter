# Cat Litter Monitor

Near-realtime cat detection system for litter box monitoring with Apple Silicon optimization.

## ✨ New: Dynamic Litter Box Detection

**No more manual zone configuration!** The system now automatically detects your litter box using Apple Vision and computer vision techniques:

- **Apple Vision Rectangle Detection**: Uses the Neural Engine to find box-like containers
- **Temporal Stabilization**: Smooths detection across frames for stable tracking
- **Background Subtraction Fallback**: Alternative method for challenging scenes
- **Camera Shift Resilience**: Automatically re-detects if camera moves

## Architecture Overview

This is a **2-phase architecture** designed for reliability and performance on Apple Silicon Macs.

### Phase 1: Detection Pipeline (PRIMARY)
- Near-realtime cat detection using Apple Neural Engine (CoreML)
- **Dynamic litter box detection** - no manual zone configuration needed
- RTSP camera capture with low-latency processing
- Event-driven architecture
- <500ms latency from capture to detection event

### Phase 2: Robot Control (Interface/Stub)
- Clean interface for robot vacuum integration
- Stub implementation provided - wire your own robot controller
- Event callbacks for cat entry, exit, and dispatch-ready

## How Dynamic Detection Works

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Frame     │────▶│  Litter Box         │────▶│  Tracked    │
│   Input     │     │  Detection          │     │  Region     │
└─────────────┘     └─────────────────────┘     └─────────────┘
                            │
                            ├──▶ Apple Vision: VNDetectRectanglesRequest
                            └──▶ Fallback: Background subtraction + contours

┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Cat       │────▶│  Zone Overlap       │────▶│  State      │
│   Detection │     │  Check              │     │  Machine    │
│   (YOLO)    │     │  (cat bbox vs box)  │     │  Events     │
└─────────────┘     └─────────────────────┘     └─────────────┘
```

### Detection Methods

| Method | Speed | Accuracy | When Used |
|--------|-------|----------|-----------|
| Apple Vision Rectangles | ~10-30ms | High | Primary method on macOS |
| Background Subtraction | ~20-50ms | Medium | Fallback for challenging scenes |
| Static Zones | 0ms | Exact | Optional fallback if dynamic fails |

## Installation

```bash
# 1. Create virtual environment
cd /path/to/cat-litter-monitor
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Download CoreML model (or convert your own)
# See "Model Setup" section below

# 4. Configure
# Edit config/settings.yaml (minimal config needed now!)

# 5. Run
python -m src.main
```

## Quick Start

1. **Point your camera at the litter box area**
2. **Run the monitor** - it will automatically detect the litter box within a few seconds
3. **Watch the green box** appear around the detected litter box
4. **Press 'r'** in the video window if you need to reset detection (e.g., after moving camera)

## Model Setup

### Option 1: Use Apple's Vision Framework (Built-in)
No model download needed - uses `VNRecognizeAnimalsRequest` (cat-only, no bounding boxes).

### Option 2: YOLOv8 CoreML (Recommended)
```bash
# Download pre-converted model or convert your own:
pip install ultralytics
yolo export model=yolov8n.pt format=coreml nms=True imgsz=640
# Move output to models/yolov8n.mlpackage
```

### Option 3: Custom CoreML Model
Place your `.mlpackage` or `.mlmodel` in the `models/` directory and update config.

## Configuration

Edit `config/settings.yaml`:

```yaml
# Camera settings
camera:
  rtsp_url: "rtsp://YOUR_CAMERA_IP/live0"
  fps: 15
  resolution: [640, 480]

# Detection settings
detection:
  provider: "coreml_yolo"
  model_path: "models/yolov8n.mlpackage"
  confidence_threshold: 0.5
  inference_interval: 0.2  # 5 FPS detection
  dynamic_zones: true      # Enable auto-detection (NEW!)

# Static zones are now OPTIONAL - only needed if you disable dynamic_zones
# zones:
#   litter_box_1:
#     name: "Main Litter Box"
#     bbox: [230, 250, 352, 456]

# State machine timing
timing:
  min_occupancy_seconds: 5    # Must see cat for 5s to count as "inside"
  cooldown_seconds: 60        # Wait 60s after exit before dispatch
  max_session_minutes: 10     # Timeout if cat sits too long

# Robot integration
robot:
  enabled: false
  room_name: "Litter"
  dispatch_delay_seconds: 5

# Event logging
events:
  log_dir: "./logs"
  max_history: 1000

# Visualization
visualization:
  enabled: true
  show_zones: true          # Green box = auto-detected litter box
```

## How It Works

### Dynamic Litter Box Detection

1. **Frame Capture**: RTSP client grabs frames at target FPS
2. **Litter Box Detection**: 
   - Primary: Apple Vision `VNDetectRectanglesRequest` finds box-like containers
   - Filters by size (5-50% of frame), aspect ratio, and centrality
   - Excludes regions where cats are detected
3. **Temporal Stabilization**: 
   - Tracks detections across frames using IoU matching
   - Applies exponential moving average for smooth bbox
   - Requires minimum stability before accepting detection
4. **Cat Detection**: YOLOv8 CoreML detects cats in frame
5. **Overlap Check**: Determines if cat bounding box overlaps with litter box
6. **State Machine**: Tracks entry/exit events

### State Machine
```
IDLE → CAT_ENTERED → CAT_INSIDE → CAT_EXITED → COOLDOWN → DISPATCH_READY → IDLE
       <5s timeout>    <min_occupancy>  <cooldown_seconds>    >fire callback>
```

### Visualization Colors
- **Green box**: Auto-detected litter box (dynamic)
- **Orange box**: Manually configured zone (static, if any)
- **Red text**: Cat inside litter box
- **Yellow text**: Cat entered/exited

## Keyboard Controls

When the video window is open:
- **q**: Quit the monitor
- **r**: Reset litter box detection (use if camera moved)

## Environment Variables

```bash
# Override settings without editing YAML
export CAMERA_RTSP_URL="rtsp://YOUR_CAMERA_IP/live0"
export DETECTION_DYNAMIC_ZONES="true"   # Enable/disable dynamic detection
export DETECTION_PROVIDER="coreml_yolo"
export ROBOT_ENABLED="false"
export ROBOT_ROOM_NAME="Litter"
```

## Phase 2: Robot Integration

The robot controller is a stub interface. To integrate your robot:

1. Implement `BaseRobotController` interface:

```python
from src.robot.interface import BaseRobotController

class MyRobotController(BaseRobotController):
    async def connect(self) -> bool:
        # Connect to robot API
        return True
    
    async def dispatch(self, room: str | None = None) -> bool:
        # Send robot to clean
        print(f"Dispatching robot to {room}")
        return True
    
    async def stop(self) -> bool:
        # Emergency stop
        return True
    
    async def return_to_dock(self) -> bool:
        # Send home
        return True
```

2. Wire it in `main.py`:

```python
from src.robot.interface import RobotAdapter

# Create your controller
my_robot = MyRobotController(...)

# Wrap in adapter (handles async/sync bridge)
robot_adapter = RobotAdapter(my_robot)

# Set dispatch callback
detector.on_dispatch_ready = robot_adapter.on_dispatch_ready
```

## Performance Tuning

### For Lower Latency
- Reduce `inference_interval` (e.g., 0.1s = 10 FPS detection)
- Use lower camera resolution (480p instead of 720p)
- Use `vision_framework` provider (fastest but no bounding boxes)

### For Lower CPU Usage
- Increase `inference_interval` (e.g., 0.5s = 2 FPS detection is often enough)
- Reduce camera FPS if camera supports it

### Apple Silicon Specific
- CoreML automatically uses Neural Engine when beneficial
- Apple Vision rectangle detection runs entirely on Neural Engine (~10-30ms)
- Model compilation happens on first run (cached for subsequent runs)

## Troubleshooting

### Litter Box Not Detected
- Ensure the litter box is clearly visible (not obstructed)
- Try pressing 'r' to reset detection
- Check that the litter box is a rectangular container ( Vision looks for rectangles)
- Ensure good lighting - very dark scenes may fail
- Check logs for detection method being used

### High Latency
```bash
# Check camera stream latency
ffplay -fflags nobuffer -flags low_delay rtsp://your-camera-url

# Verify CoreML is using Neural Engine
# Look for "ANE" in Activity Monitor during inference
```

### Model Loading Errors
```bash
# Re-download or re-convert model
# For YOLO: yolo export model=yolov8n.pt format=coreml nms=True
```

### RTSP Connection Issues
- Ensure camera is on same network
- Try TCP transport: `rtsp://...?tcp` or configure in YAML
- Check firewall settings

## Advanced: Static Zones (Fallback)

If dynamic detection doesn't work well for your setup, you can still use static zones:

```yaml
detection:
  dynamic_zones: false  # Disable auto-detection

zones:
  litter_box_1:
    name: "Main Litter Box"
    bbox: [230, 250, 352, 456]  # x1, y1, x2, y2 in pixel coords
```

## Web Dashboard

The cat-litter-monitor includes a built-in web dashboard for real-time monitoring. Access it at `http://localhost:8080` when running.

### Dashboard Features

1. **Live Camera Feed** - MJPEG stream with detection overlays
   - Green boxes: Litter box zones (auto-detected)
   - Yellow boxes: Cat detections

2. **Current State** - Large color-coded state indicator
   - **IDLE** (gray): Waiting for cat
   - **CAT ENTERED** (yellow): Cat detected, confirming
   - **CAT INSIDE** (red): Confirmed occupancy
   - **CAT EXITED** (yellow): Cat left, cooling down
   - **COOLDOWN** (blue): Waiting before robot dispatch
   - **DISPATCH READY** (green): Ready to clean

3. **Today's Summary** - Daily statistics
   - Total litter box visits
   - Average visit duration
   - Last visit time

4. **System Health** - Real-time metrics
   - FPS (inference rate)
   - Inference latency (ms)
   - Uptime

5. **Recent Events** - Live event log
   - State changes
   - Cat entries/exits
   - Detection events
   - Auto-scrolls with latest events

### Dashboard Configuration

```yaml
# config/settings.yaml
dashboard:
  enabled: true       # Enable/disable dashboard
  host: "0.0.0.0"     # Bind address (0.0.0.0 for all interfaces)
  port: 8080          # Port to serve on
```

Access the dashboard from any device on your network at `http://<machine-ip>:8080`.

### API Endpoints

The dashboard also exposes JSON API endpoints:

- `GET /api/stats` - Current system stats and zone states
- `GET /api/events?limit=50` - Recent events
- `GET /api/today` - Today's summary statistics
- `GET /video_feed` - MJPEG video stream
- `WS /ws` - WebSocket for real-time updates

## Project Structure
```
cat-litter-monitor/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   ├── config.py                  # Settings management
│   ├── camera/
│   │   ├── __init__.py
│   │   └── rtsp_client.py         # Low-latency RTSP capture
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── coreml_detector.py     # CoreML YOLO inference (cats)
│   │   ├── litter_box_detector.py # Dynamic litter box detection
│   │   └── zone_manager.py        # Zone matching logic
│   ├── state/
│   │   ├── __init__.py
│   │   └── fsm.py                 # Litter box state machine
│   ├── events/
│   │   ├── __init__.py
│   │   └── logger.py              # Event logging
│   ├── robot/
│   │   ├── __init__.py
│   │   ├── interface.py           # Base robot interface
│   │   └── stub.py                # Stub implementation
│   └── dashboard/
│       ├── __init__.py
│       ├── server.py              # FastAPI web server
│       └── templates/
│           └── index.html         # Dashboard UI
├── config/
│   └── settings.yaml
├── models/                        # CoreML models
├── logs/                          # Event logs
├── tests/
└── README.md
```

## Limitations & Future Improvements

### Current Limitations
1. **Rectangle Detection**: Works best with rectangular litter boxes; unusual shapes may not be detected
2. **Single Litter Box**: Currently optimized for detecting one primary litter box
3. **macOS Required**: Apple Vision framework requires macOS (fallback contour detection works on Linux)
4. **Initial Detection**: May take 1-2 seconds to stabilize on first run

### Future Improvements
1. **Multi-Box Support**: Detect and track multiple litter boxes simultaneously
2. **Custom Training**: Fine-tuned model specifically for litter box detection
3. **Deep SORT Tracking**: More robust tracking across occlusions
4. **Auto-Calibration**: Learn litter box location over time for more robust detection

## License
MIT
