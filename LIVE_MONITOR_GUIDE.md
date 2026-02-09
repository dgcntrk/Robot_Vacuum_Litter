# Live Monitor Quick Start Guide

## ✅ System Status: RUNNING

The cat litter monitor with full state machine is now operational!

## What's Running

**Process**: `live_monitor.py` (PID 17796, session: good-ridge)  
**Location**: `/path/to/cat-litter-monitor/`  
**Status**: ✅ **OPERATIONAL** (started at 00:54:43)

## What You Should See

A live video window titled **"Cat Litter Monitor - Live"** showing:

### 🟢 Litter Box Zone (Green Rectangle)
- **Location**: (1112, 558, 1379, 799)
- **Confidence**: 0.64
- **Label**: "Auto-Detected Litter Box (conf: 0.64)"

### 🟡 Cat Detections (Yellow Boxes)
- Appears when a cat is detected
- Shows: "Cat: 0.xx" confidence score

### 📊 State Indicator (Top Left)
Current state per zone:
- **Gray**: IDLE (no cat detected)
- **Yellow**: CAT_ENTERED (cat just entered, 5s countdown)
- **Red**: CAT_INSIDE (cat confirmed inside)
- **Orange**: CAT_EXITED (cat left)
- **Blue**: COOLDOWN (60s countdown to robot dispatch)
- **Green**: DISPATCH_READY (ready to clean!)

### 📝 Event Log (Bottom Left)
Last 8 events with timestamps:
```
00:54:43 - Monitor started
00:55:12 - Cat entered zone 'Auto-Detected Litter Box'
00:55:17 - Zone 'Auto-Detected..': cat_entered → cat_inside
00:56:02 - Cat exited zone (duration: 50.2s)
00:57:02 - ROBOT DISPATCH: Litter box needs cleaning
```

### 📈 Performance Stats (Bottom Left)
- **FPS**: ~5-10 fps (display frame rate)
- **Inference**: ~200ms (cat detection time)

## How the System Works

### Normal Flow:
1. **Cat approaches** litter box
2. **Cat enters zone** → State: CAT_ENTERED (yellow)
3. **Cat stays 5+ seconds** → State: CAT_INSIDE (red)
4. **Cat exits zone** → State: CAT_EXITED (orange)
5. **60-second countdown** → State: COOLDOWN (blue, showing timer)
6. **Countdown finishes** → State: DISPATCH_READY (green)
7. **Robot signal sent** → Back to IDLE (gray)

### If Cat Returns:
- During **CAT_EXITED** or **COOLDOWN**: Cancel countdown, back to CAT_INSIDE
- Prevents false dispatches!

### False Alarm Protection:
- Cat must stay **5+ seconds** to confirm occupancy
- Quick passes (<5s) are ignored → back to IDLE

## Controls

### Keyboard:
- **'q'** - Quit the monitor
- **'r'** - Reset litter box detection (if camera moves)

## Logs

### Console/Terminal:
Real-time events printed with timestamps

### File Logs:
- `logs/events.log` - All events (append mode)
- `logs/events_2026-02-08.jsonl` - Daily JSON logs

## Robot Dispatch Signal

When a cat finishes using the litter box, you'll see:

### Console:
```
============================================================
🤖 ROBOT DISPATCH: Litter box 'Auto-Detected Litter Box' needs cleaning
   Session: litter_box_dynamic_1739006083_1
   Duration: 125.3s
   Occupancy: 60.1s
============================================================
```

### Log File:
```
[2026-02-08 01:23:45] DISPATCH: ROBOT DISPATCH: Litter box 'Auto-Detected Litter Box' needs cleaning
```

## Restart the Monitor

If you need to restart:

```bash
# Stop current instance (if running)
# Press 'q' in the window OR:
pkill -f live_monitor.py

# Start fresh
cd /path/to/cat-litter-monitor
/opt/homebrew/bin/python3.11 live_monitor.py
```

## Troubleshooting

### No video window?
- Check if Python is running: `ps aux | grep live_monitor`
- Check Terminal for error messages
- Verify camera connection: `rtsp://YOUR_CAMERA_IP/live0`

### Litter box not detected?
- Press **'r'** to reset detection
- Wait 30 seconds for warmup
- Check lighting - needs good visibility
- Model confidence threshold: 0.5

### Cat not detected?
- Cat must be **fully visible** in frame
- Check YOLOv8n model loaded correctly
- Confidence threshold: 0.5 (adjustable in settings.yaml)

### State not changing?
- Verify cat bbox overlaps litter box zone (>30%)
- Check timing settings:
  - min_occupancy: 5.0s
  - cooldown: 60.0s
- Watch event log for state transitions

## Configuration

Edit `config/settings.yaml` to adjust:

```yaml
detection:
  confidence_threshold: 0.5      # Detection sensitivity
  inference_interval: 0.2        # 5 FPS detection

timing:
  min_occupancy_seconds: 5.0     # Confirm cat inside
  cooldown_seconds: 60.0         # Wait before dispatch
  max_session_minutes: 10.0      # Session timeout
```

## What Was Built

### ✅ Completed Components:

1. **Cat Detection** (`src/detection/coreml_detector.py`)
   - YOLOv8n CoreML model
   - Runs at 5 FPS (every 200ms)
   - Tracks cat bounding boxes

2. **Zone Overlap Logic** (`src/detection/zone_manager.py`)
   - IoU-based matching
   - 30% minimum overlap threshold
   - Cat is "inside" if center point or >30% bbox overlaps zone

3. **State Machine** (`src/state/fsm.py`)
   - 6 states: IDLE, CAT_ENTERED, CAT_INSIDE, CAT_EXITED, COOLDOWN, DISPATCH_READY
   - All transitions working correctly
   - Thread-safe with RLock
   - Session tracking with full metadata

4. **Robot Signal** (`src/events/logger.py`)
   - Console output with emoji 🤖
   - File logging (logs/events.log)
   - Event callbacks for integration
   - Full session data (duration, occupancy, confidence)

5. **Live Visualization** (`live_monitor.py`) ⭐ **NEW**
   - Real-time video with overlays
   - Litter box zones (green)
   - Cat detections (yellow)
   - State indicators per zone with timers
   - Event log display (last 8 events)
   - FPS and performance metrics

## Architecture

```
RTSP Camera (YOUR_CAMERA_IP/live0)
    ↓
Frame Capture @ 5 FPS
    ↓
YOLOv8n Cat Detection (every 200ms)
    ↓
Zone Overlap Check (IoU >= 0.3)
    ↓
State Machine Update
    ├── State: IDLE → CAT_ENTERED → CAT_INSIDE → CAT_EXITED → COOLDOWN → DISPATCH
    ├── Timers: 5s confirm, 60s cooldown
    └── Callbacks: on_state_change, on_dispatch_ready, on_cat_entered, on_cat_exited
    ↓
Event Logging (console + file)
    ↓
Live Visualization (OpenCV window)
```

## Technical Details

### Detection Models:
- **Cat**: `models/yolov8n.mlpackage` (YOLOv8n CoreML, pretrained COCO)
- **Litter Box**: `models/litter_box_detector.pt` (custom-trained YOLO)

### Current Detection:
- **Litter box**: (1112, 558, 1379, 799) @ 0.64 confidence
- **Camera**: 1920x1080 @ 5 FPS
- **Processing**: ~200ms per inference

### Performance:
- **Display FPS**: 5-10 fps (real-time)
- **Detection FPS**: 5 fps (every 200ms)
- **Zone update**: Every 500ms (separate interval)
- **Memory**: Ring buffer for events (max 10)

---

**Status**: ✅ **SYSTEM OPERATIONAL**  
**Session**: good-ridge (PID 17796)  
**Started**: 2026-02-08 00:54:43  
**Uptime**: ~2 minutes

**Ready to monitor cats! 🐱 → 📦 → 🤖**
