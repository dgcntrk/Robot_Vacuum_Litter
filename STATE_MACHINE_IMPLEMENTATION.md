# Cat Litter Monitor - State Machine Implementation

## Overview

The cat litter monitor system is now fully operational with:
- **Dynamic litter box detection** using trained YOLO model
- **Cat detection** using YOLOv8n (pretrained COCO model)
- **State machine** for tracking cat occupancy and triggering robot dispatch
- **Live visualization** showing all detections, states, and events in real-time

## Architecture

### 1. Cat Detection (`src/detection/coreml_detector.py`)
- Uses **YOLOv8n** CoreML model for cat detection
- Runs at **~5 FPS** (every 200ms) for efficiency
- Detects cats with confidence threshold of 0.5
- Returns bounding boxes: `[x1, y1, x2, y2]`

### 2. Litter Box Detection (`src/detection/litter_box_detector.py`)
- Uses **custom-trained YOLOv8** model: `models/litter_box_detector.pt`
- Auto-detects litter box zones dynamically
- Updates every 500ms (separate from cat detection)
- Current detection: `(1112, 558, 1379, 799)` with confidence 0.64

### 3. Zone Manager (`src/detection/zone_manager.py`)
- Manages detection zones (litter boxes)
- Matches cat detections to zones using **IoU (Intersection over Union)**
- Returns dict: `{zone_id: [list of cat detections]}`
- Minimum overlap threshold: **30%** for a cat to be "inside" a zone

### 4. State Machine (`src/state/fsm.py`)

#### States (per litter box zone):
```
IDLE → CAT_ENTERED → CAT_INSIDE → CAT_EXITED → COOLDOWN → DISPATCH_READY → IDLE
          ↓                                          ↑
          └─────── (false alarm) ──────────────────┘
                        ↓
          CAT_INSIDE ←─── (cat returns) ────── CAT_EXITED
```

#### Transitions:

| From State | To State | Condition | Duration |
|------------|----------|-----------|----------|
| **IDLE** | CAT_ENTERED | Cat detected in zone | immediate |
| **CAT_ENTERED** | CAT_INSIDE | Cat stays for 5+ seconds | 5s |
| **CAT_ENTERED** | IDLE | Cat leaves before 5s (false alarm) | <5s |
| **CAT_INSIDE** | CAT_EXITED | Cat no longer detected | immediate |
| **CAT_EXITED** | CAT_INSIDE | Cat returns during cooldown | anytime |
| **CAT_EXITED** | COOLDOWN | Cat stays away for 60s | 60s |
| **COOLDOWN** | DISPATCH_READY | Cooldown complete | immediate |
| **DISPATCH_READY** | IDLE | Robot signal sent | immediate |

#### Timing Configuration:
- **min_occupancy_seconds**: 5.0 (must see cat for 5s to confirm)
- **cooldown_seconds**: 60.0 (wait 60s after exit before dispatch)
- **max_session_minutes**: 10.0 (timeout if cat sits too long)

### 5. Robot Signal (`src/events/logger.py`)

When state transitions to **DISPATCH_READY**:

1. **Console output**:
   ```
   ============================================================
   🤖 ROBOT DISPATCH: Litter box 'Auto-Detected Litter Box' needs cleaning
      Session: litter_box_dynamic_1739006083_1
      Duration: 125.3s
      Occupancy: 60.1s
   ============================================================
   ```

2. **Log file** (`logs/events.log`):
   ```
   [2026-02-08 01:23:45] DISPATCH: ROBOT DISPATCH: Litter box 'Auto-Detected Litter Box' needs cleaning
   ```

3. **Event callback** (for robot integration):
   - Calls `state_machine.on_dispatch_ready(session)`
   - Session includes full tracking data (enter/exit times, confidence, etc.)

### 6. Live Visualization (`live_monitor.py`)

**Now running in background (session: good-ridge)**

The visualization window shows:
- ✅ **Litter box zones** (green rectangle with confidence score)
- ✅ **Cat detections** (yellow bounding boxes with confidence)
- ✅ **State indicators** per zone:
  - Gray = IDLE
  - Yellow = CAT_ENTERED (with countdown to confirmation)
  - Red = CAT_INSIDE
  - Orange = CAT_EXITED
  - Blue-orange = COOLDOWN (with countdown timer)
  - Green = DISPATCH_READY
- ✅ **Event log** (last 8 events in bottom-left)
- ✅ **FPS & inference time** (performance metrics)

## Running the System

### Start Live Monitor:
```bash
cd /path/to/cat-litter-monitor
/opt/homebrew/bin/python3.11 live_monitor.py
```

### OR use the full system with dashboard:
```bash
/opt/homebrew/bin/python3.11 src/main.py
```

### Controls:
- Press **'q'** to quit
- Press **'r'** to reset litter box detection (if camera moves)

## File Locations

### Core Files:
- **Main app**: `src/main.py`
- **Live monitor**: `live_monitor.py` ⭐ (NEW)
- **State machine**: `src/state/fsm.py` ✅ (COMPLETE)
- **Cat detector**: `src/detection/coreml_detector.py` ✅
- **Zone manager**: `src/detection/zone_manager.py` ✅
- **Event logger**: `src/events/logger.py` ✅

### Configuration:
- `config/settings.yaml` - System configuration
- `src/config.py` - Pydantic configuration models

### Models:
- `models/yolov8n.mlpackage` - Cat detector (YOLOv8n CoreML)
- `models/litter_box_detector.pt` - Litter box detector (custom YOLO)

### Logs:
- `logs/events.log` - Event log with all state transitions
- `logs/events_YYYY-MM-DD.jsonl` - Daily JSON logs

## Current Status

### ✅ COMPLETE Features:
1. **Cat Detection** - YOLOv8n running at 5 FPS
2. **Litter Box Detection** - Dynamic zone detection with YOLO
3. **Zone Overlap Logic** - IoU-based matching (30% threshold)
4. **State Machine** - Full FSM with all 6 states
5. **State Transitions** - All transitions working correctly
6. **Robot Signal** - Console + log file + callback
7. **Live Visualization** - Real-time display with all overlays
8. **Event Logging** - File and in-memory logging

### 🎯 System Performance:
- **Camera**: Connected to `rtsp://YOUR_CAMERA_IP/live0`
- **Litter box detected**: `(1112, 558, 1379, 799)` @ 0.64 confidence
- **Inference speed**: ~200ms per frame (5 FPS)
- **Display FPS**: Real-time visualization

### 🔧 Next Steps (Optional Enhancements):
- [ ] Robot integration (Roborock API or similar)
- [ ] Web dashboard for remote monitoring
- [ ] SMS/email notifications on dispatch
- [ ] Multi-box support (multiple litter boxes)
- [ ] Session analytics and reports
- [ ] Cat identification (if multiple cats)

## Testing the System

### Manual Test Scenarios:

1. **Cat enters box** → Should see "CAT_ENTERED" after first detection
2. **Cat stays 5+ seconds** → Should transition to "CAT_INSIDE"
3. **Cat exits** → Should see "CAT_EXITED"
4. **Wait 60 seconds** → Should see "COOLDOWN" → "DISPATCH_READY"
5. **Cat returns during cooldown** → Should cancel and return to "CAT_INSIDE"

### Expected Console Output:
```
[2026-02-08 01:23:00] ENTER: Cat entered zone 'Auto-Detected Litter Box'
[2026-02-08 01:23:05] STATE: Zone 'Auto-Detected Litter Box': cat_entered → cat_inside
[2026-02-08 01:23:45] EXIT: Cat exited zone 'Auto-Detected Litter Box' (duration: 45.2s)
[2026-02-08 01:24:45] DISPATCH: ROBOT DISPATCH: Litter box needs cleaning
```

## Technical Details

### Detection Pipeline:
```
1. RTSP Camera (rtsp://YOUR_CAMERA_IP/live0)
   ↓
2. Frame capture @ 5 FPS
   ↓
3. YOLOv8n cat detection (every 200ms)
   ↓
4. Zone overlap check (IoU >= 0.3)
   ↓
5. State machine update
   ↓
6. Event logging & visualization
```

### State Machine Thread Safety:
- Uses `threading.RLock()` for safe concurrent access
- Callbacks executed in detection thread
- Event log uses thread-safe deque

### Performance Optimizations:
- Separate intervals for cat detection (200ms) and zone updates (500ms)
- Frame skipping for smooth visualization
- In-memory event log with ring buffer (max 10 entries)
- Lazy model loading for faster startup

## Troubleshooting

### Camera not connecting:
- Check RTSP URL: `rtsp://YOUR_CAMERA_IP/live0`
- Verify network connectivity
- Check camera power and settings

### Litter box not detected:
- Press 'r' to reset detection
- Ensure good lighting
- Check model file: `models/litter_box_detector.pt`
- Model confidence threshold: 0.5

### Cat not detected:
- Check YOLOv8n model: `models/yolov8n.mlpackage`
- Adjust confidence threshold in settings.yaml
- Verify cat is visible in camera frame

### State machine not transitioning:
- Check timing configuration in settings.yaml
- Verify zone overlap (need >30% IoU)
- Check event log for false alarms

---

**System Status**: ✅ **OPERATIONAL**  
**Current Session**: `good-ridge` (PID 17796)  
**Live Monitor**: **RUNNING**  
**Last Update**: 2026-02-08 00:54:43
