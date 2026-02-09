# Quick Start Guide - Dynamic Litter Box Detection

## What's New?

The system now automatically detects litter boxes using a trained YOLOv8 model. No manual zone configuration needed!

## Running the System

### 1. Start the monitor:
```bash
cd /path/to/cat-litter-monitor
/opt/homebrew/bin/python3.11 src/main.py
```

### 2. Expected startup behavior:
```
✓ Loading CoreML cat detection model (yolov8n.mlpackage)
✓ Loading CoreML litter box model (litter_box_detector.mlpackage)
✓ Connecting to RTSP camera: rtsp://YOUR_CAMERA_IP/live0
✓ Dynamic litter box detection enabled (YOLO + Vision fallbacks)
✓ Running initial litter box detection...
  → Litter box detected at (x1, y1, x2, y2) with confidence 0.XX
✓ Zone updated dynamically
✓ Starting cat detection loop (5 FPS)
```

## How It Works

### Dynamic Zone Detection
1. **On startup:** YOLO model scans the frame and detects litter box
2. **Every 30 seconds:** Re-runs detection to track any movement
3. **Real-time tracking:** Smooths detections to reduce jitter
4. **Fallback:** Uses static zones from config if YOLO fails

### Cat Monitoring
1. YOLOv8n detects cats in each frame (5 FPS)
2. Checks if cat overlaps with detected litter box zone
3. State machine tracks: idle → approaching → inside → exiting → cooldown
4. Logs events: cat entered, cat exited, duration, etc.

## Testing

### Quick test (standalone):
```bash
/opt/homebrew/bin/python3.11 test_litter_box_detection.py
```

Expected output:
```
✅ Litter box detected!
  Method: yolo
  Confidence: 0.703
  BBox: (1073, 696, 1330, 966)
```

## Configuration

### Enable/disable dynamic detection:
Edit `config/settings.yaml`:

```yaml
detection:
  # Set to true for YOLO dynamic detection
  # Set to false to use static zones
  dynamic_zones: true
  
  # Path to litter box model
  litter_box_model_path: "models/litter_box_detector.mlpackage"
```

### Adjust detection thresholds:
In `src/main.py` or `src/detection/zone_manager.py`, modify `LitterBoxDetector` parameters:

```python
LitterBoxDetector(
    min_area_ratio=0.02,       # Lower = allow smaller boxes
    max_area_ratio=0.5,        # Higher = allow larger boxes
    aspect_ratio_range=(0.4, 3.0),  # Wider range = more flexible shapes
)
```

## Troubleshooting

### "No litter box detected"
- Check camera view - is litter box visible?
- Lower `min_area_ratio` if box is small: `0.02` → `0.01`
- Check logs for detection method used (YOLO vs fallback)
- Verify model exists: `ls models/litter_box_detector.mlpackage`

### "Detection using edge_detection instead of yolo"
- Check if area ratio is filtering detections (see logs)
- Verify YOLO model loaded successfully (look for "YOLO litter box detector loaded")
- Try lowering confidence threshold in detector init

### "Model not loading"
```bash
# Verify model files exist
ls -lh models/
# Should show:
#   litter_box_detector.mlpackage/
#   litter_box_detector.pt
#   yolov8n.mlpackage/

# Check CoreML tools installed
/opt/homebrew/bin/python3.11 -c "import coremltools; print('OK')"
```

### View detection visualization:
Enable visualization in `config/settings.yaml`:

```yaml
visualization:
  enabled: true
  show_zones: true        # Green box = YOLO detected zone
  show_detections: true   # Red box = cat detections
```

## Performance

- **Detection latency:** ~1.4 seconds per YOLO inference
- **Detection interval:** 30 seconds (configurable)
- **Cat detection:** ~5 FPS real-time
- **CPU usage:** Low (CoreML uses Apple Neural Engine)
- **Memory:** ~200MB with both models loaded

## Files Changed

| File | Change |
|------|--------|
| `models/litter_box_detector.mlpackage/` | New trained model (CoreML) |
| `models/litter_box_detector.pt` | New trained model (PyTorch) |
| `config/settings.yaml` | Enabled `dynamic_zones`, added `litter_box_model_path` |
| `src/detection/litter_box_detector.py` | Added YOLO detection method |
| `src/detection/coreml_detector.py` | Enhanced prediction parsing |
| `src/detection/zone_manager.py` | Accept model path parameter |
| `src/main.py` | Pass model path from config |

## Next Steps

1. **Test with live camera feed:** Start the main app and observe dynamic zone detection
2. **Monitor logs:** Check that YOLO is being used (not fallback methods)
3. **Tune thresholds:** Adjust area/aspect ratios if needed
4. **Enable robot dispatch:** Set `robot.enabled: true` in config (Phase 2)

---

**Integration completed:** 2026-02-08  
**See:** `INTEGRATION_SUMMARY.md` for technical details
