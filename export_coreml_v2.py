#!/opt/homebrew/bin/python3.11
from ultralytics import YOLO

print("Exporting best model to CoreML...")
model = YOLO('/path/to/cat-litter-monitor/runs/litter_box_v2/weights/best.pt')

# Export to CoreML
model.export(format='coreml', imgsz=640)

print("\nCoreML export complete!")
