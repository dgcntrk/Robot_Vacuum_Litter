#!/opt/homebrew/bin/python3.11
from ultralytics import YOLO

print("Starting YOLOv8 training...")
print("Loading model...")

model = YOLO('yolov8n.pt')

print("Starting training with MPS (Apple Silicon)...")
results = model.train(
    data='/path/to/cat-litter-monitor/training_data/dataset.yaml',
    epochs=100,
    imgsz=640,
    device='mps',
    batch=8,
    patience=20,
    project='/path/to/cat-litter-monitor/runs',
    name='litter_box_v2',
    exist_ok=True,
    verbose=True
)

# Print actual training results
print(f"\n=== Training Complete ===")
print(f"Epochs completed: {results.epoch}")
print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
