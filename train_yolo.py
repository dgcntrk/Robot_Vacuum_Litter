from ultralytics import YOLO
import time
import os

# Record start time
start_time = time.time()

# Load the base model
model_path = '/path/to/cat-litter-monitor/yolov8n.pt'
model = YOLO(model_path)

# Train the model
try:
    results = model.train(
        data='/path/to/cat-litter-monitor/training_data/dataset.yaml',
        epochs=100,
        imgsz=640,
        device='mps',  # Apple Silicon GPU
        batch=8,
        patience=20,  # Early stopping
        project='/path/to/cat-litter-monitor/runs',
        name='litter_box_detect',
        exist_ok=True
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*50}")
    print(f"Training time: {hours}h {minutes}m {seconds}s")
    print(f"Best model saved to: {results.best}")
    
    # Get final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\nFinal Metrics:")
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
except Exception as e:
    print(f"MPS failed with error: {e}")
    print("Falling back to CPU...")
    
    model = YOLO(model_path)
    results = model.train(
        data='/path/to/cat-litter-monitor/training_data/dataset.yaml',
        epochs=100,
        imgsz=640,
        device='cpu',
        batch=8,
        patience=20,
        project='/path/to/cat-litter-monitor/runs',
        name='litter_box_detect',
        exist_ok=True
    )
    
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE (CPU)")
    print(f"{'='*50}")
    print(f"Training time: {hours}h {minutes}m {seconds}s")
    print(f"Best model saved to: {results.best}")
