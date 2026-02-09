#!/opt/homebrew/bin/python3.11
from ultralytics import YOLO
import cv2
import os

print("Loading best model from v2 training...")
model = YOLO('/path/to/cat-litter-monitor/runs/litter_box_v2/weights/best.pt')

print("Running inference on fresh frame...")
results = model('/tmp/test_fresh.jpg', conf=0.25, verbose=True)

# Process results
for result in results:
    print(f"\nDetections found: {len(result.boxes)}")
    
    # Print details of each detection
    for i, box in enumerate(result.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        print(f"  {i+1}. {class_name}: {conf:.3f}")
    
    # Save annotated image
    annotated = result.plot()
    output_path = '/path/to/cat-litter-monitor/test_inference_v2.jpg'
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to: {output_path}")
    
    # Check if we have high-confidence detections
    high_conf = [box for box in result.boxes if float(box.conf[0]) > 0.5]
    print(f"High confidence detections (>0.5): {len(high_conf)}")
