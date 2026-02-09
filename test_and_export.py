from ultralytics import YOLO
import os

# Load the best model
model_path = '/path/to/cat-litter-monitor/runs/litter_box_detect/weights/best.pt'
model = YOLO(model_path)

# Run test inference on a few validation images with lower conf threshold
val_images_dir = '/path/to/cat-litter-monitor/training_data/val/images'
val_images = sorted([f for f in os.listdir(val_images_dir) if f.endswith('.jpg')])[:3]

for img_name in val_images:
    test_image = os.path.join(val_images_dir, img_name)
    print(f"\nRunning inference on: {img_name}")
    
    # Run inference with lower confidence threshold
    results = model(test_image, conf=0.25)
    
    # Print results
    for result in results:
        if len(result.boxes) > 0:
            print(f"  Detected {len(result.boxes)} objects:")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                print(f"    - {class_name}: {conf:.3f}")
        else:
            print("  No detections")

# Now export to CoreML
print("\n" + "="*50)
print("Exporting to CoreML format...")
print("="*50)

model = YOLO(model_path)
model.export(format='coreml', imgsz=640)

print("\nExport complete!")
print("CoreML model should be at: /path/to/cat-litter-monitor/runs/litter_box_detect/weights/best.mlpackage")
