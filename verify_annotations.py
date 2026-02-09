#!/usr/bin/env python3
import cv2
import os

# Configuration
base_dir = "/path/to/cat-litter-monitor/training_data"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
output_dir = base_dir

# Sample frames to verify
samples = [
    ("frame_0005.jpg", "verify_sample_1.jpg"),  # Early frame (IR/night)
    ("frame_0025.jpg", "verify_sample_2.jpg"),  # Middle frame (IR/night)
    ("frame_0063.jpg", "verify_sample_3.jpg"),  # Late frame (latest, lights-on)
]

# Class names and colors
class_names = {
    0: "litter_box_main",
    1: "litter_box_secondary"
}

colors = {
    0: (0, 255, 0),    # GREEN for class 0
    1: (255, 0, 0)     # BLUE for class 1
}

# Image dimensions (1920x1080)
img_width = 1920
img_height = 1080

def draw_annotations(image_path, label_path, output_path):
    """Draw YOLO annotations on an image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        cv2.imwrite(output_path, img)
        return True
    
    # Read label file
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Process each annotation
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # Convert normalized coords to pixel coords
        x_center = int(cx * img_width)
        y_center = int(cy * img_height)
        width = int(w * img_width)
        height = int(h * img_height)
        
        # Calculate top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Get color and class name
        color = colors.get(class_id, (128, 128, 128))
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (255, 255, 255), thickness)
        
        print(f"  Drew {class_name} at ({x1},{y1}) to ({x2},{y2})")
    
    # Save output image
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    return True

# Process each sample
for img_name, out_name in samples:
    img_path = os.path.join(images_dir, img_name)
    label_name = img_name.replace('.jpg', '.txt')
    label_path = os.path.join(labels_dir, label_name)
    out_path = os.path.join(output_dir, out_name)
    
    print(f"\nProcessing {img_name}...")
    draw_annotations(img_path, label_path, out_path)

print("\nDone! All samples generated.")
