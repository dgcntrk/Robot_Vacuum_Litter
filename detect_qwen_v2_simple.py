#!/opt/homebrew/bin/python3.11
"""
Qwen2.5-VL-7B litter box detection - simpler approach
"""
import sys
sys.path.insert(0, '/opt/homebrew/lib/python3.11/site-packages')

from mlx_vlm import load, generate
from PIL import Image
import cv2
import json
import re

# Paths
frame_path = "/path/to/cat-litter-monitor/frame_snapshot.jpg"
output_path = "/path/to/cat-litter-monitor/frame_qwen_v2.jpg"
model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

print("Loading model...")
model, processor = load(model_path)

print("Loading image...")
image = Image.open(frame_path)
print(f"Image size: {image.size}")

# Simpler prompt without complex formatting
prompt = "Return a JSON array with bounding box coordinates for the two cat litter boxes in this image. Format: [{\"label\": \"main_litter_box\", \"bbox_2d\": [x1, y1, x2, y2]}, {\"label\": \"secondary_litter_box\", \"bbox_2d\": [x1, y1, x2, y2]}]. The coordinates should be in pixels (0-1920 for x, 0-1080 for y). Only return the JSON array, nothing else."

print("Running inference (this may take a minute)...")
output = generate(
    model,
    processor,
    image=image,
    prompt=prompt,
    max_tokens=300,
    temperature=0.1,
    verbose=True
)

print("\n=== RAW OUTPUT ===")
print(output)
print("==================\n")

# Parse JSON
def extract_json(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

detections = extract_json(output)

if detections:
    print(f"Detected {len(detections)} boxes")
    for d in detections:
        print(f"  {d}")
    
    # Draw boxes
    img = cv2.imread(frame_path)
    
    for det in detections:
        bbox = det["bbox_2d"]
        label = det["label"]
        
        # Clamp
        bbox[0] = max(0, min(1920, bbox[0]))
        bbox[1] = max(0, min(1080, bbox[1]))
        bbox[2] = max(0, min(1920, bbox[2]))
        bbox[3] = max(0, min(1080, bbox[3]))
        
        color = (0, 255, 0) if "main" in label else (255, 0, 0)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Label
        text = label.replace("_", " ").title()
        cv2.putText(img, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Legend
    cv2.putText(img, "GREEN: Main", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "BLUE: Secondary", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"\nSaved to: {output_path}")
    
    # Print YAML
    print("\n=== YAML CONFIG ===")
    print("litter_boxes:")
    for d in detections:
        print(f"  - label: {d['label']}")
        print(f"    bbox: {d['bbox_2d']}")
else:
    print("No detections found")
