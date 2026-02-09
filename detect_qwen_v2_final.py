#!/opt/homebrew/bin/python3.11
"""
Qwen2.5-VL-7B litter box detection - refined prompt without examples
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

# Refined prompt - no example coordinates
prompt = """This is a 1920x1080 pixel IR night vision camera frame. There are two cat litter boxes under a desk/table.

Task: Detect each litter box and return ONLY a JSON array with precise bounding box coordinates.

Requirements:
- The bounding boxes should tightly wrap just the litter box containers themselves
- Do NOT include the floor mat, the floor, or any area below the bottom rim of each box
- Left litter box is larger (main), right one is smaller (secondary)

Return format:
[{"label": "main_litter_box", "bbox_2d": [x1, y1, x2, y2]}, {"label": "secondary_litter_box", "bbox_2d": [x3, y3, x4, y4]}]

All coordinates in pixels, max x=1920, max y=1080. Return ONLY the JSON array."""

print("Running inference (this may take 1-2 minutes)...")
print("=" * 60)
output = generate(
    model,
    processor,
    image=frame_path,
    prompt=prompt,
    max_tokens=400,
    temperature=0.1,
    verbose=True
)

print("\n" + "=" * 60)
print("RAW OUTPUT:")
print("=" * 60)
print(output)
print("=" * 60)

# Parse JSON
def extract_json(text):
    # Try to find JSON array
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            # Try to clean up common issues
            cleaned = match.group().replace("'", '"')
            try:
                return json.loads(cleaned)
            except:
                pass
    return None

detections = extract_json(output)

if detections and len(detections) >= 2:
    print(f"\n✓ Detected {len(detections)} litter boxes:")
    for d in detections:
        print(f"  - {d['label']}: {d['bbox_2d']}")
    
    # Draw boxes
    img = cv2.imread(frame_path)
    
    for det in detections:
        bbox = det["bbox_2d"]
        label = det["label"]
        
        # Clamp coordinates
        bbox[0] = max(0, min(1920, int(bbox[0])))
        bbox[1] = max(0, min(1080, int(bbox[1])))
        bbox[2] = max(0, min(1920, int(bbox[2])))
        bbox[3] = max(0, min(1080, int(bbox[3])))
        
        # Color based on label
        if "main" in label.lower():
            color = (0, 255, 0)  # GREEN for main
        else:
            color = (255, 0, 0)  # BLUE for secondary
        
        # Draw rectangle (3px)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Label with background
        text = label.replace("_", " ").title()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background
        cv2.rectangle(img, (bbox[0], bbox[1] - text_h - 10), 
                      (bbox[0] + text_w + 10, bbox[1]), color, -1)
        # Text
        cv2.putText(img, text, (bbox[0] + 5, bbox[1] - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Legend at top-left
    legend_y = 30
    cv2.putText(img, "GREEN: Main Litter Box", (10, legend_y), font, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "BLUE: Secondary Litter Box", (10, legend_y + 25), font, 0.6, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"\n✓ Saved annotated image to: {output_path}")
    
    # Print YAML config
    print("\n" + "=" * 60)
    print("YAML CONFIG SNIPPET:")
    print("=" * 60)
    print("litter_boxes:")
    for d in detections:
        bbox = d['bbox_2d']
        print(f"  - label: {d['label']}")
        print(f"    bbox: [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]")
    print("=" * 60)
    
else:
    print(f"\n✗ No valid detections found (got {len(detections) if detections else 0})")
    print("Full output was:")
    print(output)
