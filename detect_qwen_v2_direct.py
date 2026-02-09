#!/opt/homebrew/bin/python3.11
"""
Qwen2.5-VL-7B litter box detection - direct coordinate extraction
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

# More direct prompt - ask for description first then coordinates
prompt = """Analyze this 1920x1080 IR night vision image. I see two cat litter boxes under a table.

First, describe what you see: Where are the litter boxes located? What are their approximate pixel positions?

Then provide EXACT bounding box coordinates in this exact format:
main_litter_box: [x1, y1, x2, y2]
secondary_litter_box: [x1, y1, x2, y2]

The left box is larger and white. The right box is smaller and dark. Both are under the wooden table.
Coordinates must be accurate for a 1920x1080 image."""

print("Running inference...")
print("=" * 60)
output = generate(
    model,
    processor,
    image=frame_path,
    prompt=prompt,
    max_tokens=500,
    temperature=0.1,
    verbose=True
)

print("\n" + "=" * 60)
print("RAW OUTPUT:")
print("=" * 60)
print(output)
print("=" * 60)

# Parse the coordinate format
def parse_coords(text):
    """Parse coordinates from text like 'main_litter_box: [x1, y1, x2, y2]'"""
    detections = []
    
    # Look for main litter box
    main_match = re.search(r'main[^:]*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text, re.IGNORECASE)
    if main_match:
        coords = [int(x) for x in main_match.groups()]
        detections.append({"label": "main_litter_box", "bbox_2d": coords})
    
    # Look for secondary litter box
    sec_match = re.search(r'secondary[^:]*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text, re.IGNORECASE)
    if sec_match:
        coords = [int(x) for x in sec_match.groups()]
        detections.append({"label": "secondary_litter_box", "bbox_2d": coords})
    
    return detections

detections = parse_coords(output)

# Also try JSON fallback
if len(detections) < 2:
    json_match = re.search(r'\[[\s\S]*?\]', output)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if isinstance(json_data, list) and len(json_data) >= 2:
                detections = json_data
        except:
            pass

if len(detections) >= 2:
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
    cv2.rectangle(img, (5, 5), (300, 70), (0, 0, 0), -1)
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
    
    # Open the image
    import os
    os.system(f"open {output_path}")
    
else:
    print(f"\n✗ Found only {len(detections)} detections, need 2")
    print("Trying to extract any bounding boxes found...")
    
    # Try to find any 4-number sequences that look like bounding boxes
    all_boxes = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', output)
    print(f"Found {len(all_boxes)} potential bounding boxes: {all_boxes}")
