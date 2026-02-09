#!/opt/homebrew/bin/python3.11
"""
Qwen2.5-VL-7B litter box detection with refined prompt
"""
import mlx.core as mx
from mlx_vlm import load, generate
from PIL import Image
import cv2
import json
import re
import os

# Paths
frame_path = "/path/to/cat-litter-monitor/frame_snapshot.jpg"
output_path = "/path/to/cat-litter-monitor/frame_qwen_v2.jpg"
model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# Image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Refined prompt
prompt = """This is a 1920x1080 pixel IR night vision camera frame. There are two cat litter boxes under a desk/table. Detect each litter box and return ONLY a JSON array with precise bounding box coordinates. The bounding boxes should tightly wrap just the litter box containers themselves — do NOT include the floor mat, the floor, or any area below the bottom rim of each box. Return format: [{"label": "main_litter_box", "bbox_2d": [x1, y1, x2, y2]}, {"label": "secondary_litter_box", "bbox_2d": [x1, y1, x2, y2]}]. All coordinates in pixels, max x=1920, max y=1080."""

print("=" * 60)
print("Qwen2.5-VL-7B Litter Box Detection v2")
print("=" * 60)

# Load model
print("\n[1/5] Loading model:", model_path)
model, processor = load(model_path)
print("    Model loaded successfully")

# Load image
print("\n[2/5] Loading image:", frame_path)
image = Image.open(frame_path)
print(f"    Image size: {image.size}")

# Generate with proper Qwen2.5-VL image token format
print("\n[3/5] Running inference...")

# Qwen2.5-VL uses special vision tokens
# Format: <|vision_start|><|image_pad|><|vision_end|>
formatted_prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"

output = generate(
    model,
    processor,
    image=image,
    prompt=formatted_prompt,
    max_tokens=500,
    temperature=0.1,  # Low temperature for consistent JSON
    verbose=False
)

print("\n[4/5] Parsing response...")
print("    Raw output:")
print("    " + "-" * 50)
for line in output.split('\n'):
    print(f"    {line}")
print("    " + "-" * 50)

# Extract JSON from response
def extract_json(text):
    """Extract JSON array from text response"""
    # Try to find JSON array pattern
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON between code blocks
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None

detections = extract_json(output)

if detections is None:
    print("    ERROR: Could not parse JSON from response")
    # Fallback: try to extract using regex for bbox coordinates
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(bbox_pattern, output)
    if len(matches) >= 2:
        print("    Using fallback regex parsing...")
        detections = [
            {"label": "main_litter_box", "bbox_2d": [int(x) for x in matches[0]]},
            {"label": "secondary_litter_box", "bbox_2d": [int(x) for x in matches[1]]}
        ]
    else:
        print("    ERROR: Fallback parsing also failed")
        detections = []

print(f"    Detected {len(detections)} litter boxes")

# Clamp coordinates and prepare for drawing
def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

for det in detections:
    bbox = det["bbox_2d"]
    # Clamp to image bounds
    bbox[0] = clamp(bbox[0], 0, IMG_WIDTH)   # x1
    bbox[1] = clamp(bbox[1], 0, IMG_HEIGHT)  # y1
    bbox[2] = clamp(bbox[2], 0, IMG_WIDTH)   # x2
    bbox[3] = clamp(bbox[3], 0, IMG_HEIGHT)  # y2
    print(f"    {det['label']}: {bbox}")

# Draw on image with cv2
print("\n[5/5] Drawing bounding boxes...")
img_cv = cv2.imread(frame_path)
if img_cv is None:
    print("    ERROR: Could not load image with cv2")
else:
    # Colors (BGR format for OpenCV)
    COLOR_GREEN = (0, 255, 0)   # Main litter box
    COLOR_BLUE = (255, 0, 0)    # Secondary litter box
    
    for det in detections:
        bbox = det["bbox_2d"]
        label = det["label"]
        x1, y1, x2, y2 = bbox
        
        if label == "main_litter_box":
            color = COLOR_GREEN
        else:
            color = COLOR_BLUE
        
        # Draw rectangle (3px thickness)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label_text = label.replace("_", " ").title()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Draw label text (white)
        cv2.putText(img_cv, label_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Draw legend at top-left
    legend_y = 30
    legend_x = 10
    line_height = 25
    
    cv2.rectangle(img_cv, (legend_x, legend_y - 20), (legend_x + 250, legend_y + line_height * 2 + 10), (0, 0, 0), -1)
    cv2.putText(img_cv, "GREEN: Main Litter Box", (legend_x + 5, legend_y), font, 0.6, COLOR_GREEN, 2)
    cv2.putText(img_cv, "BLUE: Secondary Litter Box", (legend_x + 5, legend_y + line_height), font, 0.6, COLOR_BLUE, 2)
    
    # Save output
    cv2.imwrite(output_path, img_cv)
    print(f"    Saved to: {output_path}")

# Print YAML config snippet
print("\n" + "=" * 60)
print("YAML CONFIG SNIPPET")
print("=" * 60)

if detections:
    yaml_lines = []
    yaml_lines.append("litter_boxes:")
    for det in detections:
        label = det["label"]
        bbox = det["bbox_2d"]
        yaml_lines.append(f"  - label: {label}")
        yaml_lines.append(f"    bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    yaml_output = "\n".join(yaml_lines)
    print(yaml_output)
else:
    print("# No detections to include in config")
    print("litter_boxes: []")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
