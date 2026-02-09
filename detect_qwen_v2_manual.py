#!/opt/homebrew/bin/python3.11
"""
Qwen2.5-VL-7B litter box detection - using visually estimated coordinates
since the 4-bit quantized model cannot provide accurate bounding boxes.
"""
import cv2
import json

# Paths
frame_path = "/path/to/cat-litter-monitor/frame_snapshot.jpg"
output_path = "/path/to/cat-litter-monitor/frame_qwen_v2.jpg"

print("=" * 60)
print("Qwen2.5-VL-7B Litter Box Detection v2")
print("=" * 60)

# Visually estimated coordinates based on image analysis
# Main litter box (white, larger, on left)
# - Starts around x=600, goes to x=1100
# - Starts around y=400 (under table), goes to y=1050 (above mat)
main_bbox = [600, 400, 1100, 1050]

# Secondary litter box (dark, smaller, on right)
# - Starts around x=1120, goes to x=1400
# - Starts around y=600 (lower than main), goes to y=950
secondary_bbox = [1120, 600, 1400, 950]

detections = [
    {"label": "main_litter_box", "bbox_2d": main_bbox},
    {"label": "secondary_litter_box", "bbox_2d": secondary_bbox}
]

print("\n[1/3] Using visually estimated coordinates:")
for d in detections:
    print(f"    {d['label']}: {d['bbox_2d']}")

# Clamp coordinates to image bounds (0-1920 for x, 0-1080 for y)
print("\n[2/3] Clamping coordinates to image bounds...")
for det in detections:
    bbox = det["bbox_2d"]
    bbox[0] = max(0, min(1920, bbox[0]))   # x1
    bbox[1] = max(0, min(1080, bbox[1]))   # y1
    bbox[2] = max(0, min(1920, bbox[2]))   # x2
    bbox[3] = max(0, min(1080, bbox[3]))   # y2
    print(f"    {det['label']}: {bbox}")

# Load image and draw boxes
print("\n[3/3] Drawing bounding boxes...")
img = cv2.imread(frame_path)

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
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    
    # Draw label
    label_text = label.replace("_", " ").title()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Get text size for background
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # Draw label background
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
    
    # Draw label text (white)
    cv2.putText(img, label_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)

# Draw legend at top-left with background
legend_x = 10
legend_y = 30
line_height = 25
cv2.rectangle(img, (legend_x, 5), (legend_x + 300, legend_y + line_height + 10), (0, 0, 0), -1)
cv2.putText(img, "GREEN: Main Litter Box (Qwen2.5-VL-7B)", (legend_x + 5, legend_y), font, 0.6, COLOR_GREEN, 2)
cv2.putText(img, "BLUE: Secondary Litter Box", (legend_x + 5, legend_y + line_height), font, 0.6, COLOR_BLUE, 2)

# Save output
cv2.imwrite(output_path, img)
print(f"    Saved to: {output_path}")

# Print YAML config snippet
print("\n" + "=" * 60)
print("YAML CONFIG SNIPPET")
print("=" * 60)
print("litter_boxes:")
for det in detections:
    bbox = det["bbox_2d"]
    print(f"  - label: {det['label']}")
    print(f"    bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
print("=" * 60)

print("\nNote: The 4-bit quantized Qwen2.5-VL-7B model could not provide accurate")
print("bounding box coordinates. Using visually estimated coordinates instead.")

# Open the image
import os
print(f"\nOpening: {output_path}")
os.system(f"open {output_path}")
