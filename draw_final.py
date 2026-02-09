#!/usr/bin/env python3
"""Draw final precise annotations."""

import cv2
import numpy as np

# Load the original frame
frame_path = '/path/to/cat-litter-monitor/frame_snapshot.jpg'
frame = cv2.imread(frame_path)

# Precise coordinates from systematic analysis
main_box = [668, 448, 1055, 1038]  # [x1, y1, x2, y2]
sec_box = [1070, 688, 1330, 978]

print(f"Main box: {main_box}")
print(f"Secondary box: {sec_box}")

# Draw GREEN rectangle for main box (3px thickness)
cv2.rectangle(frame, 
              (main_box[0], main_box[1]), 
              (main_box[2], main_box[3]), 
              (0, 255, 0),  # GREEN in BGR
              3)

# Draw BLUE rectangle for secondary box (3px thickness)
cv2.rectangle(frame, 
              (sec_box[0], sec_box[1]), 
              (sec_box[2], sec_box[3]), 
              (255, 0, 0),  # BLUE in BGR
              3)

# Add labels with backgrounds
def add_label(img, text, pos, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw semi-transparent background
    cv2.rectangle(img, 
                  (pos[0], pos[1] - text_h - 5),
                  (pos[0] + text_w + 10, pos[1] + 5),
                  (0, 0, 0),
                  -1)
    
    # Draw text
    cv2.putText(img, text, (pos[0] + 5, pos[1]), 
                font, font_scale, color, thickness)

# Add labels
add_label(frame, "Main", (main_box[0] + 10, main_box[1] + 35), (0, 255, 0))
add_label(frame, "Secondary", (sec_box[0] + 10, sec_box[1] + 35), (255, 0, 0))

# Create legend at top-left
legend_x, legend_y = 20, 30
cv2.rectangle(frame, (10, 10), (300, 90), (40, 40, 40), -1)
cv2.rectangle(frame, (legend_x, legend_y), (legend_x + 40, legend_y + 20), (0, 255, 0), 2)
cv2.putText(frame, "Main Box", (legend_x + 50, legend_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.rectangle(frame, (legend_x, legend_y + 30), (legend_x + 40, legend_y + 50), (255, 0, 0), 2)
cv2.putText(frame, "Secondary Box", (legend_x + 50, legend_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Save
output_path = '/path/to/cat-litter-monitor/frame_final.jpg'
cv2.imwrite(output_path, frame)
print(f"\nSaved annotated frame to: {output_path}")

# Print YAML config
print("\n" + "="*60)
print("YAML Configuration Snippet:")
print("="*60)
print(f"""
zones:
  main:
    bbox: [{main_box[0]}, {main_box[1]}, {main_box[2]}, {main_box[3]}]  # [x1, y1, x2, y2]
    label: "Main Litter Box"
    
  secondary:
    bbox: [{sec_box[0]}, {sec_box[1]}, {sec_box[2]}, {sec_box[3]}]  # [x1, y1, x2, y2]
    label: "Secondary Litter Box"
""")
print("="*60)
