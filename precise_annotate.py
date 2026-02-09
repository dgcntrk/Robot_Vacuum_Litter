#!/usr/bin/env python3
"""Systematic approach to find precise litter box coordinates using binary search with vision."""

import cv2
import numpy as np
import os

# Load the original frame
frame_path = '/path/to/cat-litter-monitor/frame_snapshot.jpg'
frame = cv2.imread(frame_path)
h, w = frame.shape[:2]
print(f"Frame dimensions: {w}x{h}")

# Reference coordinates to start with (from previous attempts)
# Main box roughly: [700, 540, 1050, 980]
# Secondary box roughly: [1050, 680, 1320, 950]

# Create crops for systematic analysis
crops_dir = '/path/to/cat-litter-monitor/crops'
os.makedirs(crops_dir, exist_ok=True)

def save_crop(img, name, x1, y1, x2, y2):
    """Save a crop and draw reference lines."""
    crop = img[y1:y2, x1:x2].copy()
    # Add border and text
    bordered = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(50, 50, 50))
    cv2.putText(bordered, f"{name}: [{x1},{y1},{x2},{y2}]", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path = os.path.join(crops_dir, f"{name}.jpg")
    cv2.imwrite(path, bordered)
    return path

# Strategy: Create targeted crops around the suspected regions
# We'll create crops that show specific edges to help identify exact boundaries

print("\n=== Creating systematic crops for analysis ===\n")

# MAIN BOX (left side) - roughly x:700-1050, y:540-980
# Let's create crops to find each edge

# 1. Main box - LEFT edge region (vertical strip)
print("1. Main box LEFT edge region")
save_crop(frame, "main_left_edge", 600, 400, 850, 1000)

# 2. Main box - RIGHT edge region  
print("2. Main box RIGHT edge region")
save_crop(frame, "main_right_edge", 950, 400, 1150, 1000)

# 3. Main box - TOP edge region
print("3. Main box TOP edge region")
save_crop(frame, "main_top_edge", 650, 400, 1100, 650)

# 4. Main box - BOTTOM edge region
print("4. Main box BOTTOM edge region")
save_crop(frame, "main_bottom_edge", 650, 850, 1100, 1080)

# SECONDARY BOX (right side) - roughly x:1050-1320, y:680-950
# 5. Secondary box - LEFT edge region
print("5. Secondary box LEFT edge region")
save_crop(frame, "sec_left_edge", 950, 550, 1150, 1000)

# 6. Secondary box - RIGHT edge region
print("6. Secondary box RIGHT edge region")
save_crop(frame, "sec_right_edge", 1200, 550, 1400, 1000)

# 7. Secondary box - TOP edge region
print("7. Secondary box TOP edge region")
save_crop(frame, "sec_top_edge", 1000, 550, 1350, 750)

# 8. Secondary box - BOTTOM edge region
print("8. Secondary box BOTTOM edge region")
save_crop(frame, "sec_bottom_edge", 1000, 850, 1350, 1000)

# Create a full-context crop showing both boxes
print("9. Full litter box region")
save_crop(frame, "full_region", 600, 400, 1400, 1000)

print(f"\nCrops saved to: {crops_dir}")
print("Now analyze each crop to determine exact edge positions...")
