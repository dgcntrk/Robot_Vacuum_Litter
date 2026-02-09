#!/usr/bin/env python3
"""
RTSP Camera Frame Capture for YOLO Training Data
Captures frames every 2 seconds and generates YOLO annotations.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Configuration
RTSP_URL = "rtsp://YOUR_CAMERA_IP/live0"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CAPTURE_INTERVAL = 2  # seconds
DEFAULT_MAX_FRAMES = 50

# Paths
BASE_DIR = Path("/path/to/cat-litter-monitor/training_data")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"

# Bounding boxes in pixels [x1, y1, x2, y2]
BOXES = {
    0: {"name": "litter_box_main", "coords": [668, 448, 1055, 1038]},
    1: {"name": "litter_box_secondary", "coords": [1070, 688, 1330, 978]},
}


def pixel_to_yolo(box, img_w, img_h):
    """
    Convert pixel coordinates [x1, y1, x2, y2] to YOLO format [cx, cy, w, h] normalized.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    
    # Normalize to 0-1
    return (
        center_x / img_w,
        center_y / img_h,
        width / img_w,
        height / img_h
    )


def create_annotation(frame_number):
    """Create YOLO annotation file for a frame."""
    annotation_lines = []
    
    for class_id, box_info in BOXES.items():
        cx, cy, w, h = pixel_to_yolo(box_info["coords"], IMAGE_WIDTH, IMAGE_HEIGHT)
        annotation_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    label_path = LABELS_DIR / f"frame_{frame_number:04d}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(annotation_lines))
    
    return label_path


def capture_frame(output_path):
    """Capture a single frame using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-frames:v", "1",
        "-update", "1",
        "-q:v", "2",
        str(output_path),
        "-y"
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    return result.returncode == 0


def get_next_frame_number():
    """Find the next available frame number based on existing files."""
    existing = list(IMAGES_DIR.glob("frame_*.jpg"))
    if not existing:
        return 0
    
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 0


def main():
    # Parse max frames from command line
    max_frames = DEFAULT_MAX_FRAMES
    if len(sys.argv) > 1:
        try:
            max_frames = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_frames]")
            sys.exit(1)
    
    # Ensure directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get starting frame number
    frame_num = get_next_frame_number()
    captured = 0
    
    print("=" * 60)
    print("RTSP Camera Training Data Capture")
    print("=" * 60)
    print(f"Camera: {RTSP_URL}")
    print(f"Output: {IMAGES_DIR}")
    print(f"Annotations: {LABELS_DIR}")
    print(f"Capture interval: {CAPTURE_INTERVAL}s")
    print(f"Max frames: {max_frames}")
    print(f"Starting from frame: {frame_num:04d}")
    print("=" * 60)
    print("Press Ctrl+C to stop early\n")
    
    try:
        while captured < max_frames:
            frame_name = f"frame_{frame_num:04d}.jpg"
            image_path = IMAGES_DIR / frame_name
            
            # Capture frame
            if capture_frame(image_path):
                # Create annotation
                label_path = create_annotation(frame_num)
                
                captured += 1
                print(f"✓ Captured {frame_name} | Total: {captured}/{max_frames}")
                
                frame_num += 1
                
                # Wait before next capture (unless this was the last)
                if captured < max_frames:
                    time.sleep(CAPTURE_INTERVAL)
            else:
                print(f"✗ Failed to capture {frame_name}, retrying...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\nCapture stopped by user.")
    
    print("=" * 60)
    print(f"Capture complete! Total frames captured: {captured}")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Labels saved to: {LABELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
