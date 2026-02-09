#!/usr/bin/env python3
"""Interactive zone calibration tool.

Usage:
    python calibrate_zone.py --camera rtsp://YOUR_CAMERA_IP/live0
    python calibrate_zone.py --image /tmp/camera_frame.png
"""
import argparse
import os
import sys
sys.path.insert(0, '/path/to/cat-litter-monitor')

import cv2
import yaml

# Global state for mouse callback
_drawing = False
_start_point = None
_current_box = None
def _mouse_callback(event, x, y, flags, param):
    global _drawing, _start_point, _current_box
    
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _start_point = (x, y)
        _current_box = None
    
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        if _start_point:
            _current_box = (_start_point[0], _start_point[1], x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        if _start_point:
            _current_box = (_start_point[0], _start_point[1], x, y)

def calibrate_from_camera(rtsp_url: str):
    """Open camera and let user draw zone."""
    print(f"Connecting to {rtsp_url}...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Failed to open camera: {rtsp_url}")
        return None
    
    print("Camera connected. Press SPACE to freeze frame, then draw zone.")
    print("Press 'q' to quit without saving.")
    
    frozen_frame = None
    
    cv2.namedWindow("Calibrate Zone")
    cv2.setMouseCallback("Calibrate Zone", _mouse_callback)
    
    while True:
        if frozen_frame is None:
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, "Press SPACE to freeze", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display = frozen_frame.copy()
            
            # Draw current box
            if _current_box:
                x1, y1, x2, y2 = _current_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Zone: ({x1}, {y1}, {x2}, {y2})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display, "Drag to draw zone, ENTER to save", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibrate Zone", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if frozen_frame is None:
                frozen_frame = frame.copy()
                print("Frame frozen. Draw your zone.")
        
        elif key == 13:  # ENTER
            if _current_box:
                x1, y1, x2, y2 = _current_box
                # Normalize to (x1, y1, x2, y2) where x1<x2, y1<y2
                bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                cap.release()
                cv2.destroyAllWindows()
                return bbox
        
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

def calibrate_from_image(image_path: str):
    """Open image and let user draw zone."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    print("Drag to draw zone around litter box.")
    print("Press ENTER to save, 'q' to quit without saving.")
    
    cv2.namedWindow("Calibrate Zone")
    cv2.setMouseCallback("Calibrate Zone", _mouse_callback)
    
    while True:
        display = frame.copy()
        
        # Draw current box
        if _current_box:
            x1, y1, x2, y2 = _current_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Zone: ({x1}, {y1}, {x2}, {y2})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibrate Zone", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 and _current_box:  # ENTER
            x1, y1, x2, y2 = _current_box
            bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            cv2.destroyAllWindows()
            return bbox
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

def save_to_config(bbox: tuple, config_path: str = "config/settings.yaml"):
    """Save zone to config file."""
    x1, y1, x2, y2 = bbox
    
    zone_config = {
        "zones": {
            "litter_box_main": {
                "name": "Main Litter Box",
                "bbox": [x1, y1, x2, y2]
            }
        }
    }
    
    # Read existing config
    existing = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing = yaml.safe_load(f) or {}
    
    # Merge zones
    if 'zones' not in existing:
        existing['zones'] = {}
    existing['zones'].update(zone_config['zones'])
    
    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Saved zone to {config_path}")
    print(f"  bbox: [{x1}, {y1}, {x2}, {y2}]")
    print("\nAlso add this to your config to disable dynamic detection:")
    print("  detection:")
    print("    dynamic_zones: false")

def main():
    parser = argparse.ArgumentParser(description="Calibrate litter box detection zone")
    parser.add_argument("--camera", default="rtsp://YOUR_CAMERA_IP/live0",
                       help="RTSP camera URL")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--config", default="config/settings.yaml",
                       help="Config file path")
    
    args = parser.parse_args()
    
    if args.image:
        bbox = calibrate_from_image(args.image)
    else:
        bbox = calibrate_from_camera(args.camera)
    
    if bbox:
        save_to_config(bbox, args.config)
    else:
        print("Calibration cancelled.")

if __name__ == "__main__":
    main()
