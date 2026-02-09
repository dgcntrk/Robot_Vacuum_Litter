#!/usr/bin/env python3
"""
Calibration tool for annotating litter box zones on a camera frame.

Usage:
    python calibrate_zones.py

Controls:
    - Click 2 points to define Box 1 (green)
    - Click 2 more points to define Box 2 (blue)
    - Press 'r' to reset
    - Press 'q' to quit
"""

import cv2
import os

# Configuration
IMAGE_PATH = "/path/to/cat-litter-monitor/frame_snapshot.jpg"
OUTPUT_PATH = "/path/to/cat-litter-monitor/frame_calibrated.jpg"
MAX_DISPLAY_WIDTH = 1440  # Reasonable max width for display
MAX_DISPLAY_HEIGHT = 900  # Reasonable max height for display

# Global state
points = []  # List of (orig_x, orig_y) tuples in original image coordinates
scaled_points = []  # List of (display_x, display_y) tuples for drawing
scale_factor = 1.0
original_image = None
display_image = None
annotated_image = None


def compute_scale_factor(img_width, img_height):
    """Compute scale factor to fit image within display limits while maintaining aspect ratio."""
    scale_w = MAX_DISPLAY_WIDTH / img_width
    scale_h = MAX_DISPLAY_HEIGHT / img_height
    return min(scale_w, scale_h, 1.0)  # Never upscale beyond original size


def to_original_coords(display_x, display_y):
    """Convert display coordinates back to original image coordinates."""
    return int(display_x / scale_factor), int(display_y / scale_factor)


def to_display_coords(orig_x, orig_y):
    """Convert original coordinates to display coordinates."""
    return int(orig_x * scale_factor), int(orig_y * scale_factor)


def draw_annotation():
    """Draw the current annotation state on the display image."""
    global annotated_image
    
    # Start from a copy of the scaled original
    annotated_image = display_image.copy()
    
    # Draw Box 1 if we have points for it
    if len(points) >= 2:
        p1 = scaled_points[0]
        p2 = scaled_points[1]
        cv2.rectangle(annotated_image, p1, p2, (0, 255, 0), 2)
        # Label at top-left corner
        label_pos = (min(p1[0], p2[0]), min(p1[1], p2[1]) - 10)
        cv2.putText(annotated_image, "Box 1", label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw Box 2 if we have points for it
    if len(points) >= 4:
        p1 = scaled_points[2]
        p2 = scaled_points[3]
        cv2.rectangle(annotated_image, p1, p2, (255, 0, 0), 2)
        # Label at top-left corner
        label_pos = (min(p1[0], p2[0]), min(p1[1], p2[1]) - 10)
        cv2.putText(annotated_image, "Box 2", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw temporary point markers
    for i, sp in enumerate(scaled_points):
        color = (0, 255, 0) if i < 2 else (255, 0, 0)
        cv2.circle(annotated_image, sp, 5, color, -1)
    
    # Draw instructions
    if len(points) == 0:
        instruction = "Click top-left corner of Box 1"
    elif len(points) == 1:
        instruction = "Click bottom-right corner of Box 1"
    elif len(points) == 2:
        box1 = points_to_bbox(points[0], points[1])
        instruction = f"Box 1: {box1} | Click top-left of Box 2"
    elif len(points) == 3:
        instruction = "Click bottom-right corner of Box 2"
    else:
        instruction = "Press 'r' to reset, 'q' to quit"
    
    cv2.putText(annotated_image, instruction, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_image, instruction, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.imshow("Calibration", annotated_image)


def points_to_bbox(p1, p2):
    """Convert two points to a bbox [x1, y1, x2, y2] with proper ordering."""
    x1, y1 = p1
    x2, y2 = p2
    # Ensure x1 < x2 and y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [x1, y1, x2, y2]


def mouse_callback(event, x, y, flags, param):
    """Handle mouse click events."""
    global points, scaled_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 4:
            return  # Already have both boxes defined
        
        # Convert display coordinates to original coordinates
        orig_x, orig_y = to_original_coords(x, y)
        points.append((orig_x, orig_y))
        scaled_points.append((x, y))
        
        # Print progress
        if len(points) == 1:
            print(f"Box 1 - Point 1 (top-left): [{orig_x}, {orig_y}]")
        elif len(points) == 2:
            box1 = points_to_bbox(points[0], points[1])
            print(f"Box 1 - Point 2 (bottom-right): [{orig_x}, {orig_y}]")
            print(f"Box 1 complete: bbox = {box1}")
        elif len(points) == 3:
            print(f"Box 2 - Point 1 (top-left): [{orig_x}, {orig_y}]")
        elif len(points) == 4:
            box2 = points_to_bbox(points[2], points[3])
            print(f"Box 2 - Point 2 (bottom-right): [{orig_x}, {orig_y}]")
            print(f"Box 2 complete: bbox = {box2}")
            print_final_results()
        
        draw_annotation()


def print_final_results():
    """Print final results and save annotated image."""
    if len(points) < 4:
        print("Need 4 points to generate results")
        return
    
    box1 = points_to_bbox(points[0], points[1])
    box2 = points_to_bbox(points[2], points[3])
    
    print("\n" + "=" * 50)
    print("CALIBRATION COMPLETE")
    print("=" * 50)
    print(f"\nBox 1 (Main Litter Box): {box1}")
    print(f"Box 2 (Secondary Box):   {box2}")
    
    print("\nYAML snippet for config:")
    print("-" * 30)
    print(f"""zones:
  litter_box_main:
    name: "Main Litter Box"
    bbox: {box1}
  litter_box_secondary:
    name: "Secondary Box"
    bbox: {box2}""")
    print("-" * 30)
    
    # Save annotated image using original resolution
    output = original_image.copy()
    
    # Draw Box 1 on original
    p1 = points[0]
    p2 = points[1]
    cv2.rectangle(output, p1, p2, (0, 255, 0), 3)
    label_pos = (min(p1[0], p2[0]), min(p1[1], p2[1]) - 15)
    cv2.putText(output, "Box 1", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Draw Box 2 on original
    p1 = points[2]
    p2 = points[3]
    cv2.rectangle(output, p1, p2, (255, 0, 0), 3)
    label_pos = (min(p1[0], p2[0]), min(p1[1], p2[1]) - 15)
    cv2.putText(output, "Box 2", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    
    cv2.imwrite(OUTPUT_PATH, output)
    print(f"\nAnnotated image saved to: {OUTPUT_PATH}")


def reset():
    """Reset the calibration state."""
    global points, scaled_points
    points = []
    scaled_points = []
    print("\n--- Reset ---")
    draw_annotation()


def main():
    global original_image, display_image, scale_factor
    
    # Load the image
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return 1
    
    orig_h, orig_w = original_image.shape[:2]
    print(f"Loaded image: {orig_w}x{orig_h}")
    
    # Compute scale factor
    scale_factor = compute_scale_factor(orig_w, orig_h)
    
    if scale_factor < 1.0:
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        display_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Display size: {new_w}x{new_h} (scale: {scale_factor:.3f})")
    else:
        display_image = original_image.copy()
        print("Display at original size (no scaling needed)")
    
    # Create window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    # Initial draw
    draw_annotation()
    
    print("\nInstructions:")
    print("  1. Click top-left corner of Box 1")
    print("  2. Click bottom-right corner of Box 1")
    print("  3. Click top-left corner of Box 2")
    print("  4. Click bottom-right corner of Box 2")
    print("  Press 'r' to reset, 'q' to quit\n")
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            reset()
    
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    exit(main())
