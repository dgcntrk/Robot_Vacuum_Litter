#!/usr/bin/env python3
"""
Live YOLO Detection Demo for Cat Litter Monitor
Displays real-time object detection from RTSP camera feed.
"""

import cv2
import torch
from ultralytics import YOLO
import time

# Configuration
RTSP_URL = "rtsp://YOUR_CAMERA_IP/live0"
MODEL_PATH = "/path/to/cat-litter-monitor/models/litter_box_detector.pt"

# Colors for bounding boxes (BGR format for OpenCV)
COLORS = {
    "litter_box_main": (0, 255, 0),      # GREEN
    "litter_box_secondary": (255, 0, 0),  # BLUE
}

# Confidence threshold
CONF_THRESHOLD = 0.5


def main():
    print("=" * 50)
    print("Cat Litter Monitor - Live Detection Demo")
    print("=" * 50)
    
    # Load YOLO model
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Connect to RTSP camera
    print(f"Connecting to camera: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Error: Could not open camera stream")
        return
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Camera connected!")
    print("Press 'q' to quit")
    print("-" * 50)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    fps_time = time.time()
    
    # Frame skip for smoother performance (process every Nth frame)
    PROCESS_EVERY_N_FRAMES = 1  # Change to 2 if too slow
    frame_skip_counter = 0
    
    # Store last results for frames we skip
    last_results = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            time.sleep(0.1)
            continue
        
        frame_skip_counter += 1
        
        # Run inference every N frames
        if frame_skip_counter >= PROCESS_EVERY_N_FRAMES:
            frame_skip_counter = 0
            
            # Run YOLO inference
            results = model(frame, verbose=False)
            last_results = results
        else:
            results = last_results
        
        # Draw detections
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    if conf < CONF_THRESHOLD:
                        continue
                    
                    # Get class name
                    class_name = result.names[cls_id]
                    
                    # Get color for this class
                    color = COLORS.get(class_name, (128, 128, 128))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label = f"{class_name}: {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    cv2.rectangle(
                        frame, 
                        (x1, label_y - label_size[1] - 5), 
                        (x1 + label_size[0], label_y + 5), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - fps_time
        
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = current_time
        
        # Draw FPS counter
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 255), 
            2
        )
        
        # Show frame
        cv2.imshow("Cat Litter Monitor - Live Detection", frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")


if __name__ == "__main__":
    main()
