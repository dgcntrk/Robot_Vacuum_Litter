#!/usr/bin/env python3
"""Download and convert YOLOv8 to CoreML format.

Usage:
    python scripts/get_model.py
    
This will:
1. Download YOLOv8n (nano) from Ultralytics
2. Convert to CoreML format with NMS
3. Save to models/yolov8n.mlpackage
"""
import subprocess
import sys
from pathlib import Path


def main():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    output_path = models_dir / "yolov8n.mlpackage"
    
    if output_path.exists():
        print(f"Model already exists at {output_path}")
        response = input("Re-download? [y/N]: ")
        if response.lower() != 'y':
            print("Exiting")
            return 0
    
    print("Installing ultralytics...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    
    print("Downloading and converting YOLOv8n to CoreML...")
    print("(This may take a few minutes)")
    
    try:
        from ultralytics import YOLO
        
        # Download model
        model = YOLO("yolov8n.pt")
        
        # Export to CoreML with NMS
        # nms=True adds Non-Maximum Suppression to the model
        # imgsz=640 is the input resolution
        model.export(
            format="coreml",
            nms=True,
            imgsz=640,
            half=False,  # Use full precision for Mac
        )
        
        # Move to models directory
        exported_path = Path("yolov8n.mlpackage")
        if exported_path.exists():
            import shutil
            shutil.move(str(exported_path), str(output_path))
            print(f"\n✓ Model saved to {output_path}")
        else:
            print("\n✗ Export failed - model not found")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
