#!/usr/bin/env python3.11
"""Test script for litter box detection integration."""
from pathlib import Path
import sys

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detection.litter_box_detector import LitterBoxDetector


def test_litter_box_detection():
    """Test the integrated YOLO litter box detection."""
    print("=" * 60)
    print("Testing YOLO Litter Box Detection Integration")
    print("=" * 60)
    
    # Initialize detector with YOLO model
    print("\n1. Initializing LitterBoxDetector with YOLO model...")
    detector = LitterBoxDetector(
        yolo_model_path="models/litter_box_detector.mlpackage",
        use_yolo=True,
        use_vision=True,
        use_contour_fallback=True,
    )
    
    # Get detector stats
    stats = detector.get_stats()
    print(f"\nDetector Stats:")
    print(f"  YOLO available: {stats['yolo_available']}")
    print(f"  Vision available: {stats['vision_available']}")
    
    if not stats['yolo_available']:
        print("\n❌ YOLO detector not available! Check model path and dependencies.")
        return False
    
    # Try to find a test image from the training dataset
    print("\n2. Looking for test images...")
    test_image_paths = [
        "training_data/train/images",
        "training_data/images",
        "data/litter_box_dataset/images/test",
        "data/litter_box_dataset/images/train",
        "datasets/litter_box/images/test",
        "datasets/litter_box/images/train",
    ]
    
    test_image = None
    for test_dir in test_image_paths:
        test_dir_path = Path(test_dir)
        if test_dir_path.exists():
            images = list(test_dir_path.glob("*.jpg")) + list(test_dir_path.glob("*.png"))
            if images:
                test_image = images[0]
                print(f"  Found test image: {test_image}")
                break
    
    if not test_image:
        # Try to grab a frame from the camera if no test images
        print("\n  No test images found in dataset. Creating a synthetic test frame...")
        # Create a simple test frame (blank with a box)
        test_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 50
        # Draw a litter box-like rectangle
        cv2.rectangle(test_frame, (700, 540), (1050, 980), (200, 200, 200), -1)
        cv2.rectangle(test_frame, (700, 540), (1050, 980), (100, 100, 100), 3)
    else:
        # Load the test image
        test_frame = cv2.imread(str(test_image))
        if test_frame is None:
            print(f"❌ Failed to load image: {test_image}")
            return False
        print(f"  Image loaded: {test_frame.shape}")
    
    # Run detection
    print("\n3. Running litter box detection...")
    region = detector.detect(test_frame)
    
    if region:
        print(f"\n✅ Litter box detected!")
        print(f"  Method: {region.method}")
        print(f"  Confidence: {region.confidence:.3f}")
        print(f"  BBox: {region.bbox}")
        print(f"  Area: {region.area} pixels")
        print(f"  Center: {region.center}")
        
        # Draw detection on frame
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(test_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label
        label = f"{region.method}: {region.confidence:.2f}"
        cv2.putText(
            test_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        # Save result
        output_path = Path("test_detection_result.jpg")
        cv2.imwrite(str(output_path), test_frame)
        print(f"\n  Detection visualization saved to: {output_path}")
        
        # Get final stats
        final_stats = detector.get_stats()
        print(f"\n  Avg latency: {final_stats['avg_latency_ms']:.1f}ms")
        print(f"  Tracking stability: {final_stats['stability']:.2f}")
        
        return True
    else:
        print(f"\n⚠️  No litter box detected")
        print(f"  This might be expected if using a synthetic test frame.")
        print(f"  Try with real camera footage for better results.")
        
        # Still save the frame for inspection
        output_path = Path("test_detection_nodetect.jpg")
        cv2.imwrite(str(output_path), test_frame)
        print(f"\n  Frame saved to: {output_path}")
        
        return False


if __name__ == "__main__":
    try:
        success = test_litter_box_detection()
        print("\n" + "=" * 60)
        if success:
            print("✅ Integration test PASSED!")
        else:
            print("⚠️  Integration test completed (no detection)")
        print("=" * 60)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
