"""CoreML-based object detection optimized for Apple Silicon."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection result."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class CoreMLDetector:
    """YOLO-based object detection using CoreML on Apple Neural Engine.
    
    Expects a CoreML model exported from YOLOv8 with NMS enabled.
    The model should output:
    - coordinates: bounding boxes [x1, y1, x2, y2]
    - confidence: detection confidence scores
    - class labels: class indices
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes or ["cat"])
        
        self._model = None
        self._model_loaded = False
        self._class_names: dict[int, str] = {}
        self._input_size: tuple[int, int] = (640, 640)
        self._has_nms_inputs = False
        
        # Performance tracking
        self._inference_times: list[float] = []
        self._max_history = 100
        
    def _load_model(self) -> bool:
        """Lazy-load the CoreML model."""
        if self._model_loaded:
            return self._model is not None
        
        self._model_loaded = True
        
        try:
            import coremltools as ct
            from PIL import Image
            
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            logger.info(f"Loading CoreML model: {self.model_path}")
            
            # Load the model
            self._model = ct.models.MLModel(str(self.model_path))
            
            # Get model info
            spec = self._model.get_spec()
            
            # Try to extract class names from metadata
            if metadata := getattr(spec, "metadata", None):
                if user_defined := getattr(metadata, "userDefined", None):
                    if "classes" in user_defined:
                        classes = user_defined["classes"].split(",")
                        self._class_names = {i: name.strip() for i, name in enumerate(classes)}
            
            # Default class names based on target classes
            if not self._class_names:
                if self.target_classes and "cat" in self.target_classes:
                    # COCO classes for cat detection
                    self._class_names = {
                        15: "cat",
                        0: "person",
                        16: "dog",
                    }
                else:
                    # Custom model - map indices to target class names
                    self._class_names = {
                        i: name for i, name in enumerate(sorted(self.target_classes))
                    } if self.target_classes else {}
            
            # Get input size from model spec
            if spec.WhichOneof("Type") == "neuralNetworkClassifier":
                input_layer = spec.neuralNetworkClassifier.layers[0]
            elif spec.WhichOneof("Type") == "neuralNetwork":
                input_layer = spec.neuralNetwork.layers[0]
            elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
                input_layer = spec.neuralNetworkRegressor.layers[0]
            else:
                # Try to get from description
                input_desc = spec.description.input[0]
                if len(input_desc.type.multiArrayType.shape) >= 2:
                    h = input_desc.type.multiArrayType.shape[1]
                    w = input_desc.type.multiArrayType.shape[2]
                    self._input_size = (w, h)
            
            # Check for NMS threshold inputs
            input_names = {inp.name for inp in spec.description.input}
            self._has_nms_inputs = "iouThreshold" in input_names and "confidenceThreshold" in input_names
            if self._has_nms_inputs:
                logger.info("Model has built-in NMS (iouThreshold + confidenceThreshold)")
            
            logger.info(f"CoreML model loaded. Input size: {self._input_size}")
            logger.info(f"Target classes: {self.target_classes}")
            return True
            
        except ImportError:
            logger.error("coremltools not installed. Install with: pip install coremltools")
            return False
        except Exception as e:
            logger.error(f"Failed to load CoreML model: {e}")
            return False
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        # Resize to model input size
        if frame.shape[:2] != (self._input_size[1], self._input_size[0]):
            frame = cv2.resize(frame, self._input_size)
        
        # Convert BGR to RGB (CoreML expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame_rgb
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a frame.
        
        Returns list of Detection objects.
        """
        if not self._load_model():
            return []
        
        if self._model is None:
            return []
        
        try:
            from PIL import Image
            
            # Preprocess
            input_frame = self._preprocess(frame)
            
            # Convert to PIL Image for CoreML
            pil_image = Image.fromarray(input_frame)
            
            # Run inference — pass NMS thresholds if model accepts them
            start_time = time.perf_counter()
            predict_input = {"image": pil_image}
            if self._has_nms_inputs:
                predict_input["iouThreshold"] = self.iou_threshold
                predict_input["confidenceThreshold"] = self.confidence_threshold
            predictions = self._model.predict(predict_input)
            inference_time = time.perf_counter() - start_time
            
            # Track performance
            self._inference_times.append(inference_time)
            if len(self._inference_times) > self._max_history:
                self._inference_times.pop(0)
            
            # Parse predictions (format varies by model export)
            detections = self._parse_predictions(predictions, frame.shape[:2])
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _parse_predictions(
        self,
        predictions: dict,
        original_shape: tuple[int, int],
    ) -> list[Detection]:
        """Parse CoreML predictions into Detection objects."""
        import numpy as np
        
        detections = []

        # Handle different YOLOv8 CoreML output formats:
        
        # Format 1: Post-processed outputs with NMS
        # - "coordinates": [N, 4] bounding boxes [x1, y1, x2, y2]
        # - "confidence": [N, 80] class probabilities (80 COCO classes)
        if "coordinates" in predictions and "confidence" in predictions:
            boxes = np.array(predictions["coordinates"])
            confidences = np.array(predictions["confidence"])

            # Handle empty predictions
            if boxes.shape[0] == 0:
                return detections

            # Coordinates are normalized [cx, cy, w, h] (0-1 range)
            orig_h, orig_w = original_shape[:2]

            for i, box in enumerate(boxes):
                # confidence is [N, 80] - get max class prob and its index
                class_probs = confidences[i]
                conf = float(class_probs.max())
                label_idx = int(class_probs.argmax())

                if conf < self.confidence_threshold:
                    continue

                class_name = self._class_names.get(label_idx, f"class_{label_idx}")

                # Filter by target classes
                if self.target_classes and class_name not in self.target_classes:
                    continue

                # Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                cx, cy, w, h = box[0], box[1], box[2], box[3]
                x1 = int((cx - w / 2) * orig_w)
                y1 = int((cy - h / 2) * orig_h)
                x2 = int((cx + w / 2) * orig_w)
                y2 = int((cy + h / 2) * orig_h)
                
                # Clamp to frame
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=label_idx,
                    class_name=class_name,
                ))
        
        # Format 2: Raw YOLO output tensor [1, num_features, num_anchors]
        # For single class: [1, 5, 8400] = [x, y, w, h, class_conf]
        # For multiple classes: [1, 4+num_classes, 8400]
        else:
            # Find the output tensor (usually var_XXX)
            output_key = list(predictions.keys())[0]
            output = predictions[output_key]
            
            if isinstance(output, np.ndarray) and len(output.shape) == 3:
                # YOLOv8 format: [batch, features, anchors]
                batch, num_features, num_anchors = output.shape
                
                # Transpose to [anchors, features] for easier processing
                output = output[0].T  # Now [8400, num_features]
                
                # Extract boxes and scores
                # Format: [x_center, y_center, width, height, class_conf1, class_conf2, ...]
                boxes_xywh = output[:, :4]  # [8400, 4]
                class_scores = output[:, 4:]  # [8400, num_classes]
                
                # Convert from center format to corner format
                boxes = np.zeros_like(boxes_xywh)
                boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
                boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
                boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
                boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
                
                # Get max class score and index for each anchor
                max_scores = class_scores.max(axis=1)
                max_indices = class_scores.argmax(axis=1)
                
                # Filter by confidence threshold
                mask = max_scores > self.confidence_threshold
                boxes = boxes[mask]
                scores = max_scores[mask]
                indices = max_indices[mask]
                
                # Apply NMS (simple version - take top detections)
                # Scale boxes back to original frame size
                scale_x = original_shape[1] / self._input_size[0]
                scale_y = original_shape[0] / self._input_size[1]
                
                for box, score, idx in zip(boxes, scores, indices):
                    class_name = self._class_names.get(int(idx), "litter_box")
                    
                    # Filter by target classes
                    if self.target_classes and class_name not in self.target_classes:
                        continue
                    
                    # Scale box coordinates
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    
                    # Clamp to frame boundaries
                    x1 = max(0, min(x1, original_shape[1]))
                    y1 = max(0, min(y1, original_shape[0]))
                    x2 = max(0, min(x2, original_shape[1]))
                    y2 = max(0, min(y2, original_shape[0]))
                    
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score),
                        class_id=int(idx),
                        class_name=class_name,
                    ))
                
                # Simple NMS: sort by confidence and remove overlapping boxes
                if len(detections) > 0:
                    detections = self._apply_nms(detections, self.iou_threshold)

        return detections
    
    def _apply_nms(self, detections: list[Detection], iou_threshold: float) -> list[Detection]:
        """Apply Non-Maximum Suppression to filter overlapping detections."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep highest confidence detection
            best = detections.pop(0)
            keep.append(best)
            
            # Remove detections that overlap significantly with best
            detections = [
                det for det in detections
                if self._compute_iou(best.bbox, det.bbox) < iou_threshold
            ]
        
        return keep
    
    @staticmethod
    def _compute_iou(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        inter_area = inter_width * inter_height
        
        if inter_area == 0:
            return 0.0
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_avg_latency_ms(self) -> float:
        """Get average inference latency in milliseconds."""
        if not self._inference_times:
            return 0.0
        return (sum(self._inference_times) / len(self._inference_times)) * 1000
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "model_loaded": self._model is not None,
            "model_path": str(self.model_path),
            "input_size": self._input_size,
            "avg_latency_ms": self.get_avg_latency_ms(),
            "confidence_threshold": self.confidence_threshold,
        }


class VisionFrameworkDetector:
    """Apple Vision framework animal detector (built-in, no model needed).
    
    Uses VNRecognizeAnimalsRequest which can detect cats and dogs.
    Pros: Fast, no model download needed, optimized for Apple Silicon.
    Cons: No bounding boxes, only tells you IF a cat is present, not WHERE.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._available = False
        self._check_availability()
        
        self._inference_times: list[float] = []
    
    def _check_availability(self):
        """Check if Vision framework is available."""
        try:
            import Vision
            self._available = True
        except ImportError:
            logger.warning("Vision framework not available (requires macOS)")
            self._available = False
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect animals in frame using Vision framework.
        
        Returns a single detection if cat is present (no bbox info available).
        """
        if not self._available:
            return []
        
        try:
            import Vision
            import objc
            from Cocoa import NSImage
            from PIL import Image
            
            start_time = time.perf_counter()
            
            # Convert numpy array to NSImage
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Create Vision request
            request = Vision.VNRecognizeAnimalsRequest.alloc().init()
            
            # Create handler
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                pil_image.tobytes(),
                None
            )
            
            # Perform request
            success, error = handler.performRequests_error_([request], None)
            
            inference_time = time.perf_counter() - start_time
            self._inference_times.append(inference_time)
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)
            
            if not success or error:
                return []
            
            results = request.results()
            if not results:
                return []
            
            # Check for cat detection
            for observation in results:
                # VNRecognizedAnimalObservation has animalIdentifier and confidence
                animal_id = observation.animalIdentifier()
                confidence = observation.confidence()
                
                if animal_id == "cat" and confidence >= self.confidence_threshold:
                    # Return detection covering center of frame (no precise bbox available)
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, h // 2
                    bw, bh = w // 3, h // 3  # Approximate size
                    
                    return [Detection(
                        bbox=(cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2),
                        confidence=float(confidence),
                        class_id=15,
                        class_name="cat",
                    )]
            
            return []
            
        except Exception as e:
            logger.error(f"Vision detection error: {e}")
            return []
    
    def get_avg_latency_ms(self) -> float:
        """Get average inference latency in milliseconds."""
        if not self._inference_times:
            return 0.0
        return (sum(self._inference_times) / len(self._inference_times)) * 1000


def create_detector(
    provider: str,
    model_path: str | None = None,
    confidence_threshold: float = 0.5,
    target_classes: list[str] | None = None,
) -> CoreMLDetector | VisionFrameworkDetector:
    """Factory function to create appropriate detector."""
    if provider == "vision_framework":
        return VisionFrameworkDetector(confidence_threshold)
    
    # Default to CoreML YOLO
    if model_path is None:
        model_path = "models/yolov8n.mlpackage"
    
    return CoreMLDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        target_classes=target_classes or ["cat"],
    )
