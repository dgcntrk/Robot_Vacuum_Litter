"""Dynamic litter box detection using Apple Vision and computer vision techniques."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class LitterBoxRegion:
    """Detected litter box region."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    method: str  # Detection method used
    class_name: str = ""  # Detection class name
    timestamp: float = field(default_factory=time.time)
    
    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class TrackedRegion:
    """A region being tracked over time."""
    region: LitterBoxRegion
    history: deque[LitterBoxRegion] = field(default_factory=lambda: deque(maxlen=10))
    first_seen: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.history.append(self.region)
    
    def update(self, region: LitterBoxRegion):
        """Update with new detection."""
        self.region = region
        self.history.append(region)
    
    @property
    def stability_score(self) -> float:
        """Calculate stability based on position consistency."""
        if len(self.history) < 3:
            return 0.5
        
        # Calculate variance in center positions
        centers = [r.center for r in self.history]
        x_vals = [c[0] for c in centers]
        y_vals = [c[1] for c in centers]
        
        x_var = np.var(x_vals) if len(x_vals) > 1 else 0
        y_var = np.var(y_vals) if len(y_vals) > 1 else 0
        
        # Lower variance = higher stability
        max_acceptable_var = 1000  # pixels squared
        stability = 1.0 - min(1.0, (x_var + y_var) / (2 * max_acceptable_var))
        return stability
    
    @property
    def smoothed_bbox(self) -> tuple[int, int, int, int]:
        """Get temporally smoothed bounding box."""
        if len(self.history) < 2:
            return self.region.bbox
        
        # Exponential moving average
        alpha = 0.7
        boxes = [r.bbox for r in self.history]
        
        smoothed = list(boxes[-1])  # Start with most recent
        for bbox in reversed(boxes[:-1]):
            for i in range(4):
                smoothed[i] = int(alpha * smoothed[i] + (1 - alpha) * bbox[i])
        
        return tuple(smoothed)


class LitterBoxDetector:
    """Dynamic litter box detection using multiple CV techniques.
    
    Uses YOLOv8 model trained on litter box images as primary method,
    with Apple Vision framework and edge detection as fallbacks.
    """
    
    def __init__(
        self,
        min_area_ratio: float = 0.02,      # Min 2% of frame area (for smaller litter boxes)
        max_area_ratio: float = 0.5,       # Max 50% of frame area
        aspect_ratio_range: tuple[float, float] = (0.4, 3.0),  # More flexible aspect ratio
        stability_threshold: float = 0.3,   # Min stability to use detection
        use_yolo: bool = True,
        yolo_model_path: str | None = None,
        use_vision: bool = True,
        use_contour_fallback: bool = True,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.stability_threshold = stability_threshold
        self.use_yolo = use_yolo
        self.use_vision = use_vision
        self.use_contour_fallback = use_contour_fallback
        
        # Tracking state (multiple regions)
        self._tracked_regions: dict[str, TrackedRegion] = {}
        self._background_model: np.ndarray | None = None
        self._background_alpha = 0.01  # Background learning rate
        
        # Performance tracking
        self._inference_times: list[float] = []
        
        # YOLO detector for litter box
        self._yolo_detector = None
        self._yolo_available = False
        if use_yolo:
            self._yolo_available = self._setup_yolo_detector(yolo_model_path)
        
        # Check Vision availability
        self._vision_available = self._check_vision_availability()
    
    def _setup_yolo_detector(self, model_path: str | None = None) -> bool:
        """Setup YOLO detector for litter box detection."""
        try:
            from pathlib import Path
            from src.detection.coreml_detector import CoreMLDetector
            
            # Default to the trained litter box model
            if model_path is None:
                model_path = "models/litter_box_detector.mlpackage"
            
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                logger.warning(f"YOLO litter box model not found at {model_path}")
                return False
            
            # Create detector with litter box target classes (matches dataset.yaml)
            self._yolo_detector = CoreMLDetector(
                model_path=str(model_path),
                confidence_threshold=0.3,  # Lower threshold for litter box detection
                target_classes=["litter_box_main", "litter_box_secondary"],
            )
            
            logger.info(f"YOLO litter box detector loaded: {model_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to setup YOLO detector: {e}")
            return False
    
    def _check_vision_availability(self) -> bool:
        """Check if Apple Vision framework is available."""
        if not self.use_vision:
            return False
        try:
            import Vision
            return True
        except ImportError:
            logger.warning("Vision framework not available (requires macOS)")
            return False
    
    def detect(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> LitterBoxRegion | None:
        """Detect litter box in frame (returns first/best). Use detect_all for multiple."""
        results = self.detect_all(frame, excluded_regions)
        return results[0] if results else None

    def detect_all(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> list[LitterBoxRegion]:
        """Detect all litter boxes in frame.
        
        Args:
            frame: Input BGR frame
            excluded_regions: Regions to exclude (e.g., where cats are detected)
        
        Returns:
            List of LitterBoxRegion detections
        """
        start_time = time.perf_counter()
        
        regions = []
        
        # Try YOLO first (trained specifically for litter boxes, most accurate)
        if self._yolo_available and self._yolo_detector:
            regions = self._detect_all_with_yolo(frame, excluded_regions)
        
        # Fallback to other methods if YOLO found nothing
        if not regions:
            region = None
            if self._vision_available:
                region = self._detect_with_vision(frame, excluded_regions)
            if region is None:
                region = self._detect_with_edges(frame, excluded_regions)
            if region is None and self.use_contour_fallback:
                region = self._detect_with_contours(frame, excluded_regions)
            if region:
                regions = [region]
        
        # Update tracking for each region
        for region in regions:
            key = region.method + "_" + (region.class_name if hasattr(region, 'class_name') else "unknown")
            # Find matching tracked region by IoU
            matched = False
            for track_key, tracked in self._tracked_regions.items():
                iou = self._compute_iou(region.bbox, tracked.region.bbox)
                if iou > 0.3:
                    tracked.update(region)
                    matched = True
                    break
            if not matched:
                self._tracked_regions[key + f"_{len(self._tracked_regions)}"] = TrackedRegion(region)
        
        # Track performance
        inference_time = time.perf_counter() - start_time
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        
        # Return stabilized results
        stabilized = []
        for region in regions:
            # Find best matching tracked region
            best_tracked = None
            best_iou = 0.0
            for tracked in self._tracked_regions.values():
                iou = self._compute_iou(region.bbox, tracked.region.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tracked = tracked
            
            if best_tracked and best_tracked.stability_score >= self.stability_threshold:
                smoothed = best_tracked.smoothed_bbox
                stabilized.append(LitterBoxRegion(
                    bbox=smoothed,
                    confidence=best_tracked.region.confidence * best_tracked.stability_score,
                    method=best_tracked.region.method,
                    class_name=region.class_name,
                ))
            else:
                stabilized.append(region)
        
        return stabilized
    
    def _detect_with_yolo(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> LitterBoxRegion | None:
        """Detect best litter box using trained YOLO model."""
        results = self._detect_all_with_yolo(frame, excluded_regions)
        return results[0] if results else None

    def _detect_all_with_yolo(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> list[LitterBoxRegion]:
        """Detect ALL litter boxes using trained YOLO model."""
        try:
            h, w = frame.shape[:2]
            frame_area = h * w
            
            detections = self._yolo_detector.detect(frame)
            logger.debug(f"YOLO detections: {len(detections)} found")
            
            if not detections:
                return []
            
            regions = []
            
            for det in detections:
                logger.debug(f"YOLO detection: bbox={det.bbox}, conf={det.confidence:.3f}, class={det.class_name}")
                
                area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
                area_ratio = area / frame_area
                
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                
                bbox_w = det.bbox[2] - det.bbox[0]
                bbox_h = det.bbox[3] - det.bbox[1]
                aspect = bbox_w / bbox_h if bbox_h > 0 else 1.0
                
                if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
                    continue
                
                if excluded_regions:
                    excluded = False
                    for excl in excluded_regions:
                        if self._compute_iou(det.bbox, excl) > 0.5:
                            excluded = True
                            break
                    if excluded:
                        continue
                
                regions.append(LitterBoxRegion(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    method="yolo",
                    class_name=det.class_name,
                ))
            
            # Sort by confidence descending
            regions.sort(key=lambda r: r.confidence, reverse=True)
            return regions
            
        except Exception as e:
            logger.debug(f"YOLO detection failed: {e}")
            return []
    
    def _detect_with_vision(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> LitterBoxRegion | None:
        """Detect using Apple Vision rectangle detection."""
        try:
            import Vision
            from Cocoa import NSImage
            from PIL import Image
            
            h, w = frame.shape[:2]
            frame_area = h * w
            
            # Convert to PIL then to NSImage
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Create rectangle detection request
            request = Vision.VNDetectRectanglesRequest.alloc().init()
            request.setMinimumAspectRatio_(self.aspect_ratio_range[0])
            request.setMaximumAspectRatio_(self.aspect_ratio_range[1])
            request.setMinimumSize_(0.1)  # At least 10% of image dimension
            request.setQuadratureTolerance_(30.0)  # Allow some perspective distortion
            
            # Create image handler
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                pil_image.tobytes(),
                None
            )
            
            # Perform detection
            success, error = handler.performRequests_error_([request], None)
            
            if not success or error:
                return None
            
            results = request.results()
            if not results:
                return None
            
            # Filter and score rectangles
            best_region = None
            best_score = 0.0
            
            for observation in results:
                # Get bounding box (normalized coordinates)
                bbox = observation.boundingBox()
                
                # Convert to pixel coordinates (Vision uses bottom-left origin)
                x1 = int(bbox.origin.x * w)
                y1 = int((1.0 - bbox.origin.y - bbox.size.height) * h)
                x2 = int((bbox.origin.x + bbox.size.width) * w)
                y2 = int((1.0 - bbox.origin.y) * h)
                
                pixel_bbox = (x1, y1, x2, y2)
                
                # Filter by area
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / frame_area
                
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                
                # Check excluded regions
                if excluded_regions:
                    excluded = False
                    for excl in excluded_regions:
                        if self._compute_iou(pixel_bbox, excl) > 0.5:
                            excluded = True
                            break
                    if excluded:
                        continue
                
                # Score based on confidence and area centrality
                confidence = float(observation.confidence()) if hasattr(observation, 'confidence') else 0.8
                
                # Prefer regions near center of frame (litter boxes usually centrally placed)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                frame_cx, frame_cy = w // 2, h // 2
                dist_from_center = np.sqrt((cx - frame_cx)**2 + (cy - frame_cy)**2)
                max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                centrality = 1.0 - (dist_from_center / max_dist)
                
                score = confidence * (0.5 + 0.5 * centrality)  # Weight centrality
                
                if score > best_score:
                    best_score = score
                    best_region = LitterBoxRegion(
                        bbox=pixel_bbox,
                        confidence=confidence,
                        method="vision_rectangle",
                    )
            
            return best_region
            
        except Exception as e:
            logger.debug(f"Vision detection failed: {e}")
            return None
    
    def _detect_with_contours(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> LitterBoxRegion | None:
        """Detect using background subtraction and contour analysis."""
        try:
            h, w = frame.shape[:2]
            frame_area = h * w
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Update background model (running average)
            if self._background_model is None:
                self._background_model = gray.astype(np.float32)
                return None
            
            cv2.accumulateWeighted(gray, self._background_model, self._background_alpha)
            
            # Compute difference from background
            diff = cv2.absdiff(gray, self._background_model.astype(np.uint8))
            
            # Threshold to get foreground mask
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Invert mask - we're looking for static objects (background)
            static_mask = cv2.bitwise_not(mask)
            
            # Find contours in static regions
            contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_region = None
            best_score = 0.0
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                area_ratio = area / frame_area
                
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                
                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(contour)
                bbox = (x, y, x + bw, y + bh)
                
                # Check aspect ratio
                aspect = bw / bh if bh > 0 else 1.0
                if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
                    continue
                
                # Check excluded regions
                if excluded_regions:
                    excluded = False
                    for excl in excluded_regions:
                        if self._compute_iou(bbox, excl) > 0.5:
                            excluded = True
                            break
                    if excluded:
                        continue
                
                # Score based on rectangularity and centrality
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    rectangularity = 4 * np.pi * area / (perimeter ** 2)  # Circle = 1, square ~ 0.785
                else:
                    rectangularity = 0
                
                # Prefer regions near center
                cx, cy = x + bw // 2, y + bh // 2
                frame_cx, frame_cy = w // 2, h // 2
                dist_from_center = np.sqrt((cx - frame_cx)**2 + (cy - frame_cy)**2)
                max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                centrality = 1.0 - (dist_from_center / max_dist)
                
                score = rectangularity * centrality
                
                if score > best_score:
                    best_score = score
                    best_region = LitterBoxRegion(
                        bbox=bbox,
                        confidence=min(1.0, score * 2),  # Scale up for confidence
                        method="contour_static",
                    )
            
            return best_region
            
        except Exception as e:
            logger.debug(f"Contour detection failed: {e}")
            return None
    
    def _detect_with_edges(
        self,
        frame: np.ndarray,
        excluded_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> LitterBoxRegion | None:
        """Detect using edge detection and shape analysis.
        
        This works better for finding rectangular objects in static scenes
        where background subtraction doesn't help.
        """
        try:
            h, w = frame.shape[:2]
            frame_area = h * w
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate edges to connect gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_region = None
            best_score = 0.0
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                area_ratio = area / frame_area
                
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                
                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(contour)
                bbox = (x, y, x + bw, y + bh)
                
                # Check aspect ratio
                aspect = bw / bh if bh > 0 else 1.0
                if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
                    continue
                
                # Check excluded regions
                if excluded_regions:
                    excluded = False
                    for excl in excluded_regions:
                        if self._compute_iou(bbox, excl) > 0.5:
                            excluded = True
                            break
                    if excluded:
                        continue
                
                # Approximate polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Score based on:
                # - Having 4-8 sides (box-like, allowing for curves)
                # - Fill ratio (solid vs hollow)
                # - Centrality
                
                n_sides = len(approx)
                box_score = 1.0 if 4 <= n_sides <= 8 else 0.5 if n_sides > 8 else 0.3
                
                # Fill ratio
                bbox_area = bw * bh
                fill_ratio = area / bbox_area if bbox_area > 0 else 0
                
                # Prefer solid-ish regions
                fill_score = fill_ratio if fill_ratio > 0.3 else 0.3
                
                # Centrality
                cx, cy = x + bw // 2, y + bh // 2
                frame_cx, frame_cy = w // 2, h // 2
                dist_from_center = np.sqrt((cx - frame_cx)**2 + (cy - frame_cy)**2)
                max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                centrality = 1.0 - (dist_from_center / max_dist)
                
                # Combined score
                score = box_score * fill_score * centrality
                
                if score > best_score:
                    best_score = score
                    best_region = LitterBoxRegion(
                        bbox=bbox,
                        confidence=min(1.0, score),
                        method="edge_detection",
                    )
            
            return best_region
            
        except Exception as e:
            logger.debug(f"Edge detection failed: {e}")
            return None
    
    def reset_tracking(self):
        """Reset tracked regions (e.g., after camera movement)."""
        self._tracked_regions.clear()
        self._background_model = None
        logger.info("Litter box tracking reset")
    
    def get_tracked_region(self) -> LitterBoxRegion | None:
        """Get the first stable tracked region."""
        for tracked in self._tracked_regions.values():
            if tracked.stability_score >= self.stability_threshold:
                return LitterBoxRegion(
                    bbox=tracked.smoothed_bbox,
                    confidence=tracked.region.confidence,
                    method=tracked.region.method,
                )
        return None
    
    @staticmethod
    def _compute_iou(
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
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
        """Get average detection latency in milliseconds."""
        if not self._inference_times:
            return 0.0
        return (sum(self._inference_times) / len(self._inference_times)) * 1000
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "yolo_available": self._yolo_available,
            "vision_available": self._vision_available,
            "tracked": len(self._tracked_regions) > 0,
            "tracked_count": len(self._tracked_regions),
            "avg_latency_ms": self.get_avg_latency_ms(),
        }
