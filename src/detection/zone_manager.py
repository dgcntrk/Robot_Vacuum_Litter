"""Zone management for litter box detection areas."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.detection.coreml_detector import Detection
from src.detection.litter_box_detector import LitterBoxDetector, LitterBoxRegion

if TYPE_CHECKING:
    import cv2

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A detection zone (litter box area)."""
    id: str
    name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    dynamic: bool = False  # Whether this zone is dynamically detected
    confidence: float = 1.0
    
    @property
    def area(self) -> int:
        """Calculate zone area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    @property
    def center(self) -> tuple[int, int]:
        """Calculate zone center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class ZoneManager:
    """Manages detection zones with support for dynamic litter box detection."""
    
    def __init__(
        self,
        zones: dict[str, tuple[int, int, int, int]] | None = None,
        dynamic_detection: bool = True,
        fallback_to_static: bool = False,
        litter_box_model_path: str | None = None,
    ):
        """Initialize with zone configurations.
        
        Args:
            zones: Dict mapping zone ID to bbox tuple (x1, y1, x2, y2)
            dynamic_detection: Enable automatic litter box detection
            fallback_to_static: Use static zones if dynamic detection fails
            litter_box_model_path: Path to YOLO litter box detection model
        """
        self._static_zones: dict[str, Zone] = {}
        self._dynamic_zones: dict[str, Zone] = {}
        self._dynamic_detection = dynamic_detection
        self._fallback_to_static = fallback_to_static and bool(zones)
        
        # Initialize litter box detector if dynamic detection enabled
        self._litter_detector: LitterBoxDetector | None = None
        if dynamic_detection:
            self._litter_detector = LitterBoxDetector(
                yolo_model_path=litter_box_model_path,
            )
            logger.info("Dynamic litter box detection enabled (YOLO + Vision fallbacks)")
        
        # Add static zones if provided
        if zones:
            for zone_id, bbox in zones.items():
                self.add_zone(zone_id, zone_id.replace("_", " ").title(), bbox, dynamic=False)
            
            if not dynamic_detection:
                logger.info(f"Using {len(zones)} static zone(s)")
    
    def add_zone(
        self,
        zone_id: str,
        name: str,
        bbox: tuple[int, int, int, int],
        dynamic: bool = False,
        confidence: float = 1.0,
    ):
        """Add a detection zone."""
        zone = Zone(
            id=zone_id,
            name=name,
            bbox=bbox,
            dynamic=dynamic,
            confidence=confidence,
        )
        
        if dynamic:
            self._dynamic_zones[zone_id] = zone
        else:
            self._static_zones[zone_id] = zone
        
        logger.info(f"Added {'dynamic' if dynamic else 'static'} zone '{name}' at {bbox}")
    
    def remove_zone(self, zone_id: str):
        """Remove a detection zone."""
        if zone_id in self._static_zones:
            del self._static_zones[zone_id]
        elif zone_id in self._dynamic_zones:
            del self._dynamic_zones[zone_id]
    
    def update_dynamic_zone(self, frame: np.ndarray, cat_detections: list[Detection]) -> bool:
        """Update dynamic litter box zones based on current frame.
        
        Args:
            frame: Current video frame
            cat_detections: Current cat detections to exclude from litter box search
        
        Returns:
            True if any valid zones were detected/updated
        """
        if not self._litter_detector:
            return False
        
        # Exclude regions where cats are detected
        excluded = [d.bbox for d in cat_detections]
        
        # Detect all litter boxes
        regions = self._litter_detector.detect_all(frame, excluded_regions=excluded)
        
        if regions:
            for region in regions:
                if not region.class_name:
                    continue  # Skip detections without a class name to avoid zone duplication
                zone_id = f"litter_box_{region.class_name}"
                zone_name = region.class_name.replace("_", " ").title()
                
                if zone_id in self._dynamic_zones:
                    self._dynamic_zones[zone_id].bbox = region.bbox
                    self._dynamic_zones[zone_id].confidence = region.confidence
                else:
                    self._dynamic_zones[zone_id] = Zone(
                        id=zone_id,
                        name=zone_name,
                        bbox=region.bbox,
                        dynamic=True,
                        confidence=region.confidence,
                    )
            return True
        
        return False
    
    def get_zones(self) -> list[Zone]:
        """Get all active zones (static + dynamic)."""
        zones = list(self._static_zones.values())
        zones.extend(self._dynamic_zones.values())
        
        if not zones and self._fallback_to_static and self._static_zones:
            zones = list(self._static_zones.values())
        
        return zones
    
    def get_zone(self, zone_id: str) -> Zone | None:
        """Get a specific zone by ID."""
        if zone_id in self._static_zones:
            return self._static_zones[zone_id]
        if zone_id in self._dynamic_zones:
            return self._dynamic_zones[zone_id]
        return None
    
    def get_primary_zone(self) -> Zone | None:
        """Get the primary zone for litter box detection.
        
        Returns first dynamic zone if available, otherwise first static zone.
        """
        if self._dynamic_zones:
            return next(iter(self._dynamic_zones.values()))
        if self._static_zones:
            return next(iter(self._static_zones.values()))
        return None
    
    def match_detections_to_zones(
        self,
        detections: list[Detection],
        min_overlap: float = 0.1,
    ) -> dict[str, list[Detection]]:
        """Match detections to zones based on bounding box overlap.
        
        Args:
            detections: List of detection results
            min_overlap: Minimum IoU to consider a match
        
        Returns:
            Dict mapping zone ID to list of detections in that zone
        """
        zones = self.get_zones()
        matches: dict[str, list[Detection]] = {z.id: [] for z in zones}
        
        for det in detections:
            for zone in zones:
                overlap = self._compute_iou(det.bbox, zone.bbox)
                if overlap >= min_overlap:
                    matches[zone.id].append(det)
        
        return matches
    
    def get_best_matching_zone(self, detection: Detection) -> tuple[str | None, float]:
        """Find the zone with highest overlap for a detection.
        
        Returns:
            Tuple of (zone_id, overlap_iou) or (None, 0) if no match
        """
        best_zone = None
        best_overlap = 0.0
        
        for zone in self.get_zones():
            overlap = self._compute_iou(detection.bbox, zone.bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_zone = zone.id
        
        return best_zone, best_overlap
    
    def is_detection_in_zone(self, detection: Detection, min_overlap: float = 0.3) -> bool:
        """Check if a detection overlaps with any zone."""
        zone = self.get_primary_zone()
        if not zone:
            return False
        
        overlap = self._compute_iou(detection.bbox, zone.bbox)
        return overlap >= min_overlap
    
    def reset_dynamic_detection(self):
        """Reset dynamic litter box detection (e.g., after camera movement)."""
        if self._litter_detector:
            self._litter_detector.reset_tracking()
        self._dynamic_zones.clear()
        logger.info("Dynamic zone detection reset")
    
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
    
    def draw_zones(
        self,
        frame: np.ndarray,
        static_color: tuple[int, int, int] = (255, 128, 0),  # Orange
        dynamic_color: tuple[int, int, int] = (0, 255, 128),  # Green
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Draw zone boundaries on frame for visualization."""
        import cv2
        
        for zone in self.get_zones():
            x1, y1, x2, y2 = zone.bbox
            color = dynamic_color if zone.dynamic else static_color
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label = f"{zone.name}"
            if zone.dynamic:
                label += f" ({zone.confidence:.2f})"
            
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 8),
                (x1 + text_w + 8, y1),
                color,
                -1,
            )
            
            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )
        
        return frame
    
    def get_stats(self) -> dict:
        """Get zone manager statistics."""
        stats = {
            "static_zones": len(self._static_zones),
            "dynamic_zones": len(self._dynamic_zones),
            "has_dynamic_zone": len(self._dynamic_zones) > 0,
            "dynamic_detection_enabled": self._litter_detector is not None,
        }
        
        if self._litter_detector:
            stats["litter_detector"] = self._litter_detector.get_stats()
        
        for zone_id, zone in self._dynamic_zones.items():
            stats[f"dynamic_{zone_id}_confidence"] = zone.confidence
        
        return stats
