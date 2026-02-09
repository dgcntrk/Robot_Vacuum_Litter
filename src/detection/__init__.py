"""Detection module exports."""
from src.detection.coreml_detector import (
    CoreMLDetector,
    Detection,
    VisionFrameworkDetector,
    create_detector,
)
from src.detection.litter_box_detector import LitterBoxDetector, LitterBoxRegion
from src.detection.zone_manager import Zone, ZoneManager

__all__ = [
    "CoreMLDetector",
    "VisionFrameworkDetector",
    "create_detector",
    "Detection",
    "LitterBoxDetector",
    "LitterBoxRegion",
    "Zone",
    "ZoneManager",
]
