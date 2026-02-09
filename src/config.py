"""Configuration management with Pydantic settings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default paths
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class CameraConfig(BaseModel):
    """RTSP camera configuration."""
    rtsp_url: str = "rtsp://YOUR_CAMERA_IP/live0"
    fps: int = 5
    resolution: tuple[int, int] = (640, 480)
    reconnect_interval: int = 5
    buffer_size: int = 1
    use_tcp: bool = True


class DetectionConfig(BaseModel):
    """Detection model configuration."""
    provider: Literal["coreml_yolo", "vision_framework", "manual"] = "coreml_yolo"
    model_path: str = "models/yolov8n.mlpackage"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    inference_interval: float = 0.2  # 200ms = 5 FPS detection
    target_classes: list[str] = Field(default_factory=lambda: ["cat"])
    dynamic_zones: bool = True  # Enable automatic litter box detection
    litter_box_model_path: Optional[str] = "models/litter_box_detector.mlpackage"  # Path to litter box detector model


class ZoneConfig(BaseModel):
    """A detection zone (litter box area)."""
    name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class TimingConfig(BaseModel):
    """State machine timing configuration."""
    min_occupancy_seconds: float = 5.0
    cooldown_seconds: float = 60.0
    max_session_minutes: float = 10.0


class RobotConfig(BaseModel):
    """Robot vacuum integration configuration."""
    enabled: bool = False
    room_name: str = "Litter"
    dispatch_delay_seconds: float = 5.0
    emergency_stop_on_cat_detected: bool = True
    household_id: str = ""
    dsn: str = ""
    floor_id: str = ""


class EventConfig(BaseModel):
    """Event logging configuration."""
    log_dir: str = "./logs"
    max_history: int = 1000
    console_output: bool = True


class VisualizationConfig(BaseModel):
    """Debug visualization configuration."""
    enabled: bool = True
    window_name: str = "Cat Litter Monitor"
    show_zones: bool = True
    show_detections: bool = True
    font_scale: float = 0.5
    thickness: int = 2


class DashboardConfig(BaseModel):
    """Web dashboard configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080


class Settings(BaseSettings):
    """Application settings loaded from YAML with env var overrides."""
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    zones: dict[str, ZoneConfig] = Field(default_factory=dict)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to config."""
    # Camera overrides
    if url := os.environ.get("CAMERA_RTSP_URL"):
        config.setdefault("camera", {})["rtsp_url"] = url
    
    # Detection overrides
    if provider := os.environ.get("DETECTION_PROVIDER"):
        config.setdefault("detection", {})["provider"] = provider
    if model := os.environ.get("DETECTION_MODEL"):
        config.setdefault("detection", {})["model_path"] = model
    if dynamic := os.environ.get("DETECTION_DYNAMIC_ZONES"):
        config.setdefault("detection", {})["dynamic_zones"] = dynamic.lower() == "true"
    
    # Robot overrides
    if enabled := os.environ.get("ROBOT_ENABLED"):
        config.setdefault("robot", {})["enabled"] = enabled.lower() == "true"
    if room := os.environ.get("ROBOT_ROOM_NAME"):
        config.setdefault("robot", {})["room_name"] = room
    
    return config


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML with env var overrides."""
    config_dict = load_yaml_config(path)
    config_dict = apply_env_overrides(config_dict)
    return Settings(**config_dict)


# Global settings instance
settings = load_settings()
