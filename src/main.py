"""Cat Litter Monitor - Main entry point.

Near-realtime cat detection for litter box monitoring with Apple Silicon optimization.
"""
from __future__ import annotations

import os
# Set FFmpeg RTSP transport option BEFORE importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;5000000|probesize;5000000|buffer_size;65536|stimeout;60000000"

import asyncio
import logging
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table

from src.camera import RTSPClient
from src.config import settings
from src.dashboard import DashboardServer
from src.detection import create_detector, ZoneManager
from src.events import EventLogger
from src.robot import create_robot_controller, RobotAdapter
from src.state import MultiZoneStateMachine, Session, StateEvent

if TYPE_CHECKING:
    from src.detection.coreml_detector import Detection

# Setup logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("cat_litter_monitor")
console = Console()


class CatLitterMonitor:
    """Main application class."""
    
    def __init__(self):
        self.running = False
        self._shutdown_event = threading.Event()
        
        # Components
        self.camera: RTSPClient | None = None
        self.detector = None
        self.zone_manager: ZoneManager | None = None
        self.state_machine: MultiZoneStateMachine | None = None
        self.event_logger = EventLogger(
            log_dir=settings.events.log_dir,
            max_history=settings.events.max_history,
        )
        self.robot_adapter: RobotAdapter | None = None
        self.dashboard: DashboardServer | None = None
        
        # Stats
        self._detection_count = 0
        self._last_inference_time = 0.0
        self._avg_latency = 0.0
        self._last_zone_update = 0.0
        self._zone_update_interval = 0.5  # Update litter box position every 500ms
        
    def _setup_camera(self) -> RTSPClient:
        """Initialize RTSP camera client."""
        return RTSPClient(
            rtsp_url=settings.camera.rtsp_url,
            fps=settings.camera.fps,
            resolution=settings.camera.resolution,
            reconnect_interval=settings.camera.reconnect_interval,
            use_tcp=settings.camera.use_tcp,
        )
    
    def _setup_detector(self):
        """Initialize detection model."""
        return create_detector(
            provider=settings.detection.provider,
            model_path=settings.detection.model_path,
            confidence_threshold=settings.detection.confidence_threshold,
            target_classes=settings.detection.target_classes,
        )
    
    def _setup_zones(self) -> ZoneManager:
        """Initialize zone manager with dynamic detection."""
        # Check if we should use dynamic detection
        dynamic_enabled = getattr(settings.detection, 'dynamic_zones', True)
        
        # Get litter box model path if configured
        litter_box_model = getattr(settings.detection, 'litter_box_model_path', None)
        
        # Load static zones from config as fallback
        zones_config = {}
        if hasattr(settings, 'zones') and settings.zones:
            zones_config = {
                zone_id: zone.bbox
                for zone_id, zone in settings.zones.items()
            }
        
        zone_manager = ZoneManager(
            zones=zones_config if not dynamic_enabled else None,
            dynamic_detection=dynamic_enabled,
            fallback_to_static=bool(zones_config),
            litter_box_model_path=litter_box_model,
        )
        
        if dynamic_enabled:
            logger.info("Dynamic litter box detection enabled - will auto-detect litter box")
        elif zones_config:
            logger.info(f"Using {len(zones_config)} static zone(s) from config")
        else:
            logger.warning("No zones configured! Please enable dynamic detection or configure static zones.")
        
        return zone_manager
    
    def _setup_state_machine(self) -> MultiZoneStateMachine:
        """Initialize state machine for all zones."""
        sm = MultiZoneStateMachine(
            min_occupancy_seconds=settings.timing.min_occupancy_seconds,
            cooldown_seconds=settings.timing.cooldown_seconds,
            max_session_minutes=settings.timing.max_session_minutes,
        )
        
        # Add zone(s) - will be updated dynamically as litter box is detected
        # For now add a placeholder that will be updated
        sm.add_zone("litter_box_dynamic", "Auto-Detected Litter Box")
        
        # Wire up callbacks
        sm.on_state_change = self._on_state_change
        sm.on_dispatch_ready = self._on_dispatch_ready
        sm.on_cat_entered = self._on_cat_entered
        sm.on_cat_exited = self._on_cat_exited
        
        return sm
    
    def _setup_robot(self) -> RobotAdapter | None:
        """Initialize robot controller if enabled."""
        if not settings.robot.enabled:
            logger.info("Robot control disabled")
            return None
        
        controller = create_robot_controller(
            enabled=settings.robot.enabled,
            room_name=settings.robot.room_name,
        )
        
        if controller is None:
            return None
        
        adapter = RobotAdapter(
            controller=controller,
            room_name=settings.robot.room_name,
            dispatch_delay=settings.robot.dispatch_delay_seconds,
            emergency_stop_on_cat=settings.robot.emergency_stop_on_cat_detected,
        )
        
        return adapter
    
    def _on_state_change(self, event: StateEvent):
        """Handle state change events."""
        self.event_logger.log_state_change(event)
    
    def _on_dispatch_ready(self, session: Session):
        """Handle dispatch ready events."""
        self.event_logger.log_dispatch_ready(session)
        
        if self.robot_adapter:
            self.robot_adapter.on_dispatch_ready(session)
    
    def _on_cat_entered(self, session: Session):
        """Handle cat entered events."""
        self.event_logger.log_cat_entered(session)
        
        if self.robot_adapter:
            self.robot_adapter.on_cat_entered(session)
    
    def _on_cat_exited(self, session: Session):
        """Handle cat exited events."""
        self.event_logger.log_cat_exited(session)
    
    def _detection_loop(self):
        """Main detection loop (runs in separate thread)."""
        logger.info("Detection loop started")
        
        last_inference = 0.0
        interval = settings.detection.inference_interval
        
        # Warm-up: wait for first valid frame and detect litter box
        if self.zone_manager and hasattr(self.zone_manager, '_litter_detector'):
            logger.info("Warming up dynamic litter box detection...")
            warmup_frames = 0
            while not self._shutdown_event.is_set() and warmup_frames < 30:
                frame = self.camera.get_frame() if self.camera else None
                if frame is not None:
                    self.zone_manager.update_dynamic_zone(frame, [])
                    warmup_frames += 1
                time.sleep(0.033)
            
            zones = self.zone_manager.get_zones()
            if zones:
                for zone in zones:
                    logger.info(f"✓ Litter box detected: '{zone.name}' at {zone.bbox} (confidence: {zone.confidence:.2f})")
                    # Register each zone in state machine
                    if self.state_machine:
                        self.state_machine.add_zone(zone.id, zone.name)
            else:
                logger.warning("Could not auto-detect litter box, will try continuously...")
        
        while not self._shutdown_event.is_set():
            loop_start = time.monotonic()
            
            # Get frame from camera (may be None if camera disconnected)
            frame = self.camera.get_frame() if self.camera else None
            
            # Send placeholder to dashboard if no frame available
            if frame is None:
                if self.dashboard:
                    self.dashboard.update_frame(None, None)
                time.sleep(0.1)  # Sleep longer when no camera
                continue
            
            detections: list[Detection] = []
            
            # Run inference at configured interval
            if loop_start - last_inference >= interval:
                inference_start = time.perf_counter()
                detections = self.detector.detect(frame)
                inference_time = time.perf_counter() - inference_start
                
                self._last_inference_time = inference_time
                self._detection_count += 1
                
                # Update running average
                alpha = 0.1
                self._avg_latency = (1 - alpha) * self._avg_latency + alpha * inference_time
                
                last_inference = loop_start
                
                # Log detection
                if detections:
                    best_conf = max(d.confidence for d in detections)
                    self.event_logger.log_detection(None, best_conf, inference_time * 1000)
            
            # Update dynamic litter box zone (separate from cat detection)
            if self.zone_manager:
                now = time.time()
                if now - self._last_zone_update >= self._zone_update_interval:
                    updated = self.zone_manager.update_dynamic_zone(frame, detections)
                    self._last_zone_update = now
                    
                    # Register any new zones with state machine
                    if updated and self.state_machine:
                        for zone in self.zone_manager.get_zones():
                            self.state_machine.add_zone(zone.id, zone.name)
            
            # Match detections to zones
            if self.zone_manager and self.state_machine:
                zone_detections = self.zone_manager.match_detections_to_zones(detections)

                # Update state machine
                self.state_machine.update_zone_detections(zone_detections)

            # Send frame to dashboard
            if self.dashboard:
                overlay = None
                if self.zone_manager:
                    overlay = self.zone_manager.draw_zones(
                        frame.copy(),
                        static_color=(255, 128, 0),
                        dynamic_color=(0, 255, 128),
                        thickness=2,
                        font_scale=0.5,
                    )
                    # Add detection boxes
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det.bbox)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        label = f"{det.class_name}: {det.confidence:.2f}"
                        cv2.putText(overlay, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                self.dashboard.update_frame(frame, overlay)

            # Small sleep to prevent busy-waiting
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, 0.001 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Detection loop stopped")
    
    def _create_status_table(self) -> Table:
        """Create Rich table for status display."""
        table = Table(title="Cat Litter Monitor Status")
        
        table.add_column("Zone", style="cyan")
        table.add_column("State", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Sessions", style="blue")
        
        if self.state_machine:
            for stats in self.state_machine.get_all_stats():
                zone_name = stats.get("zone_name", "Unknown")
                state = stats.get("state", "unknown")
                duration = f"{stats.get('state_duration_seconds', 0):.1f}s"
                sessions = str(stats.get("total_sessions", 0))
                
                # Color-code states
                state_style = {
                    "idle": "dim",
                    "cat_entered": "yellow",
                    "cat_inside": "red",
                    "cat_exited": "yellow",
                    "cooldown": "blue",
                    "dispatch_ready": "green bold",
                }.get(state, "white")
                
                table.add_row(zone_name, f"[{state_style}]{state}[/{state_style}]", duration, sessions)
        
        # Add zone detection status
        if self.zone_manager:
            zone = self.zone_manager.get_primary_zone()
            if zone and zone.dynamic:
                table.add_row(
                    "[dim]Zone Detection[/dim]",
                    f"[green]Active[/green]" if zone.confidence > 0.5 else f"[yellow]Low Conf[/yellow]",
                    "",
                    f"{zone.confidence:.2f}"
                )
        
        return table
    
    def _visualization_loop(self):
        """Visualization loop for debugging (runs in main thread)."""
        if not settings.visualization.enabled:
            # Block until shutdown when running headless
            self._shutdown_event.wait()
            return
        
        window_name = settings.visualization.window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Create placeholder frame for when camera is unavailable
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Feed", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        while self.running and not self._shutdown_event.is_set():
            frame = self.camera.get_frame() if self.camera else None
            if frame is None:
                # Show placeholder when camera unavailable
                cv2.imshow(window_name, placeholder)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    self.shutdown()
                    break
                continue
            
            # Draw zones
            if settings.visualization.show_zones and self.zone_manager:
                frame = self.zone_manager.draw_zones(
                    frame,
                    static_color=(255, 128, 0),      # Orange for static
                    dynamic_color=(0, 255, 128),     # Green for dynamic
                    thickness=settings.visualization.thickness,
                    font_scale=settings.visualization.font_scale,
                )
            
            # Draw stats overlay
            stats_text = [
                f"Inference: {self._last_inference_time*1000:.1f}ms",
                f"Avg: {self._avg_latency*1000:.1f}ms",
                f"Detections: {self._detection_count}",
            ]
            
            # Add zone detection status
            if self.zone_manager:
                zone_stats = self.zone_manager.get_stats()
                if zone_stats.get("has_dynamic_zone"):
                    stats_text.append(f"Dynamic Zones: {zone_stats.get('dynamic_zones', 0)}")
            
            y_offset = 30
            for text in stats_text:
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                y_offset += 25
            
            # Draw zone states
            if self.state_machine:
                y_offset = 150
                for stats in self.state_machine.get_all_stats():
                    zone_name = stats.get("zone_name", "Unknown")
                    state = stats.get("state", "unknown")
                    text = f"{zone_name}: {state}"
                    
                    color = {
                        "idle": (128, 128, 128),
                        "cat_entered": (0, 255, 255),
                        "cat_inside": (0, 0, 255),
                        "cat_exited": (0, 255, 255),
                        "cooldown": (255, 128, 0),
                        "dispatch_ready": (0, 255, 0),
                    }.get(state, (255, 255, 255))
                    
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
                    y_offset += 25
            
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
                break
            
            # Handle reset key
            if cv2.waitKey(1) & 0xFF == ord('r'):
                if self.zone_manager:
                    logger.info("Manual reset of litter box detection")
                    self.zone_manager.reset_dynamic_detection()
        
        cv2.destroyAllWindows()
    
    async def start(self):
        """Start the monitor."""
        logger.info("=" * 50)
        logger.info("Cat Litter Monitor Starting")
        logger.info("=" * 50)
        
        self.running = True
        
        # Initialize components that don't depend on camera FIRST
        logger.info("Initializing detector...")
        self.detector = self._setup_detector()
        
        logger.info("Initializing zones...")
        self.zone_manager = self._setup_zones()
        
        logger.info("Initializing state machine...")
        self.state_machine = self._setup_state_machine()
        
        logger.info("Initializing robot controller...")
        robot = self._setup_robot()
        if robot:
            self.robot_adapter = robot
            self.robot_adapter.set_event_loop(asyncio.get_event_loop())
            await robot.connect()
        
        # Start dashboard FIRST (before camera) so it's always available
        if settings.dashboard.enabled:
            logger.info("Starting dashboard server...")
            self.dashboard = DashboardServer(
                monitor=self,
                host=settings.dashboard.host,
                port=settings.dashboard.port,
            )
            self.dashboard.start()
            logger.info(f"Dashboard available at http://{settings.dashboard.host}:{settings.dashboard.port}")
        
        # Start camera initialization in background thread with retry logic
        logger.info("Initializing camera (background)...")
        self._camera_thread = threading.Thread(target=self._camera_init_loop, daemon=True)
        self._camera_thread.start()
        
        # Start detection thread (handles missing camera gracefully)
        logger.info("Starting detection loop...")
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()
        
        # Log startup
        self.event_logger.log("system", {"message": "Monitor started", "dynamic_zones": True})
        
        logger.info("Monitor started successfully!")
        logger.info("Press 'q' in video window to quit, 'r' to reset litter box detection")
        
        return True
    
    def _camera_init_loop(self):
        """Background thread: Initialize camera with retry logic."""
        retry_interval = 5  # seconds
        
        while not self._shutdown_event.is_set():
            try:
                if self.camera is None:
                    logger.info("Creating camera client...")
                    self.camera = self._setup_camera()
                    self.camera.start()
                
                # Wait for camera to connect (up to 15 seconds — camera needs time for stream negotiation + warmup)
                for _ in range(150):  # 15 seconds
                    if self.camera.is_connected():
                        logger.info("✓ Camera connected successfully!")
                        # Camera is connected, just keep monitoring
                        while self.camera.is_connected() and not self._shutdown_event.is_set():
                            time.sleep(1)
                        logger.warning("Camera disconnected, will retry...")
                        break
                    time.sleep(0.1)
                else:
                    logger.warning(f"Camera not connected, retrying in {retry_interval}s...")
                
                # Clean up and retry
                if self.camera:
                    self.camera.stop()
                    self.camera = None
                
                time.sleep(retry_interval)
                
            except Exception as e:
                logger.error(f"Camera init error: {e}")
                if self.camera:
                    try:
                        self.camera.stop()
                    except:
                        pass
                    self.camera = None
                time.sleep(retry_interval)
    
    def shutdown(self):
        """Shutdown the monitor."""
        if not self.running:
            return
        
        logger.info("Shutting down...")
        self.running = False
        self._shutdown_event.set()
        
        # Stop dashboard
        if self.dashboard:
            logger.info("Stopping dashboard server...")
            self.dashboard.stop()

        # Stop detection thread
        if hasattr(self, '_detection_thread'):
            self._detection_thread.join(timeout=5)
        
        # Stop camera thread
        if hasattr(self, '_camera_thread'):
            self._camera_thread.join(timeout=5)

        # Stop camera
        if self.camera:
            self.camera.stop()

        # Log shutdown
        self.event_logger.log("system", {"message": "Monitor stopped"})
        
        logger.info("Shutdown complete")
    
    async def run(self):
        """Main run loop."""
        if not await self.start():
            return 1
        
        try:
            # Run visualization (blocks until quit)
            self._visualization_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.shutdown()
        
        return 0


def main():
    """Main entry point."""
    monitor = CatLitterMonitor()
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        monitor.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    try:
        exit_code = asyncio.run(monitor.run())
        sys.exit(exit_code)
    except Exception as e:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
