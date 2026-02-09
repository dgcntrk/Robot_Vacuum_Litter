#!/usr/bin/env python3
"""
Cat Litter Monitor - Live Monitoring with State Machine
Real-time visualization showing:
- Litter box zones (auto-detected)
- Cat detections
- State machine status per zone
- Countdown timers
- Event log
"""

import os
# Set FFmpeg RTSP transport option BEFORE importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;5000000|probesize;5000000|buffer_size;65536|stimeout;60000000"

import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque
import cv2
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import threading

from src.camera import RTSPClient
from src.detection import create_detector, ZoneManager
from src.state import MultiZoneStateMachine, LitterState, StateEvent, Session
from src.robot.shark import SharkController
from src.config import settings

# Constants
WINDOW_NAME = "Cat Litter Monitor - Live"
EVENT_LOG_SIZE = 10
HEADLESS = os.environ.get('HEADLESS', '').lower() in ('1', 'true', 'yes')

STATE_COLORS = {
    LitterState.IDLE: (128, 128, 128),  # Gray
    LitterState.CAT_ENTERED: (0, 255, 255),  # Yellow
    LitterState.CAT_INSIDE: (0, 0, 255),  # Red
    LitterState.CAT_EXITED: (255, 165, 0),  # Orange
    LitterState.COOLDOWN: (255, 128, 0),  # Blue-orange
    LitterState.DISPATCH_READY: (0, 255, 0),  # Green
}


class LiveMonitor:
    """Live monitoring system with visualization."""
    
    def __init__(self):
        self.running = False
        self.start_time = time.time()
        
        # Components
        self.camera: RTSPClient | None = None
        self.detector = None
        self.zone_manager: ZoneManager | None = None
        self.state_machine: MultiZoneStateMachine | None = None
        
        # Event log (in-memory)
        self.event_log = deque(maxlen=EVENT_LOG_SIZE)
        
        # Robot dispatch visual indicator
        self._dispatch_indicator: tuple[str, float] | None = None  # (message, expiry_time)
        
        # Temporal smoothing buffer (last 5 detection results)
        self.detection_buffer = deque(maxlen=5)
        
        # Robot controller
        self.shark: SharkController | None = None
        self._robot_loop: asyncio.AbstractEventLoop | None = None
        
        # Stats
        self.fps = 0.0
        self.inference_time = 0.0
        self.frame_count = 0
        self.fps_time = time.time()
        
        # Log file
        self.log_file = Path("logs/events.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Dashboard export tracking
        self.last_dashboard_update = 0.0
    
    def _log_event(self, message: str, level: str = "INFO"):
        """Log event to console, file, and in-memory log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        # Console
        print(log_entry)
        
        # File
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")
        except:
            pass
        
        # In-memory (for overlay)
        self.event_log.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })
    
    def _on_state_change(self, event: StateEvent):
        """Handle state change events."""
        self._log_event(
            f"Zone '{event.zone_name}': {event.old_state.value} → {event.new_state.value}",
            level="STATE"
        )
    
    def _on_dispatch_ready(self, session: Session):
        """Handle dispatch ready events — dispatch robot to clean."""
        message = f"ROBOT DISPATCH: Litter box '{session.zone_name}' needs cleaning"
        self._log_event(message, level="DISPATCH")
        print("\n" + "=" * 60)
        print(f"🤖 {message}")
        print(f"   Session: {session.session_id}")
        print(f"   Duration: {session.duration_seconds:.1f}s")
        print(f"   Occupancy: {session.occupancy_seconds:.1f}s")
        print("=" * 60 + "\n")
        
        # Dispatch the Shark robot
        if self.shark and self._robot_loop:
            room = getattr(settings, 'robot', None) and settings.robot.room_name or "Litter"
            self._log_event(f"Sending Shark robot to clean '{room}'", level="ROBOT")
            
            def _dispatch():
                asyncio.run(self._async_dispatch(room))
            
            threading.Thread(target=_dispatch, daemon=True).start()
            
            # Show visual indicator for 10 seconds
            self._dispatch_indicator = (f"🦈 ROBOT DISPATCHED TO '{room}'!", time.time() + 10)
    
    async def _async_dispatch(self, room: str):
        """Async dispatch robot to clean."""
        try:
            shark = SharkController(
                household_id=settings.robot.household_id,
                dsn=settings.robot.dsn,
                floor_id=settings.robot.floor_id,
            )
            connected = await shark.connect()
            if connected:
                success = await shark.dispatch(room)
                if success:
                    self._log_event(f"✅ Shark robot dispatched to '{room}'!", level="ROBOT")
                else:
                    self._log_event(f"❌ Shark dispatch failed", level="ROBOT")
            else:
                self._log_event(f"❌ Could not connect to Shark robot", level="ROBOT")
        except Exception as e:
            self._log_event(f"❌ Robot dispatch error: {e}", level="ROBOT")
    
    def _on_cat_entered(self, session: Session):
        """Handle cat entered events."""
        self._log_event(
            f"Cat entered zone '{session.zone_name}' (session {session.session_id})",
            level="ENTER"
        )
    
    def _on_cat_exited(self, session: Session):
        """Handle cat exited events."""
        self._log_event(
            f"Cat exited zone '{session.zone_name}' (duration: {session.duration_seconds:.1f}s)",
            level="EXIT"
        )
    
    def setup(self):
        """Initialize all components."""
        print("=" * 60)
        print("Cat Litter Monitor - Live Monitoring")
        print("=" * 60)
        
        # Camera
        print("Initializing camera...")
        self.camera = RTSPClient(
            rtsp_url=settings.camera.rtsp_url,
            fps=settings.camera.fps,
            resolution=settings.camera.resolution,
            reconnect_interval=settings.camera.reconnect_interval,
            use_tcp=settings.camera.use_tcp,
        )
        self.camera.start()
        
        # Wait for connection (up to 15 seconds)
        print("Waiting for camera connection...")
        for _ in range(150):
            if self.camera.is_connected():
                print("✓ Camera connected!")
                break
            time.sleep(0.1)
        else:
            print("⚠ Camera not connected, will continue anyway")
        
        # Detector (YOLOv8n for cats)
        print("Loading cat detector (YOLOv8n)...")
        self.detector = create_detector(
            provider=settings.detection.provider,
            model_path=settings.detection.model_path,
            confidence_threshold=settings.detection.confidence_threshold,
            target_classes=["cat"],  # Only detect cats
        )
        print("✓ Cat detector loaded!")
        
        # Zone Manager (with litter box detection)
        print("Initializing zone manager (dynamic litter box detection)...")
        self.zone_manager = ZoneManager(
            zones=None,  # No static zones
            dynamic_detection=True,
            fallback_to_static=False,
            litter_box_model_path=settings.detection.litter_box_model_path,
        )
        print("✓ Zone manager initialized!")
        
        # State Machine
        print("Initializing state machine...")
        self.state_machine = MultiZoneStateMachine(
            min_occupancy_seconds=settings.timing.min_occupancy_seconds,
            cooldown_seconds=settings.timing.cooldown_seconds,
            max_session_minutes=settings.timing.max_session_minutes,
        )
        
        # Wire up callbacks
        self.state_machine.on_state_change = self._on_state_change
        self.state_machine.on_dispatch_ready = self._on_dispatch_ready
        self.state_machine.on_cat_entered = self._on_cat_entered
        self.state_machine.on_cat_exited = self._on_cat_exited
        
        print("✓ State machine initialized!")
        
        # Robot controller
        if settings.robot.enabled and settings.robot.household_id and settings.robot.dsn:
            self.shark = True  # Flag to enable dispatch (connections created per-dispatch)
            self._robot_loop = True
            print(f"✓ Shark robot enabled (room: {settings.robot.room_name})")
        else:
            print("⚠ Robot dispatch disabled (enable in settings.yaml)")
        
        # Warm-up litter box detection
        print("Warming up litter box detection...")
        warmup_count = 0
        for _ in range(30):
            frame = self.camera.get_frame()
            if frame is not None:
                self.zone_manager.update_dynamic_zone(frame, [])
                warmup_count += 1
            time.sleep(0.033)
        
        zone = self.zone_manager.get_primary_zone()
        if zone:
            print(f"✓ Litter box detected at {zone.bbox} (confidence: {zone.confidence:.2f})")
        else:
            print("⚠ Litter box not detected yet, will continue trying...")
        
        print("\nMonitor ready!")
        print("Press 'q' to quit, 'r' to reset litter box detection")
        print("-" * 60)
        
        self._log_event("Monitor started", level="SYSTEM")
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw all overlays on frame."""
        h, w = frame.shape[:2]
        
        # 0. Draw robot dispatch banner (if active)
        if self._dispatch_indicator:
            message, expiry = self._dispatch_indicator
            if time.time() < expiry:
                # Big green banner at top
                banner_height = 60
                cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 200, 0), -1)
                cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 255, 0), 3)
                
                # Center text
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = banner_height - 15
                cv2.putText(frame, message, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            else:
                self._dispatch_indicator = None
        
        # 1. Draw litter box zones
        if self.zone_manager:
            zones = self.zone_manager.get_zones()
            for zone in zones:
                x1, y1, x2, y2 = zone.bbox
                color = (0, 255, 128) if zone.dynamic else (255, 128, 0)
                
                # Draw zone rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Zone label
                label = f"{zone.name}"
                if zone.dynamic:
                    label += f" (conf: {zone.confidence:.2f})"
                
                cv2.putText(
                    frame, label, (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
        
        # 2. Draw state machine status per zone
        y_offset = 40
        if self.state_machine:
            for stats in self.state_machine.get_all_stats():
                zone_name = stats.get("zone_name", "Unknown")
                state = LitterState(stats.get("state", "idle"))
                duration = stats.get("state_duration_seconds", 0)
                
                # State color
                color = STATE_COLORS.get(state, (255, 255, 255))
                
                # State text
                state_text = f"{zone_name}: {state.value.upper()}"
                
                # Add countdown for cooldown state
                if state == LitterState.COOLDOWN and "cooldown_remaining" in stats:
                    remaining = stats["cooldown_remaining"]
                    state_text += f" ({remaining:.1f}s)"
                elif state == LitterState.CAT_ENTERED and "time_to_confirmation" in stats:
                    remaining = stats["time_to_confirmation"]
                    state_text += f" ({remaining:.1f}s to confirm)"
                else:
                    state_text += f" ({duration:.1f}s)"
                
                # Draw state with background
                text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame, (10, y_offset - 20), (10 + text_size[0] + 10, y_offset + 5),
                    color, -1
                )
                cv2.putText(
                    frame, state_text, (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                y_offset += 35
        
        # 3. Draw FPS and inference time
        stats_y = h - 120
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            frame, f"Inference: {self.inference_time*1000:.1f}ms", (10, stats_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        # 4. Draw event log (bottom left)
        log_y = h - 250
        cv2.putText(
            frame, "Event Log:", (10, log_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        log_y += 25
        
        for i, event in enumerate(list(self.event_log)[-8:]):
            level_colors = {
                "INFO": (200, 200, 200),
                "STATE": (0, 255, 255),
                "ENTER": (0, 255, 0),
                "EXIT": (0, 165, 255),
                "DISPATCH": (0, 255, 0),
                "SYSTEM": (255, 255, 255),
            }
            color = level_colors.get(event["level"], (200, 200, 200))
            
            text = f"{event['timestamp'].split()[1]} - {event['message'][:50]}"
            cv2.putText(
                frame, text, (10, log_y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        return frame
    
    def run(self):
        """Main monitoring loop."""
        self.running = True
        
        # Only create window if not headless
        if not HEADLESS:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        last_inference = 0.0
        inference_interval = settings.detection.inference_interval
        last_zone_update = 0.0
        zone_update_interval = 0.5  # Update zone every 500ms
        
        # Placeholder frame
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "Waiting for camera...", (400, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
        )
        
        while self.running:
            loop_start = time.monotonic()
            
            # Get frame
            frame = self.camera.get_frame() if self.camera else None
            
            if frame is None:
                if not HEADLESS:
                    cv2.imshow(WINDOW_NAME, placeholder)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.1)
                continue
            
            detections = []
            
            # Run cat detection at configured interval (~5 FPS)
            if loop_start - last_inference >= inference_interval:
                inference_start = time.perf_counter()
                raw_detections = self.detector.detect(frame)
                self.inference_time = time.perf_counter() - inference_start
                last_inference = loop_start
                
                # Add raw detections to temporal buffer
                self.detection_buffer.append(raw_detections)
            
            # Apply temporal smoothing: if cat detected in ANY of last 5 frames, treat as present
            # Use highest-confidence detection from buffer for display
            cat_detected_in_buffer = False
            best_detection = None
            
            for buffered_detections in self.detection_buffer:
                if buffered_detections:  # Non-empty detection list
                    cat_detected_in_buffer = True
                    # Find highest confidence detection in this frame
                    for det in buffered_detections:
                        if best_detection is None or det.confidence > best_detection.confidence:
                            best_detection = det
            
            # If cat detected in buffer, use best detection; otherwise empty list
            detections = [best_detection] if cat_detected_in_buffer and best_detection else []
            
            # Update litter box zone (separate from cat detection)
            now = time.time()
            if now - last_zone_update >= zone_update_interval:
                self.zone_manager.update_dynamic_zone(frame, detections)
                last_zone_update = now
            
            # Auto-register any new dynamic zones into the state machine
            for zone in self.zone_manager.get_zones():
                if zone.id not in self.state_machine._machines:
                    self.state_machine.add_zone(zone.id, zone.name)
                    self._log_event(f"Registered zone '{zone.name}' ({zone.id})", level="SYSTEM")
            
            # Match cat detections to zones
            zone_detections = self.zone_manager.match_detections_to_zones(detections)
            
            # Update state machine
            self.state_machine.update_zone_detections(zone_detections)
            
            # Draw visualization
            display_frame = frame.copy()
            
            # Draw cat detection boxes (yellow)
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"Cat: {det.confidence:.2f}"
                cv2.putText(
                    display_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
            
            # Draw overlays
            display_frame = self.draw_overlay(display_frame)
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.fps_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_time = current_time
            
            # Export status and frame for web dashboard (every 1 second)
            if current_time - self.last_dashboard_update >= 1.0:
                try:
                    cv2.imwrite('logs/latest_frame.jpg', display_frame)
                    
                    status = {
                        'timestamp': current_time,
                        'zones': [],
                        'events': [],
                        'uptime': current_time - self.start_time,
                        'robot_enabled': bool(self.shark),
                        'fps': self.fps,
                    }
                    
                    # Add zone stats
                    if self.state_machine:
                        status['zones'] = self.state_machine.get_all_stats()
                    
                    # Add recent events (convert deque to list of dicts)
                    status['events'] = list(self.event_log)
                    
                    with open('logs/status.json', 'w') as f:
                        json.dump(status, f, indent=2)
                    
                    self.last_dashboard_update = current_time
                except Exception as e:
                    print(f"Error writing dashboard data: {e}")
            
            # Show frame (only if not headless)
            if not HEADLESS:
                cv2.imshow(WINDOW_NAME, display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._log_event("User quit", level="SYSTEM")
                    break
                elif key == ord('r'):
                    self._log_event("Reset litter box detection", level="SYSTEM")
                    self.zone_manager.reset_dynamic_detection()
            else:
                # In headless mode, use sleep for timing
                time.sleep(0.033)
        
        # Cleanup
        if not HEADLESS:
            cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()
        
        self._log_event("Monitor stopped", level="SYSTEM")


def main():
    monitor = LiveMonitor()
    
    try:
        monitor.setup()
        monitor.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleanup complete")


if __name__ == "__main__":
    main()
