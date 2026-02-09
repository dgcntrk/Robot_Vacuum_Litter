"""Dashboard server for cat-litter-monitor.

Provides real-time web dashboard with WebSocket updates.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from src.main import CatLitterMonitor
    from src.events.logger import EventLogger
    from src.state.fsm import MultiZoneStateMachine

logger = logging.getLogger(__name__)


class DashboardServer:
    """FastAPI-based dashboard server with WebSocket support."""
    
    def __init__(
        self,
        monitor: CatLitterMonitor,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.monitor = monitor
        self.host = host
        self.port = port
        
        self.app = FastAPI(title="Cat Litter Monitor Dashboard")
        self._setup_routes()
        
        # WebSocket connections
        self._ws_connections: list[WebSocket] = []
        self._ws_lock = threading.Lock()
        
        # Stats tracking
        self._start_time = time.time()
        self._frame_lock = threading.Lock()
        self._current_frame: np.ndarray | None = None
        self._frame_overlay: np.ndarray | None = None
        
        # Event subscriber reference
        self._event_callback = None
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Serve the dashboard HTML."""
            template_path = Path(__file__).parent / "templates" / "index.html"
            if template_path.exists():
                return template_path.read_text()
            return HTMLResponse("<h1>Dashboard template not found</h1>", status_code=500)
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get current system stats."""
            return self._get_stats()
        
        @self.app.get("/api/events")
        async def get_events(limit: int = 50):
            """Get recent events."""
            return self._get_recent_events(limit)
        
        @self.app.get("/api/today")
        async def get_today_stats():
            """Get today's summary stats."""
            return self._get_today_stats()
        
        @self.app.get("/video_feed")
        async def video_feed():
            """MJPEG video stream endpoint."""
            return StreamingResponse(
                self._generate_mjpeg_stream(),
                media_type="multipart/x-mixed-replace;boundary=frame"
            )
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            
            with self._ws_lock:
                self._ws_connections.append(websocket)
            
            try:
                # Send initial state
                await websocket.send_json({
                    "type": "connected",
                    "data": {
                        "stats": self._get_stats(),
                        "events": self._get_recent_events(20),
                        "today": self._get_today_stats(),
                    }
                })
                
                # Keep connection alive and handle client messages
                while True:
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0
                        )
                        # Handle ping/keepalive
                        if message == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({"type": "heartbeat"})
                        
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.debug(f"WebSocket error: {e}")
            finally:
                with self._ws_lock:
                    if websocket in self._ws_connections:
                        self._ws_connections.remove(websocket)
    
    def _get_stats(self) -> dict:
        """Get current system stats."""
        stats = {
            "uptime_seconds": int(time.time() - self._start_time),
            "fps": 0,
            "inference_latency_ms": 0,
            "detection_count": 0,
            "zones": [],
        }
        
        if self.monitor:
            stats["inference_latency_ms"] = round(self.monitor._avg_latency * 1000, 1)
            stats["detection_count"] = self.monitor._detection_count
            
            # Calculate effective FPS
            uptime = time.time() - self._start_time
            if uptime > 0:
                stats["fps"] = round(self.monitor._detection_count / uptime, 1)
            
            # Zone stats
            if self.monitor.state_machine:
                stats["zones"] = self.monitor.state_machine.get_all_stats()
        
        return stats
    
    def _get_recent_events(self, limit: int = 50) -> list:
        """Get recent events from the event logger."""
        if self.monitor and self.monitor.event_logger:
            return self.monitor.event_logger.get_recent_events(limit)
        return []
    
    def _get_today_stats(self) -> dict:
        """Calculate today's stats from event log."""
        today = date.today().isoformat()
        
        stats = {
            "date": today,
            "total_visits": 0,
            "average_duration_seconds": 0,
            "last_visit_time": None,
            "durations": [],
        }
        
        if not self.monitor or not self.monitor.event_logger:
            return stats
        
        # Get all events for today
        events = self.monitor.event_logger.get_recent_events(1000)
        
        sessions_completed = []
        current_sessions = {}
        
        for event in events:
            event_time = event.get("timestamp", "")
            if not event_time.startswith(today):
                continue
                
            event_type = event.get("type", "")
            session_id = event.get("session_id", "")
            
            if event_type == "cat_entered":
                current_sessions[session_id] = {
                    "entered_at": event_time,
                }
            elif event_type == "cat_exited" and session_id in current_sessions:
                session = current_sessions[session_id]
                duration = event.get("duration_seconds")
                if duration:
                    sessions_completed.append({
                        "entered_at": session["entered_at"],
                        "exited_at": event_time,
                        "duration": duration,
                    })
                del current_sessions[session_id]
        
        if sessions_completed:
            stats["total_visits"] = len(sessions_completed)
            durations = [s["duration"] for s in sessions_completed if s["duration"]]
            if durations:
                stats["average_duration_seconds"] = round(sum(durations) / len(durations), 1)
                stats["durations"] = durations
            
            last = sessions_completed[-1]
            stats["last_visit_time"] = last["exited_at"]
        
        return stats
    
    def _generate_no_camera_frame(self) -> np.ndarray:
        """Generate a placeholder frame when camera is unavailable."""
        # Create a 640x480 black frame with text
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add "No Camera Feed" text
        text = "No Camera Feed"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        color = (0, 0, 255)  # Red
        
        # Center the text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, timestamp, ((640 - ts_size[0]) // 2, text_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        # Add status indicator
        status = "Waiting for camera connection..."
        cv2.putText(frame, status, ((640 - 300) // 2, text_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        return frame

    def _generate_mjpeg_stream(self):
        """Generate MJPEG stream from current frames."""
        while True:
            frame = None
            
            with self._frame_lock:
                if self._frame_overlay is not None:
                    frame = self._frame_overlay.copy()
                elif self._current_frame is not None:
                    frame = self._current_frame.copy()
            
            # Use placeholder if no frame available
            if frame is None:
                frame = self._generate_no_camera_frame()
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            
            time.sleep(0.033)  # ~30 FPS stream
    
    def update_frame(self, frame: np.ndarray | None, overlay: np.ndarray | None = None):
        """Update the current frame for streaming.
        
        Args:
            frame: The raw frame, or None if camera unavailable
            overlay: The overlay frame with annotations, or None
        """
        with self._frame_lock:
            if frame is not None:
                self._current_frame = frame.copy()
            else:
                self._current_frame = None
            
            if overlay is not None:
                self._frame_overlay = overlay.copy()
            else:
                self._frame_overlay = None
    
    def _on_event(self, event: dict):
        """Handle events from the event logger - broadcasts to WebSockets."""
        asyncio.create_task(self._broadcast_event(event))
    
    async def _broadcast_event(self, event: dict):
        """Broadcast event to all connected WebSockets."""
        message = {
            "type": "event",
            "data": event,
            "stats": self._get_stats(),
        }
        
        # Get a copy of connections to avoid holding lock during send
        with self._ws_lock:
            connections = self._ws_connections.copy()
        
        disconnected = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        # Clean up disconnected clients
        if disconnected:
            with self._ws_lock:
                for ws in disconnected:
                    if ws in self._ws_connections:
                        self._ws_connections.remove(ws)
    
    async def broadcast_state_update(self):
        """Broadcast state update to all clients."""
        message = {
            "type": "state_update",
            "data": {
                "stats": self._get_stats(),
                "today": self._get_today_stats(),
            }
        }
        
        with self._ws_lock:
            connections = self._ws_connections.copy()
        
        disconnected = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        if disconnected:
            with self._ws_lock:
                for ws in disconnected:
                    if ws in self._ws_connections:
                        self._ws_connections.remove(ws)
    
    def start(self):
        """Start the dashboard server in a background thread."""
        import uvicorn
        
        # Subscribe to events
        if self.monitor and self.monitor.event_logger:
            self._event_callback = self._on_event
            self.monitor.event_logger.subscribe(self._event_callback)
            logger.info("Subscribed to event logger")
        
        def run_server():
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the dashboard server."""
        if self._event_callback and self.monitor and self.monitor.event_logger:
            self.monitor.event_logger.unsubscribe(self._event_callback)
            logger.info("Unsubscribed from event logger")
        
        logger.info("Dashboard server stopped")
    
    def get_frame_update_callback(self):
        """Get a callback function to update frames from the detection loop."""
        def callback(frame: np.ndarray, overlay: np.ndarray | None = None):
            self.update_frame(frame, overlay)
        return callback
