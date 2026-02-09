"""Event logging and broadcasting."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.state.fsm import Session, StateEvent

logger = logging.getLogger(__name__)


class EventLogger:
    """Logs events to file and optionally broadcasts to subscribers."""
    
    def __init__(self, log_dir: str = "./logs", max_history: int = 1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        
        self._event_history: list[dict] = []
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[dict], None]] = []
        
        # Daily log file
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        self._log_file = self.log_dir / f"events_{self._current_date}.jsonl"
    
    def _get_log_file(self) -> Path:
        """Get current log file, rotating daily."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != self._current_date:
            self._current_date = current_date
            self._log_file = self.log_dir / f"events_{self._current_date}.jsonl"
        return self._log_file
    
    def subscribe(self, callback: Callable[[dict], None]):
        """Subscribe to events in real-time."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[dict], None]):
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def log(self, event_type: str, data: dict):
        """Log an event to file and notify subscribers."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data,
        }
        
        with self._lock:
            # Add to in-memory history
            self._event_history.append(event)
            if len(self._event_history) > self.max_history:
                self._event_history = self._event_history[-self.max_history//2:]
            
            # Write to file
            try:
                log_file = self._get_log_file()
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                logger.error(f"Failed to write event log: {e}")
        
        # Notify subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")
    
    def log_state_change(self, event: StateEvent):
        """Log a state change event."""
        self.log("state_change", event.to_dict())
    
    def log_dispatch_ready(self, session: Session):
        """Log a dispatch ready event."""
        self.log("dispatch_ready", {
            "session": session.to_dict(),
            "message": f"Ready to dispatch robot for zone '{session.zone_name}'",
        })
    
    def log_cat_entered(self, session: Session):
        """Log cat entered event."""
        self.log("cat_entered", {
            "session_id": session.session_id,
            "zone_id": session.zone_id,
            "zone_name": session.zone_name,
        })
    
    def log_cat_exited(self, session: Session):
        """Log cat exited event."""
        self.log("cat_exited", {
            "session_id": session.session_id,
            "zone_id": session.zone_id,
            "zone_name": session.zone_name,
            "duration_seconds": session.duration_seconds,
        })
    
    def log_detection(self, zone_id: str | None, confidence: float, inference_time_ms: float):
        """Log a detection event."""
        self.log("detection", {
            "zone_id": zone_id,
            "confidence": confidence,
            "inference_time_ms": inference_time_ms,
        })
    
    def log_error(self, message: str, exception: Exception | None = None):
        """Log an error event."""
        data = {"message": message}
        if exception:
            data["exception"] = str(exception)
            data["exception_type"] = type(exception).__name__
        self.log("error", data)
    
    def get_recent_events(self, count: int = 100, event_type: str | None = None) -> list[dict]:
        """Get recent events from history."""
        with self._lock:
            events = self._event_history
            if event_type:
                events = [e for e in events if e.get("type") == event_type]
            return events[-count:]
    
    def get_events_for_session(self, session_id: str) -> list[dict]:
        """Get all events for a specific session."""
        with self._lock:
            return [
                e for e in self._event_history
                if e.get("session_id") == session_id
            ]
