"""Litter box state machine for tracking cat occupancy."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable

from src.detection.coreml_detector import Detection

logger = logging.getLogger(__name__)


class LitterState(Enum):
    """States in the litter box occupancy state machine."""
    IDLE = "idle"
    CAT_ENTERED = "cat_entered"
    CAT_INSIDE = "cat_inside"
    CAT_EXITED = "cat_exited"
    COOLDOWN = "cooldown"
    DISPATCH_READY = "dispatch_ready"


@dataclass
class StateEvent:
    """An event emitted when state changes."""
    timestamp: float
    zone_id: str
    zone_name: str
    old_state: LitterState
    new_state: LitterState
    session_id: str
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    @property
    def datetime_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).isoformat()
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "old_state": self.old_state.value,
            "new_state": self.new_state.value,
            "session_id": self.session_id,
            "confidence": self.confidence,
            **self.metadata,
        }


@dataclass
class Session:
    """A single cat litter box session."""
    session_id: str
    zone_id: str
    zone_name: str
    entered_at: float
    confirmed_at: float | None = None
    exited_at: float | None = None
    ready_at: float | None = None
    max_confidence: float = 0.0
    detection_count: int = 0
    
    @property
    def duration_seconds(self) -> float | None:
        """Total session duration from enter to exit."""
        if self.exited_at:
            return self.exited_at - self.entered_at
        return None
    
    @property
    def occupancy_seconds(self) -> float | None:
        """Time from confirmed inside to exit."""
        if self.confirmed_at and self.exited_at:
            return self.exited_at - self.confirmed_at
        return None
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "entered_at": datetime.fromtimestamp(self.entered_at).isoformat(),
            "confirmed_at": datetime.fromtimestamp(self.confirmed_at).isoformat() if self.confirmed_at else None,
            "exited_at": datetime.fromtimestamp(self.exited_at).isoformat() if self.exited_at else None,
            "ready_at": datetime.fromtimestamp(self.ready_at).isoformat() if self.ready_at else None,
            "duration_seconds": self.duration_seconds,
            "occupancy_seconds": self.occupancy_seconds,
            "max_confidence": self.max_confidence,
            "detection_count": self.detection_count,
        }


class ZoneStateMachine:
    """State machine for a single litter box zone."""
    
    def __init__(
        self,
        zone_id: str,
        zone_name: str,
        min_occupancy_seconds: float = 5.0,
        cooldown_seconds: float = 60.0,
        max_session_minutes: float = 10.0,
    ):
        self.zone_id = zone_id
        self.zone_name = zone_name
        self.min_occupancy_seconds = min_occupancy_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_session_seconds = max_session_minutes * 60
        
        self.state = LitterState.IDLE
        self.current_session: Session | None = None
        self.session_history: list[Session] = []
        
        self._state_entry_time = time.time()
        self._session_counter = 0
        self._lock = threading.RLock()
        
        # Callbacks
        self.on_state_change: Callable[[StateEvent], None] | None = None
        self.on_dispatch_ready: Callable[[Session], None] | None = None
        self.on_cat_entered: Callable[[Session], None] | None = None
        self.on_cat_exited: Callable[[Session], None] | None = None
    
    def _transition(
        self,
        new_state: LitterState,
        confidence: float = 0.0,
        metadata: dict | None = None,
    ) -> StateEvent | None:
        """Transition to a new state, emitting event and calling callbacks."""
        old_state = self.state
        
        if old_state == new_state:
            return None
        
        self.state = new_state
        self._state_entry_time = time.time()
        
        event = StateEvent(
            timestamp=self._state_entry_time,
            zone_id=self.zone_id,
            zone_name=self.zone_name,
            old_state=old_state,
            new_state=new_state,
            session_id=self.current_session.session_id if self.current_session else "none",
            confidence=confidence,
            metadata=metadata or {},
        )
        
        logger.info(
            f"Zone '{self.zone_name}': {old_state.value} → {new_state.value}"
        )
        
        # Call state change callback
        if self.on_state_change:
            try:
                self.on_state_change(event)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
        
        # State-specific callbacks
        if new_state == LitterState.CAT_ENTERED and self.on_cat_entered:
            try:
                self.on_cat_entered(self.current_session)
            except Exception as e:
                logger.error(f"Cat entered callback error: {e}")
        
        if new_state == LitterState.CAT_EXITED and self.on_cat_exited:
            try:
                self.on_cat_exited(self.current_session)
            except Exception as e:
                logger.error(f"Cat exited callback error: {e}")
        
        return event
    
    def update(self, cat_detected: bool, confidence: float = 0.0):
        """Update the state machine with a new detection tick.
        
        Args:
            cat_detected: Whether a cat is currently detected in this zone
            confidence: Detection confidence (0.0-1.0)
        """
        with self._lock:
            now = time.time()
            
            # Timeout guard for long sessions (cat might be sleeping in box)
            if self.state in (LitterState.CAT_ENTERED, LitterState.CAT_INSIDE):
                if self.current_session:
                    session_duration = now - self.current_session.entered_at
                    if session_duration > self.max_session_seconds:
                        logger.warning(
                            f"Zone '{self.zone_name}': Session timeout ({session_duration:.0f}s), resetting"
                        )
                        self._transition(LitterState.IDLE, metadata={"reason": "timeout"})
                        self.current_session = None
                        return
            
            # State machine logic
            # Track last time we saw a cat (for grace periods)
            if cat_detected:
                self._last_detection_time = now
            
            if self.state == LitterState.IDLE:
                if cat_detected:
                    self._session_counter += 1
                    self.current_session = Session(
                        session_id=f"{self.zone_id}_{int(now)}_{self._session_counter}",
                        zone_id=self.zone_id,
                        zone_name=self.zone_name,
                        entered_at=now,
                        max_confidence=confidence,
                    )
                    self._transition(LitterState.CAT_ENTERED, confidence)
            
            elif self.state == LitterState.CAT_ENTERED:
                if not cat_detected:
                    # Grace period: allow up to 3s of missed detections before treating as false alarm
                    if not hasattr(self, '_last_detection_time'):
                        self._last_detection_time = now
                    if (now - self._last_detection_time) > 3.0:
                        self._transition(LitterState.IDLE, metadata={"reason": "false_alarm"})
                        self.current_session = None
                elif (now - self.current_session.entered_at) >= self.min_occupancy_seconds:
                    # Cat has stayed long enough to confirm
                    self.current_session.confirmed_at = now
                    self.current_session.max_confidence = max(
                        self.current_session.max_confidence, confidence
                    )
                    self._transition(LitterState.CAT_INSIDE, confidence)
                else:
                    # Still in grace period, track confidence
                    self.current_session.detection_count += 1
                    self.current_session.max_confidence = max(
                        self.current_session.max_confidence, confidence
                    )
            
            elif self.state == LitterState.CAT_INSIDE:
                if cat_detected:
                    self.current_session.detection_count += 1
                    self.current_session.max_confidence = max(
                        self.current_session.max_confidence, confidence
                    )
                else:
                    # Grace period: allow 3s gap before considering exit
                    if (now - self._last_detection_time) > 3.0:
                        self.current_session.exited_at = now
                        self._transition(LitterState.CAT_EXITED)
            
            elif self.state == LitterState.CAT_EXITED:
                elapsed = now - self.current_session.exited_at
                if cat_detected:
                    # Cat returned - go back to inside
                    self._transition(LitterState.CAT_INSIDE, confidence, {"reason": "returned"})
                elif elapsed >= self.cooldown_seconds:
                    # Cooldown complete
                    self._transition(LitterState.COOLDOWN)
            
            elif self.state == LitterState.COOLDOWN:
                # Move to dispatch ready and complete session
                self.current_session.ready_at = now
                session = self.current_session
                
                self._transition(LitterState.DISPATCH_READY)
                
                # Store session history
                self.session_history.append(session)
                if len(self.session_history) > 100:
                    self.session_history.pop(0)
                
                self.current_session = None
                
                # Call dispatch callback
                if self.on_dispatch_ready:
                    try:
                        self.on_dispatch_ready(session)
                    except Exception as e:
                        logger.error(f"Dispatch callback error: {e}")
                
                # Return to idle
                self._transition(LitterState.IDLE, metadata={"reason": "dispatch_complete"})
    
    def get_stats(self) -> dict:
        """Get current state statistics."""
        with self._lock:
            now = time.time()
            stats = {
                "zone_id": self.zone_id,
                "zone_name": self.zone_name,
                "state": self.state.value,
                "state_duration_seconds": now - self._state_entry_time,
                "total_sessions": len(self.session_history),
            }
            
            if self.current_session:
                stats["current_session"] = self.current_session.to_dict()
                
                if self.state == LitterState.CAT_ENTERED:
                    time_to_confirm = self.min_occupancy_seconds - (now - self.current_session.entered_at)
                    stats["time_to_confirmation"] = max(0, time_to_confirm)
                elif self.state == LitterState.CAT_EXITED:
                    cooldown_remaining = self.cooldown_seconds - (now - self.current_session.exited_at)
                    stats["cooldown_remaining"] = max(0, cooldown_remaining)
            
            return stats
    
    def reset(self):
        """Reset the state machine to idle."""
        with self._lock:
            self.state = LitterState.IDLE
            self._state_entry_time = time.time()
            self.current_session = None


class MultiZoneStateMachine:
    """Manages state machines for multiple zones."""
    
    def __init__(
        self,
        min_occupancy_seconds: float = 5.0,
        cooldown_seconds: float = 60.0,
        max_session_minutes: float = 10.0,
    ):
        self.min_occupancy_seconds = min_occupancy_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_session_minutes = max_session_minutes
        
        self._machines: dict[str, ZoneStateMachine] = {}
        self._lock = threading.Lock()
        
        # Global callbacks
        self.on_state_change: Callable[[StateEvent], None] | None = None
        self.on_dispatch_ready: Callable[[Session], None] | None = None
        self.on_cat_entered: Callable[[Session], None] | None = None
        self.on_cat_exited: Callable[[Session], None] | None = None
    
    def add_zone(self, zone_id: str, zone_name: str) -> ZoneStateMachine:
        """Add a zone to track."""
        with self._lock:
            if zone_id not in self._machines:
                machine = ZoneStateMachine(
                    zone_id=zone_id,
                    zone_name=zone_name,
                    min_occupancy_seconds=self.min_occupancy_seconds,
                    cooldown_seconds=self.cooldown_seconds,
                    max_session_minutes=self.max_session_minutes,
                )
                
                # Wire up callbacks
                machine.on_state_change = self.on_state_change
                machine.on_dispatch_ready = self.on_dispatch_ready
                machine.on_cat_entered = self.on_cat_entered
                machine.on_cat_exited = self.on_cat_exited
                
                self._machines[zone_id] = machine
                logger.info(f"Added zone '{zone_name}' to state machine")
            
            return self._machines[zone_id]
    
    def remove_zone(self, zone_id: str):
        """Remove a zone."""
        with self._lock:
            if zone_id in self._machines:
                del self._machines[zone_id]
    
    def update_zone_detections(self, zone_detections: dict[str, list[Detection]]):
        """Update all zones with detection results.
        
        Args:
            zone_detections: Dict mapping zone_id to list of detections in that zone
        """
        with self._lock:
            for zone_id, machine in self._machines.items():
                detections = zone_detections.get(zone_id, [])
                
                if detections:
                    # Cat detected - use highest confidence
                    best_det = max(detections, key=lambda d: d.confidence)
                    machine.update(cat_detected=True, confidence=best_det.confidence)
                else:
                    # No cat in this zone
                    machine.update(cat_detected=False)
    
    def get_all_stats(self) -> list[dict]:
        """Get stats for all zones."""
        with self._lock:
            return [machine.get_stats() for machine in self._machines.values()]
    
    def get_zone_stats(self, zone_id: str) -> dict | None:
        """Get stats for a specific zone."""
        with self._lock:
            if machine := self._machines.get(zone_id):
                return machine.get_stats()
            return None
    
    def reset_all(self):
        """Reset all zones to idle."""
        with self._lock:
            for machine in self._machines.values():
                machine.reset()
    
    def get_recent_sessions(self, count: int = 10) -> list[Session]:
        """Get recent completed sessions across all zones."""
        with self._lock:
            all_sessions = []
            for machine in self._machines.values():
                all_sessions.extend(machine.session_history)
            
            # Sort by ready time (most recent first)
            all_sessions.sort(
                key=lambda s: s.ready_at or s.exited_at or s.entered_at,
                reverse=True,
            )
            
            return all_sessions[:count]
