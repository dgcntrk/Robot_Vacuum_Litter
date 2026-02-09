"""Robot vacuum controller interface.

This module defines the interface for robot vacuum integration.
Implement BaseRobotController for your specific robot.
"""
from __future__ import annotations

import abc
import asyncio
import logging
from typing import Callable

from src.state.fsm import Session

logger = logging.getLogger(__name__)


class BaseRobotController(abc.ABC):
    """Abstract base class for robot vacuum controllers.
    
    Implement this interface to integrate your robot with the cat litter monitor.
    
    Example implementation for Shark vacuum (from old project):
    
        class SharkController(BaseRobotController):
            def __init__(self, household_id: str, dsn: str, token: str):
                self.household_id = household_id
                self.dsn = dsn
                self.token = token
                self._client = None
            
            async def connect(self) -> bool:
                self._client = StakraClient(self.token, self.household_id, self.dsn)
                await self._client.authenticate()
                return True
            
            async def dispatch(self, room: str | None = None) -> bool:
                # Send robot to clean specific room
                areas_payload = json.dumps({
                    "areas_to_clean": {"UserRoom": [room]},
                    "clean_count": 1,
                    "floor_id": self._floor_id,
                    "cleantype": "dry",
                })
                await self._client.set_desired_properties({
                    "AreasToClean_V3": areas_payload,
                    "Operating_Mode": 2,  # START
                })
                return True
            
            async def stop(self) -> bool:
                await self._client.set_desired_properties({
                    "Operating_Mode": 0,  # STOP
                })
                return True
            
            async def return_to_dock(self) -> bool:
                await self._client.set_desired_properties({
                    "Operating_Mode": 4,  # RETURN
                })
                return True
    """
    
    @abc.abstractmethod
    async def connect(self) -> bool:
        """Connect to the robot.
        
        Returns:
            True if connection successful, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    async def dispatch(self, room: str | None = None) -> bool:
        """Dispatch the robot to clean.
        
        Args:
            room: Optional room name for targeted cleaning.
                 If None, robot may clean entire area.
        
        Returns:
            True if dispatch command sent successfully, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    async def stop(self) -> bool:
        """Emergency stop the robot.
        
        Returns:
            True if stop command sent successfully, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    async def return_to_dock(self) -> bool:
        """Send robot back to dock.
        
        Returns:
            True if return command sent successfully, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if robot is available for dispatch.
        
        Returns:
            True if robot is ready to clean, False otherwise.
        """
        pass


class StubRobotController(BaseRobotController):
    """Stub implementation that just logs actions.
    
    Use this for testing without a real robot.
    """
    
    def __init__(self, room_name: str = "Litter"):
        self.room_name = room_name
        self._connected = False
        self._available = True
    
    async def connect(self) -> bool:
        logger.info("[STUB] Robot connected")
        self._connected = True
        return True
    
    async def dispatch(self, room: str | None = None) -> bool:
        target = room or self.room_name
        logger.info(f"[STUB] Robot dispatched to clean: {target}")
        self._available = False
        return True
    
    async def stop(self) -> bool:
        logger.info("[STUB] Robot stopped")
        return True
    
    async def return_to_dock(self) -> bool:
        logger.info("[STUB] Robot returning to dock")
        self._available = True
        return True
    
    def is_available(self) -> bool:
        return self._available


class RobotAdapter:
    """Adapter that bridges sync event callbacks to async robot controller.
    
    This handles the asyncio thread safety issues when calling robot methods
    from detection callbacks (which run in a separate thread).
    """
    
    def __init__(
        self,
        controller: BaseRobotController,
        room_name: str = "Litter",
        dispatch_delay: float = 5.0,
        emergency_stop_on_cat: bool = True,
    ):
        self.controller = controller
        self.room_name = room_name
        self.dispatch_delay = dispatch_delay
        self.emergency_stop_on_cat = emergency_stop_on_cat
        
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._dispatch_scheduled: bool = False
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for async operations."""
        self._event_loop = loop
    
    def _run_async(self, coro):
        """Run an async coroutine safely from sync context."""
        if self._event_loop and self._event_loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        else:
            # No event loop, just log
            logger.warning("No event loop available for robot command")
    
    def on_dispatch_ready(self, session: Session):
        """Callback for when a zone is ready for robot dispatch.
        
        This is called from the detection thread, so we use
        run_coroutine_threadsafe to safely call async robot methods.
        """
        if not self.controller.is_available():
            logger.info(f"Robot not available, skipping dispatch for {session.zone_name}")
            return
        
        logger.info(
            f"Dispatch ready for zone '{session.zone_name}' "
            f"(session: {session.session_id}, delay: {self.dispatch_delay}s)"
        )
        
        # Schedule dispatch with delay
        if self._event_loop:
            self._event_loop.call_later(
                self.dispatch_delay,
                self._do_dispatch,
                session,
            )
    
    def _do_dispatch(self, session: Session):
        """Actually dispatch the robot."""
        self._run_async(self.controller.dispatch(self.room_name))
    
    def on_cat_entered(self, session: Session):
        """Callback for when a cat enters a zone.
        
        Can be used for emergency stop if robot is cleaning.
        """
        if self.emergency_stop_on_cat and not self.controller.is_available():
            # Robot is cleaning and cat detected - emergency stop
            logger.warning(
                f"Cat entered zone '{session.zone_name}' while robot cleaning! "
                "Initiating emergency stop."
            )
            self._run_async(self.controller.stop())
    
    async def connect(self) -> bool:
        """Connect to the robot."""
        return await self.controller.connect()
    
    async def disconnect(self):
        """Disconnect from the robot."""
        pass  # Override if needed


def create_robot_controller(
    enabled: bool = False,
    room_name: str = "Litter",
    **kwargs,
) -> BaseRobotController | None:
    """Factory function to create a robot controller.
    
    Args:
        enabled: Whether robot control is enabled
        room_name: Name of room to clean
        **kwargs: Additional config for specific implementations
    
    Returns:
        Robot controller instance or None if not enabled
    """
    if not enabled:
        return None
    
    # For now, return stub. Replace with your implementation.
    # Example:
    # if kwargs.get("type") == "shark":
    #     from .shark import SharkController
    #     return SharkController(...)
    
    return StubRobotController(room_name)
