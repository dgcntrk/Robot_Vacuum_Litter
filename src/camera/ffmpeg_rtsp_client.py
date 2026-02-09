"""Alternative RTSP client using FFmpeg subprocess."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class FFmpegRTSPClient:
    """RTSP camera capture using FFmpeg subprocess."""
    
    def __init__(
        self,
        rtsp_url: str,
        fps: int = 15,
        resolution: tuple[int, int] = (640, 480),
        reconnect_interval: int = 5,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ):
        self.rtsp_url = rtsp_url
        self.target_fps = fps
        self.resolution = resolution
        self.reconnect_interval = reconnect_interval
        self.on_frame = on_frame
        
        self._process: subprocess.Popen | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._connected = False
        self._capture_thread: threading.Thread | None = None
        
        self._frames_received = 0
        self._start_time: float = 0
        
    def _create_process(self) -> subprocess.Popen | None:
        """Create FFmpeg subprocess for RTSP capture."""
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-vf", f"scale={self.resolution[0]}:{self.resolution[1]}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-r", str(self.target_fps),
            "-an",  # Disable audio
            "-sn",  # Disable subtitles
            "-",
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )
            logger.info(f"FFmpeg process started for: {self.rtsp_url}")
            return process
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return None
    
    def _capture_loop(self):
        """Background thread: read frames from FFmpeg."""
        frame_size = self.resolution[0] * self.resolution[1] * 3  # bgr24 = 3 bytes per pixel
        
        while self._running:
            if not self._connected or self._process is None:
                logger.info("Attempting RTSP reconnect...")
                self._release_process()
                
                self._process = self._create_process()
                if self._process is None:
                    time.sleep(self.reconnect_interval)
                    continue
                
                self._connected = True
                self._start_time = time.monotonic()
                logger.info(f"Connected to RTSP stream: {self.rtsp_url}")
            
            try:
                if self._process.stdout is None:
                    self._connected = False
                    continue
                
                # Read raw frame data
                raw_frame = self._process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    logger.warning("Incomplete frame read, reconnecting...")
                    self._connected = False
                    continue
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.resolution[1], self.resolution[0], 3))
                
                self._frames_received += 1
                
                # Store latest frame
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                
                # Call callback if provided
                if self.on_frame:
                    self.on_frame(frame)
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                self._connected = False
                continue
    
    def _release_process(self):
        """Release the FFmpeg process."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._connected = False
    
    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("FFmpeg RTSP capture started")
    
    def stop(self):
        """Stop the capture thread and release resources."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None
        
        self._release_process()
        logger.info("FFmpeg RTSP capture stopped")
    
    def get_frame(self) -> np.ndarray | None:
        """Get the latest decoded frame (thread-safe copy)."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None
    
    def is_connected(self) -> bool:
        """Check if connected to the stream."""
        return self._connected
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "connected": self._connected,
            "frames_received": self._frames_received,
            "elapsed_seconds": elapsed,
            "fps": self._frames_received / elapsed if elapsed > 0 else 0,
        }
