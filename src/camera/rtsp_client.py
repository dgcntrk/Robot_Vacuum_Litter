"""Low-latency RTSP camera client using ffmpeg subprocess."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
import socket
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class RTSPClient:
    """RTSP camera capture using ffmpeg subprocess.
    
    Uses ffmpeg to pipe raw BGR24 frames to stdout, which are read
    in a background thread. This avoids OpenCV's FFmpeg backend which
    has a hardcoded 30-second timeout that breaks Eufy camera connections.
    """
    
    def __init__(
        self,
        rtsp_url: str,
        fps: int = 5,
        resolution: tuple[int, int] = (640, 480),
        reconnect_interval: int = 5,
        buffer_size: int = 1,
        use_tcp: bool = True,
        on_frame: Callable[[np.ndarray], None] | None = None,
        warmup_seconds: float = 3.0,
    ):
        self.rtsp_url = rtsp_url
        self.target_fps = fps
        self.resolution = resolution
        self.reconnect_interval = reconnect_interval
        self.buffer_size = buffer_size
        self.use_tcp = use_tcp
        self.on_frame = on_frame
        self.warmup_seconds = warmup_seconds
        
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._connected = False
        self._capture_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._frame_interval = 1.0 / fps
        
        # Calculate frame size for raw BGR24: width * height * 3 bytes
        self._frame_size = resolution[0] * resolution[1] * 3
        
        # Stats
        self._frames_received = 0
        self._frames_decoded = 0
        self._dropped_frames = 0
        self._start_time: float = 0
        self._last_frame_time: float = 0
        
        # Find ffmpeg binary
        self._ffmpeg_bin = self._find_ffmpeg()
        
    def _find_ffmpeg(self) -> str:
        """Find ffmpeg binary location."""
        import shutil
        for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]:
            if shutil.which(path):
                return path
        return "ffmpeg"  # Fallback
    
    def _build_ffmpeg_cmd(self) -> list[str]:
        """Build the ffmpeg command for capturing RTSP stream."""
        cmd = [
            self._ffmpeg_bin,
            "-rtsp_transport", "tcp",
            "-analyzeduration", "10000000",
            "-probesize", "10000000",
            "-i", self.rtsp_url,
            "-an",  # Skip audio
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.resolution[0]}:{self.resolution[1]}",
            "-r", str(self.target_fps),
            "-"  # Output to stdout
        ]
        return cmd
    
    def _send_teardown(self):
        """Send RTSP TEARDOWN to free camera session slots."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            host = self.rtsp_url.split("//")[1].split("/")[0].split(":")[0]
            port = 554
            s.connect((host, port))
            teardown = f'TEARDOWN {self.rtsp_url} RTSP/1.0\r\nCSeq: 1\r\n\r\n'
            s.send(teardown.encode())
            try:
                s.recv(1024)
            except Exception:
                pass
            s.close()
            logger.debug("Sent RTSP TEARDOWN")
        except Exception as e:
            logger.debug(f"TEARDOWN failed (may be normal): {e}")
    
    def _start_ffmpeg(self) -> subprocess.Popen | None:
        """Start ffmpeg subprocess."""
        self._send_teardown()  # Clear any stale sessions first
        time.sleep(0.5)
        
        cmd = self._build_ffmpeg_cmd()
        logger.info(f"Starting ffmpeg: {' '.join(cmd)}")
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr for debugging
                bufsize=self._frame_size * 2,  # Buffer ~2 frames
            )
            logger.info(f"ffmpeg started with PID {proc.pid}")
            # Start stderr reader thread
            self._stderr_thread = threading.Thread(
                target=self._read_stderr, args=(proc.stderr,), daemon=True
            )
            self._stderr_thread.start()
            return proc
        except Exception as e:
            logger.error(f"Failed to start ffmpeg: {e}")
            return None
    
    def _read_stderr(self, stderr_pipe):
        """Read and log ffmpeg stderr output."""
        try:
            while True:
                line = stderr_pipe.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    # Log errors at warning level, info at debug
                    if 'error' in line_str.lower() or 'failed' in line_str.lower():
                        logger.warning(f"ffmpeg stderr: {line_str}")
                    else:
                        logger.debug(f"ffmpeg: {line_str}")
        except Exception:
            pass
    
    def _kill_ffmpeg(self):
        """Kill ffmpeg process."""
        if self._ffmpeg_proc is not None:
            try:
                # First try graceful termination
                self._ffmpeg_proc.terminate()
                try:
                    self._ffmpeg_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self._ffmpeg_proc.kill()
                    self._ffmpeg_proc.wait()
                logger.debug(f"ffmpeg process {self._ffmpeg_proc.pid} killed")
            except Exception as e:
                logger.debug(f"Error killing ffmpeg: {e}")
            finally:
                self._ffmpeg_proc = None
    
    def _read_frame_from_pipe(self) -> np.ndarray | None:
        """Read a single raw BGR24 frame from ffmpeg stdout."""
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
            return None
        
        try:
            # Use select to check if data is available (with timeout)
            import select
            ready, _, _ = select.select([self._ffmpeg_proc.stdout], [], [], 1.0)
            if not ready:
                return None  # Timeout - no data available yet
            
            # Read exactly frame_size bytes
            raw_bytes = self._ffmpeg_proc.stdout.read(self._frame_size)
            
            if len(raw_bytes) < self._frame_size:
                # EOF or error
                logger.debug(f"Read only {len(raw_bytes)} bytes, expected {self._frame_size}")
                return None
            
            # Convert bytes to numpy array
            frame = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = frame.reshape((self.resolution[1], self.resolution[0], 3))
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame from pipe: {e}")
            return None
    
    def _monitor_ffmpeg(self):
        """Monitor ffmpeg process and restart if it dies."""
        while self._running:
            if self._ffmpeg_proc is not None:
                ret = self._ffmpeg_proc.poll()
                if ret is not None and self._running:
                    logger.warning(f"ffmpeg exited with code {ret}, will restart")
                    self._connected = False
            time.sleep(1)
    
    def _capture_loop(self):
        """Background thread: read frames from ffmpeg pipe."""
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self._running:
            if not self._connected or self._ffmpeg_proc is None:
                logger.info("Attempting RTSP reconnect...")
                self._kill_ffmpeg()
                self._send_teardown()
                
                self._ffmpeg_proc = self._start_ffmpeg()
                if self._ffmpeg_proc is None:
                    logger.warning(f"Failed to start ffmpeg, retrying in {self.reconnect_interval}s...")
                    time.sleep(self.reconnect_interval)
                    continue
                
                # Wait for ffmpeg to start producing frames
                # Eufy camera needs 10-15s for SPS/PPS negotiation, use minimum 30s warmup
                actual_warmup = max(self.warmup_seconds, 30.0)
                logger.info(f"Warming up stream (max {actual_warmup}s)...")
                warmup_start = time.monotonic()
                warmup_success = False
                
                while time.monotonic() - warmup_start < actual_warmup:
                    frame = self._read_frame_from_pipe()
                    if frame is not None:
                        warmup_success = True
                        # Store first frame
                        with self._frame_lock:
                            self._latest_frame = frame
                        self._last_frame_time = time.monotonic()
                        break
                    time.sleep(0.1)
                
                if not warmup_success:
                    logger.warning("Stream warmup failed, retrying...")
                    self._kill_ffmpeg()
                    time.sleep(self.reconnect_interval)
                    continue
                
                self._connected = True
                self._start_time = time.monotonic()
                consecutive_failures = 0
                logger.info(f"Connected to RTSP stream: {self.rtsp_url}")
            
            try:
                # Read frame from pipe
                frame = self._read_frame_from_pipe()
                
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive read failures ({consecutive_failures}), reconnecting...")
                        self._connected = False
                        consecutive_failures = 0
                    else:
                        time.sleep(0.01)
                    continue
                
                # Reset failure counter
                consecutive_failures = 0
                self._frames_received += 1
                self._last_frame_time = time.monotonic()
                
                # Rate limiting - only process at target FPS
                now = time.monotonic()
                time_since_frame = now - self._last_frame_time
                
                # Store latest frame (always keep newest)
                with self._frame_lock:
                    self._latest_frame = frame
                
                self._frames_decoded += 1
                
                # Call callback if provided
                if self.on_frame:
                    self.on_frame(frame)
                
                # Small sleep to prevent busy-waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    self._connected = False
                    consecutive_failures = 0
                time.sleep(0.1)
                continue
    
    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        
        # Clear any stale sessions before starting
        self._send_teardown()
        time.sleep(0.5)
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor_ffmpeg, daemon=True)
        self._monitor_thread.start()
        logger.info("RTSP capture started")
    
    def stop(self):
        """Stop the capture thread and release resources."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
            self._monitor_thread = None
        
        if self._stderr_thread:
            self._stderr_thread.join(timeout=2)
            self._stderr_thread = None
        
        self._kill_ffmpeg()
        self._send_teardown()
        self._connected = False
        logger.info("RTSP capture stopped")
    
    def get_frame(self) -> np.ndarray | None:
        """Get the latest decoded frame (thread-safe copy)."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None
    
    def is_connected(self) -> bool:
        """Check if connected to the stream."""
        # Consider connected if we got a frame recently (within 2 seconds)
        recently_active = (time.monotonic() - self._last_frame_time) < 2.0
        return self._connected and self._ffmpeg_proc is not None and recently_active
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "connected": self.is_connected(),
            "frames_received": self._frames_received,
            "frames_decoded": self._frames_decoded,
            "dropped_frames": self._dropped_frames,
            "elapsed_seconds": elapsed,
            "decode_fps": self._frames_decoded / elapsed if elapsed > 0 else 0,
        }
