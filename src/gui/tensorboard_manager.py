"""
TensorBoard Manager for GUI integration.

Manages TensorBoard subprocess lifecycle, including:
- Auto port detection (6006-6010)
- Subprocess launching and termination
- Browser auto-open
"""

import logging
import socket
import subprocess
import webbrowser
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TensorBoardManager:
    """Manages TensorBoard subprocess for training visualization."""

    def __init__(self):
        """Initialize TensorBoard manager."""
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.logdir: Optional[str] = None

    def get_available_port(self, start_port: int = 6006, end_port: int = 6010) -> Optional[int]:
        """
        Find an available port in the specified range.

        Args:
            start_port: Starting port number (default: 6006)
            end_port: Ending port number (default: 6010)

        Returns:
            Available port number, or None if no port available
        """
        for port in range(start_port, end_port + 1):
            try:
                # Try to bind to the port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                # Port is in use, try next
                continue

        logger.warning(f"No available port found in range {start_port}-{end_port}")
        return None

    def start_tensorboard(self, logdir: str, port: Optional[int] = None, auto_open: bool = True) -> bool:
        """
        Start TensorBoard subprocess.

        Args:
            logdir: Directory containing TensorBoard logs
            port: Port number (auto-detect if None)
            auto_open: Whether to auto-open browser

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running():
            logger.warning("TensorBoard is already running")
            return True

        # Validate logdir
        logdir_path = Path(logdir)
        if not logdir_path.exists():
            logger.error(f"Log directory does not exist: {logdir}")
            return False

        # Get available port
        if port is None:
            port = self.get_available_port()
            if port is None:
                logger.error("No available port for TensorBoard")
                return False

        try:
            # Start TensorBoard subprocess using Python module
            # This ensures we use the TensorBoard from the current Python environment
            import sys
            cmd = [
                sys.executable,
                "-m",
                "tensorboard",
                f"--logdir={logdir}",
                f"--port={port}",
                "--host=localhost"
            ]

            logger.info(f"Starting TensorBoard: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            self.port = port
            self.logdir = str(logdir)

            logger.info(f"TensorBoard started on port {port}")

            # Auto-open browser
            if auto_open:
                url = self.get_tensorboard_url()
                logger.info(f"Opening TensorBoard in browser: {url}")
                webbrowser.open(url)

            return True

        except FileNotFoundError:
            logger.error("TensorBoard not found. Make sure it's installed in current environment: pip install tensorboard")
            return False
        except ModuleNotFoundError:
            logger.error("TensorBoard module not found. Install it with: pip install tensorboard")
            return False
        except Exception as e:
            logger.error(f"Failed to start TensorBoard: {e}")
            return False

    def stop_tensorboard(self) -> bool:
        """
        Stop TensorBoard subprocess.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_running():
            logger.info("TensorBoard is not running")
            return True

        try:
            logger.info("Stopping TensorBoard...")
            self.process.terminate()

            # Wait for process to terminate (max 5 seconds)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("TensorBoard did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()

            self.process = None
            self.port = None
            self.logdir = None

            logger.info("TensorBoard stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop TensorBoard: {e}")
            return False

    def is_running(self) -> bool:
        """
        Check if TensorBoard is currently running.

        Returns:
            True if running, False otherwise
        """
        if self.process is None:
            return False

        # Check if process is still alive
        return_code = self.process.poll()
        if return_code is not None:
            # Process has terminated
            logger.info(f"TensorBoard process terminated with code {return_code}")
            self.process = None
            self.port = None
            self.logdir = None
            return False

        return True

    def get_tensorboard_url(self) -> Optional[str]:
        """
        Get TensorBoard URL.

        Returns:
            URL string if TensorBoard is running, None otherwise
        """
        if self.is_running() and self.port is not None:
            return f"http://localhost:{self.port}"
        return None

    def get_status(self) -> dict:
        """
        Get TensorBoard status information.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self.is_running(),
            "port": self.port,
            "logdir": self.logdir,
            "url": self.get_tensorboard_url()
        }

    def restart_tensorboard(self, logdir: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Restart TensorBoard with new parameters.

        Args:
            logdir: New log directory (use current if None)
            port: New port (auto-detect if None)

        Returns:
            True if restarted successfully, False otherwise
        """
        # Use current logdir if not specified
        if logdir is None:
            logdir = self.logdir
            if logdir is None:
                logger.error("Cannot restart: no logdir specified")
                return False

        # Stop current instance
        self.stop_tensorboard()

        # Start new instance
        return self.start_tensorboard(logdir, port)

    def __del__(self):
        """Cleanup: stop TensorBoard when manager is destroyed."""
        if self.is_running():
            logger.info("Cleaning up TensorBoard on exit...")
            self.stop_tensorboard()
