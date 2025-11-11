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

        # Try multiple methods to launch TensorBoard
        import sys
        import shutil
        import time

        # Try methods in order of reliability
        methods = []

        # Method 1: Direct tensorboard command (most reliable in conda env)
        tensorboard_exe = shutil.which("tensorboard")
        if tensorboard_exe:
            methods.append(("Direct tensorboard command", [
                tensorboard_exe,
                f"--logdir={logdir}",
                f"--port={port}",
                "--host=localhost"
            ]))

        # Method 2: Python -m tensorboard.main (correct module path)
        methods.append(("Python -m tensorboard.main", [
            sys.executable,
            "-m",
            "tensorboard.main",
            f"--logdir={logdir}",
            f"--port={port}",
            "--host=localhost"
        ]))

        # Method 3: Check Scripts folder (Windows conda installations)
        scripts_tensorboard = Path(sys.executable).parent / "Scripts" / "tensorboard.exe"
        if scripts_tensorboard.exists():
            methods.append(("Scripts folder tensorboard", [
                str(scripts_tensorboard),
                f"--logdir={logdir}",
                f"--port={port}",
                "--host=localhost"
            ]))

        # Try each method until one succeeds
        last_error = ""
        for method_name, cmd in methods:
            logger.info(f"Trying {method_name}: {' '.join(cmd)}")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )

                # Wait briefly to check if process started successfully
                time.sleep(2)

                if self.process.poll() is None:
                    # Process is still running - success!
                    self.port = port
                    self.logdir = str(logdir)
                    logger.info(f"✓ TensorBoard started successfully on port {port} using {method_name}")

                    # Auto-open browser
                    if auto_open:
                        url = self.get_tensorboard_url()
                        logger.info(f"Opening TensorBoard in browser: {url}")
                        webbrowser.open(url)

                    return True
                else:
                    # Process terminated - read error
                    stderr = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else ""
                    last_error = stderr[:300] if stderr else "Process terminated immediately"
                    logger.warning(f"✗ {method_name} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"✗ {method_name} exception: {e}")
                continue

        # All methods failed
        logger.error("=" * 60)
        logger.error("All TensorBoard launch methods failed!")
        logger.error(f"Last error: {last_error}")
        logger.error("=" * 60)
        logger.error("Solutions:")
        logger.error("1. Ensure TensorBoard is installed in conda environment:")
        logger.error("   conda activate wire_sag")
        logger.error("   pip install tensorboard")
        logger.error("2. Start the app using run_annotation.bat")
        logger.error("3. Or manually start TensorBoard:")
        logger.error(f"   tensorboard --logdir={logdir} --port={port}")
        logger.error("=" * 60)
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
