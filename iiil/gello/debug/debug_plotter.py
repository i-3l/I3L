"""Base class for real-time debug data plotting."""

import json
import threading
import time
import subprocess
import pickle
import os
import sys
from typing import Optional

from iiil.gello.debug.debug_buffer import CircularBuffer
from iiil.gello.debug.debug_config import PlottingConfig


class DebugPlotter:
    """Manages real-time plotting in a separate subprocess.

    Launches matplotlib in a completely fresh Python interpreter to avoid
    GUI conflicts with Isaac Sim's bundled libraries.
    """

    def __init__(
        self,
        config: PlottingConfig,
        dof: int,
        buffer: CircularBuffer,
        logger=None,
    ):
        """Initialize debug plotter.

        Args:
            config: Plotting configuration
            dof: Degrees of freedom (number of joints)
            buffer: Shared circular buffer with debug data
            logger: Optional logger for warnings/errors
        """
        self.config = config
        self.dof = dof
        self.buffer = buffer
        self.logger = logger

        self._subprocess = None
        self._update_thread = None
        self._stop_event = threading.Event()
        self._data_file = None

    def start(self) -> None:
        """Start plotting in a separate subprocess."""
        if self._subprocess is not None and self._subprocess.poll() is None:
            return  # Already running

        if self.config.backend != "matplotlib":
            if self.logger:
                self.logger.warning(f"Unknown plotting backend: {self.config.backend}")
            return

        try:
            print(f"[DebugPlotter] Starting matplotlib in separate subprocess...")
            print(f"[DebugPlotter] This will open an interactive window on your display")

            # Create temporary file for data sharing
            import tempfile
            self._data_file = os.path.join(tempfile.gettempdir(), f"gello_debug_plot_{os.getpid()}.pkl")

            # Find the standalone plotter script
            plotter_script = os.path.join(
                os.path.dirname(__file__),
                'plotters', 'standalone_plotter.py'
            )

            # Launch subprocess with fresh Python interpreter
            # Use the same Python that's running this code
            python_exe = sys.executable

            # Convert y_limits to regular dict (in case it's a DictConfig from Hydra)
            y_limits_dict = dict(self.config.y_limits)
            # Also convert nested values if they're DictConfig
            for key, value in y_limits_dict.items():
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    y_limits_dict[key] = list(value)

            cmd = [
                python_exe,
                plotter_script,
                '--data-file', self._data_file,
                '--update-rate', str(self.config.update_rate),
                '--window-duration', str(self.config.window_duration),
                '--dof', str(self.dof),
                '--y-limits', json.dumps(y_limits_dict),
            ]

            print(f"[DebugPlotter] Launching: {' '.join(cmd)}")

            self._subprocess = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Start thread to monitor subprocess output
            output_thread = threading.Thread(
                target=self._monitor_subprocess_output,
                daemon=True
            )
            output_thread.start()

            # Start thread to update data file from buffer
            self._update_thread = threading.Thread(target=self._update_data_file, daemon=True)
            self._update_thread.start()

            print(f"[DebugPlotter] Plotter subprocess started (PID: {self._subprocess.pid})")
            print(f"[DebugPlotter] Interactive matplotlib window should appear now!")
            print(f"[DebugPlotter] Data file: {self._data_file}")

            if self.logger:
                self.logger.info(f"Debug plotting started in subprocess (PID: {self._subprocess.pid})")

        except Exception as e:
            print(f"[WARNING] Failed to start plotter subprocess: {e}")
            import traceback
            traceback.print_exc()
            if self.logger:
                self.logger.warning(f"Failed to start plotter subprocess: {e}")

    def _monitor_subprocess_output(self) -> None:
        """Monitor and print subprocess output."""
        try:
            for line in self._subprocess.stdout:
                print(f"[PlotterSubprocess] {line.rstrip()}")
        except:
            pass

    def _update_data_file(self) -> None:
        """Update data file from buffer (runs in separate thread)."""
        try:
            while not self._stop_event.is_set():
                # Check if subprocess is still alive
                if self._subprocess and self._subprocess.poll() is not None:
                    print(f"[DebugPlotter] WARNING: Plotter subprocess exited with code {self._subprocess.returncode}")
                    break

                # Get recent data from buffer
                window_samples = int(self.config.window_duration * 100)  # 100 Hz
                data = self.buffer.get_recent(window_samples)

                if len(data.get("timestamp", [])) > 0:
                    # Write data to pickle file
                    try:
                        with open(self._data_file, 'wb') as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        # Ignore write errors (file might be being read)
                        pass

                # Sleep briefly
                time.sleep(0.05)  # 20 Hz update to data file

        except Exception as e:
            print(f"[DebugPlotter] Error updating data file: {e}")

    def stop(self) -> None:
        """Stop the plotting subprocess."""
        self._stop_event.set()

        if self._update_thread is not None and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)

        if self._subprocess is not None:
            self._subprocess.terminate()
            try:
                self._subprocess.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._subprocess.kill()

        # Clean up data file
        if self._data_file and os.path.exists(self._data_file):
            try:
                os.remove(self._data_file)
            except:
                pass

        self._subprocess = None
        self._update_thread = None
        self._data_file = None

        if self.logger:
            self.logger.info("Debug plotting stopped")

    @property
    def is_running(self) -> bool:
        """Check if plotter is currently running."""
        return self._subprocess is not None and self._subprocess.poll() is None
