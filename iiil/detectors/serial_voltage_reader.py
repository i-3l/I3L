import serial
import time


class SerialVoltageReader:
    """
    Encapsulates serial port logic for reading voltage values from a force sensor or similar device.
    """
    def __init__(self, serial_port, baudrate, timeout):
        self._serial_port = serial_port
        self._baudrate = baudrate
        self._timeout = timeout
        self._last_value = None
        self._open_serial()

    def _open_serial(self):
        self._ser = serial.Serial(
            port=self._serial_port,
            baudrate=self._baudrate,
            timeout=self._timeout
        )
        # Give some time for the serial port to initialize
        time.sleep(0.5)

    def read_voltage(self):
        """
        Reads the voltage from the serial port of the first half of the voltage divider.
        Returns the most recent value, or the last value if no new data is available.
        """
        if not self._ser.is_open:
            self._ser.open()
        try:
            # Read all available lines and take the last one
            lines = []
            while self._ser.in_waiting:
                line = self._ser.readline().decode('utf-8').strip()
                if line:  # Only append non-empty lines
                    lines.append(line)
            if lines:  # If we got any lines
                value = float(lines[-1])  # Take the most recent value
                self._last_value = value
                print(f"Voltage: {value}")
                return value
            else:
                return self._last_value if self._last_value is not None else float('inf')
        except Exception:
            # If reading fails, return last value or a high value (no interrupt)
            return self._last_value if self._last_value is not None else float('inf')

    def close(self):
        if self._ser.is_open:
            self._ser.close() 