from typing import Optional


class ArduinoSerial:
    """Simple serial writer that formats RC commands for the Arduino PPM bridge.

    Behavior:
      - Auto-detects common Arduino/USB-serial adapters if port is not provided.
      - Raises a helpful RuntimeError if pyserial is missing.
      - send(roll,pitch,throttle,yaw) clamps to [1000,2000] and writes lines like:
            R:1500 P:1500 T:1000 Y:1500\n
    Use close() to shut the port gracefully.
    """

    def __init__(self, port: Optional[str] = None, baud: int = 115200):
        try:
            import serial
            import serial.tools.list_ports
        except Exception as e:
            raise RuntimeError("pyserial is required for Arduino serial output. Install with: pip install pyserial") from e

        chosen = port
        if chosen is None:
            # Try to auto-detect common Arduino/USB-serial devices
            for p in serial.tools.list_ports.comports():
                desc = (p.description or "").lower()
                if any(k in desc for k in ("arduino", "ch340", "ft232", "usb-serial")):
                    chosen = p.device
                    print(f"[Arduino] Auto-detected: {chosen} ({p.description})")
                    break
            # Fallback to COM3 on Windows if it's present
            if chosen is None:
                ports = [p.device for p in serial.tools.list_ports.comports()]
                if "COM3" in ports:
                    chosen = "COM3"
                    print("[Arduino] No obvious Arduino found — falling back to COM3")

        if chosen is None:
            raise RuntimeError("No Arduino found. Use --port COM<X> to specify one or plug the device.")

        try:
            self._ser = serial.Serial(chosen, baud, timeout=1)
            try:
                self._ser.inter_byte_timeout = 0.01
            except Exception:
                pass
            print(f"[Arduino] Connected on {chosen} @ {baud} baud")
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {chosen}: {e}") from e

    def send(self, roll: int, pitch: int, throttle: int, yaw: int) -> None:
        r, p, t, y = [max(1000, min(2000, v)) for v in (roll, pitch, throttle, yaw)]
        try:
            self._ser.write(f"R:{r} P:{p} T:{t} Y:{y}\n".encode())
            try:
                self._ser.flush()
            except Exception:
                pass
        except Exception as e:
            print(f"[Arduino] Write error: {e}")

    def close(self):
        try:
            if hasattr(self, "_ser") and self._ser and self._ser.is_open:
                self._ser.close()
                print("[Arduino] Port closed.")
        except Exception:
            pass
