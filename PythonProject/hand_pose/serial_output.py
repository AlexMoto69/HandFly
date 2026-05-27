from typing import Optional
import time
# import threading


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
            # Give the Arduino time to reset after opening the serial port (DTR toggles and reboots it).
            # This prevents the PC from flooding the Arduino while it boots the sketch.
            time.sleep(2)
            print(f"[Arduino] Connected on {chosen} @ {baud} baud")

            # Output spike guard state
            self._last_sent = None
            self._pending_spike = None

            # Background RX logging state (echo Arduino prints into Python terminal)
            # self._rx_stop = threading.Event()
            # self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            # self._rx_thread.start()

            # Maximum allowed one-frame jump before confirmation is required
            self._max_jump = {
                "r": 120,
                "p": 120,
                "t": 200,
                "y": 120,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {chosen}: {e}") from e

    # def _rx_loop(self):
    #     try:
    #         while not getattr(self, "_rx_stop", None) or not self._rx_stop.is_set():
    #             try:
    #                 raw = self._ser.readline()
    #             except Exception:
    #                 break
    #
    #             if not raw:
    #                 continue
    #
    #             try:
    #                 line = raw.decode(errors="replace").strip()
    #             except Exception:
    #                 line = repr(raw)
    #
    #             if line:
    #                 print(f"[Arduino RX] {line}")
    #     except Exception:
    #         pass

    def _sanitize(self, roll: int, pitch: int, throttle: int, yaw: int):
        vals = []
        for v in (roll, pitch, throttle, yaw):
            try:
                vals.append(int(v))
            except Exception:
                vals.append(1500)
        return tuple(max(1000, min(2000, v)) for v in vals)

    def _is_immediate_preset(self, cmd):
        # Let common mode-transition presets pass instantly.
        return cmd in {
            (1500, 1500, 1500, 1500),
            (1500, 1500, 1400, 1500),
            (1500, 1500, 1600, 1500),
        }

    def send(self, roll: int, pitch: int, throttle: int, yaw: int, force: bool = False) -> None:
        r, p, t, y = self._sanitize(roll, pitch, throttle, yaw)

        cmd = (r, p, t, y)
        if force or self._last_sent is None or self._is_immediate_preset(cmd):
            out = cmd
            self._pending_spike = None
        else:
            dr = abs(cmd[0] - self._last_sent[0])
            dp = abs(cmd[1] - self._last_sent[1])
            dt = abs(cmd[2] - self._last_sent[2])
            dy = abs(cmd[3] - self._last_sent[3])

            spike = (
                dr > self._max_jump["r"]
                or dp > self._max_jump["p"]
                or dt > self._max_jump["t"]
                or dy > self._max_jump["y"]
            )

            if spike and self._pending_spike != cmd:
                # First suspicious frame: hold previous command.
                self._pending_spike = cmd
                out = self._last_sent
            else:
                # Non-spike, or confirmed same spike twice in a row.
                self._pending_spike = None
                out = cmd

        r, p, t, y = out
        try:
            self._ser.write(f"R:{r} P:{p} T:{t} Y:{y}\n".encode())
            try:
                self._ser.flush()
            except Exception:
                pass
            self._last_sent = out
            # Throttle PC -> Arduino updates: give Arduino 40ms to process and avoid overrun
            time.sleep(0.04)
        except Exception as e:
            print(f"[Arduino] Write error: {e}")

    def close(self):
        try:
            # if hasattr(self, "_rx_stop"):
            #     self._rx_stop.set()
            # if hasattr(self, "_rx_thread") and self._rx_thread and self._rx_thread.is_alive():
            #     self._rx_thread.join(timeout=1.0)
            if hasattr(self, "_ser") and self._ser and self._ser.is_open:
                self._ser.close()
                print("[Arduino] Port closed.")
        except Exception:
            pass
