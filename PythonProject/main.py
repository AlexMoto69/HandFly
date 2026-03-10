"""
main.py
-------
Entry point. Parses args, connects to the OAK-D, builds the pipeline,
and hands off to the display loop.

Run:
    python main.py
    python main.py --fps_limit 15
    python main.py --device <DeviceID-or-IP>
"""
import argparse
import depthai as dai

from hand_pose.pipeline import build_pipeline
from hand_pose.renderer import run_display_loop
from hand_pose.flight_control import DroneGestureController
from hand_pose.serial_output import ArduinoSerial


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="OAK-D hand pose → drone flight control")
    p.add_argument("-d",   "--device",    default=None,
                   help="DeviceID or IP of the OAK camera (auto-detect if omitted)")
    p.add_argument("-fps", "--fps_limit", default=None, type=int,
                   help="FPS cap (default: 15 for RVC2, 30 for RVC4)")
    p.add_argument("-p",   "--port",      default=None,
                   help="Arduino serial port e.g. COM3 or /dev/ttyUSB0 "
                        "(omit to run without hardware — commands printed to screen only)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Camera / device ───────────────────────────────────────────────────────
    device   = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    fps      = args.fps_limit or (30 if platform == "RVC4" else 15)
    print(f"Connected — Platform: {platform}  |  FPS: {fps}")

    # ── Flight controller ─────────────────────────────────────────────────────
    flight_ctrl = DroneGestureController(smoothing=0.2, deadzone=40)

    # ── Arduino serial (optional) ─────────────────────────────────────────────
    arduino = None
    if args.port:
        arduino = ArduinoSerial(port=args.port)
    else:
        print("[main] No --port given — running in dry-run mode (no serial output)")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline, queues = build_pipeline(device, fps)
    print("Running — show your hand!  FIST/PEACE = stop.  FIVE = fly.  'q' = quit.")

    try:
        run_display_loop(pipeline, queues, flight_ctrl, arduino)
    finally:
        if arduino:
            arduino.close()
        print("Done.")


if __name__ == "__main__":
    main()
