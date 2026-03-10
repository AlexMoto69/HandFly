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


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="OAK-D hand pose estimation")
    p.add_argument("-d",   "--device",    default=None,
                   help="DeviceID or IP of the OAK camera (auto-detect if omitted)")
    p.add_argument("-fps", "--fps_limit", default=None, type=int,
                   help="FPS cap (default: 15 for RVC2, 30 for RVC4)")
    return p.parse_args()


def main():
    args = parse_args()

    device   = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    fps      = args.fps_limit or (30 if platform == "RVC4" else 15)

    print(f"Connected — Platform: {platform}  |  FPS: {fps}")

    pipeline, queues = build_pipeline(device, fps)
    print("Running — show your hand!  Press 'q' to quit.")

    run_display_loop(pipeline, queues)
    print("Done.")


if __name__ == "__main__":
    main()

