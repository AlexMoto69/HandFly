#!/usr/bin/env python3
"""
main_modular.py
---------------
Clean, modular hand gesture drone control using MediaPipe hand detection.
Uses OAK-D Pro stereo depth for throttle (Z-axis).

Run:
    python main_modular.py
    python main_modular.py --port COM3  # with Arduino
"""

from typing import List, Tuple, Optional
import argparse
import cv2
import numpy as np
import depthai as dai
import time
import warnings
import math

from hand_pose.gesture import recognize_gesture
from hand_pose.flight_control import DroneGestureController
from hand_pose.serial_output import ArduinoSerial
from hand_pose.config import (
    HAND_CONNECTIONS, COLOR_JOINT, COLOR_BONE,
    THROTTLE_NEAR_MM, THROTTLE_FAR_MM, YAW_ANGLE_MAX
)


def main():
    print("START", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=None, help="Arduino serial port e.g. COM3")
    parser.add_argument("--gesture-hold", type=int, default=3, help="Number of consistent frames to confirm a gesture")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # Init flight controller and optional Arduino
    print("Creating flight_ctrl...", flush=True)
    flight_ctrl = DroneGestureController(smoothing=0.15, deadzone=40)
    print("flight_ctrl created", flush=True)

    arduino = None
    try:
        if args.port:
            print(f"Trying to connect Arduino on {args.port}...", flush=True)
            arduino = ArduinoSerial(port=args.port)
            print("Arduino connected", flush=True)
        else:
            print("[main] No --port — dry-run mode (no serial output)", flush=True)
    except Exception as e:
        print(f"[main] Arduino init error: {e}", flush=True)
        arduino = None

    # Load MediaPipe Hands with Tasks API in IMAGE mode (synchronous, ~20-25ms latency)
    print("Loading MediaPipe Hands (Tasks API + IMAGE mode - synchronous)...", flush=True)

    from mediapipe.tasks.python.vision import hand_landmarker
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision.core import image as mp_image

    try:
        # Try to create with GPU delegate
        base_options = base_options_module.BaseOptions(
            model_asset_path='models/hand_landmarker.task',
            delegate=base_options_module.Delegate.GPU  # RTX 3070 Ti
        )
        print("Attempting GPU acceleration...", flush=True)
    except AttributeError:
        # Fallback if Delegate.GPU not available
        base_options = base_options_module.BaseOptions(
            model_asset_path='models/hand_landmarker.task'
        )
        print("GPU delegate not available, using CPU", flush=True)

    # Use IMAGE mode for synchronous detection (no callbacks)
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    mp_landmarker = hand_landmarker.HandLandmarker.create_from_options(options)
    print("✓ HandLandmarker initialized (IMAGE mode - synchronous, ~20-25ms latency)", flush=True)

    # Build DepthAI pipeline
    print("Creating device...", flush=True)
    device = dai.Device()
    print("device created", flush=True)
    platform = device.getPlatform().name
    fps = 30 if platform == "RVC4" else 15
    frame_type = (dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i)
    print(f"Connected — Platform: {platform}  |  FPS: {fps}", flush=True)

    pipeline = dai.Pipeline(device)
    print("pipeline created", flush=True)

    # RGB camera
    print("Creating RGB camera...", flush=True)
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((1280, 720), frame_type, fps=fps)
    print("RGB camera ready", flush=True)

    # Stereo depth
    print("Creating stereo depth...", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setFps(fps)
    mono_right.setFps(fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
    stereo.setSubpixel(False)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    # Add spatial and temporal filtering for cleaner depth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setDepthLowerThreshold(100)      # Minimum depth (mm)
    stereo.setDepthUpperThreshold(5000)     # Maximum depth (mm)

    # Spatial filter: reduces noise in the depth map
    spatial = stereo.initializedFilters()
    spatial.addFilter(dai.node.SpatialFilter())

    # Temporal filter: smooths depth across frames
    temporal = stereo.initializedFilters()
    temporal.addFilter(dai.node.TemporalFilter())

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Create queues
    print("Creating queues...", flush=True)
    q_video = cam_out.createOutputQueue(maxSize=1, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    print("Queues created, starting pipeline...", flush=True)

    pipeline.start()
    print("Pipeline started", flush=True)

    # Enable laser projector for more accurate stereo depth
    try:
        device.setIrLaserDotProjectorIntensity(0.4)
        device.setIrFloodLightIntensity(0.0)
        print("Laser Dot Projector ENABLED at 40% intensity for accurate depth.", flush=True)
    except Exception as e:
        print(f"Laser Dot Projector not available on this device: {e}", flush=True)

    print("Running — FIVE=fly  FIST/PEACE=stop  'q'=quit  'r'=recalibrate yaw", flush=True)

    last_depth_mm = float((THROTTLE_NEAR_MM + THROTTLE_FAR_MM) / 2)

    # Gesture debouncing: require N consistent frames before we commit a gesture
    GESTURE_HOLD_FRAMES = args.gesture_hold
    _temp_gesture = None
    _temp_count = 0
    _reported_gesture = None

    try:
        frame_count = 0
        display_counter = 0
        h, w = 720, 1280

        while True:
            # BLOCKING frame retrieval - wait for fresh video frame (stabilizes OS thread)
            in_video = q_video.get()
            if in_video is None:
                continue

            # NON-BLOCKING depth - grab latest available or None
            in_depth = q_depth.tryGet()
            depth_frame = None
            if in_depth is not None:
                depth_frame = in_depth.getFrame()

            frame = in_video.getCvFrame()
            h, w = frame.shape[:2]

            # OPTIMIZED: Inline color conversion
            rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_for_mp)

            # Synchronous detection (~16-20ms per frame)
            result = mp_landmarker.detect(mp_image_obj)

            # Initialize default commands
            # Throttle defaults to 1500 (hover) for safety when no hand
            roll, pitch, throttle, yaw = 1500, 1500, 1500, 1500

            # Process detection results - ALL DRAWING INSIDE THIS BLOCK
            if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_landmarks = result.hand_landmarks[0]
                kpts = [(lm.x, lm.y) for lm in hand_landmarks]

                # Get depth from wrist (landmark 0) - ONLY if depth is available
                if depth_frame is not None:
                    px_x = int(np.clip(kpts[0][0] * w, 0, w - 1))
                    px_y = int(np.clip(kpts[0][1] * h, 0, h - 1))

                    # OPTIMIZED: 4x4 ROI for fastest median calculation
                    roi = depth_frame[max(0, px_y - 2):min(h, px_y + 2), max(0, px_x - 2):min(w, px_x + 2)]
                    valid_depths = roi[roi > 0]
                    if len(valid_depths) > 0:
                        last_depth_mm = float(np.median(valid_depths))

                # Gesture detection + debouncing
                current_gesture = recognize_gesture(kpts) or "UNKNOWN"

                # debounce logic: require GESTURE_HOLD_FRAMES identical frames to commit
                # But ONLY change if we see a valid (non-UNKNOWN) gesture
                if current_gesture != "UNKNOWN":
                    if current_gesture == _temp_gesture:
                        _temp_count += 1
                    else:
                        _temp_gesture = current_gesture
                        _temp_count = 1

                    if _temp_count >= GESTURE_HOLD_FRAMES:
                        if _reported_gesture != _temp_gesture:
                            _reported_gesture = _temp_gesture
                            print(f"[Gesture] Reported -> {_reported_gesture}")

                # Use last valid reported gesture, or current if first time
                gesture_to_use = _reported_gesture if _reported_gesture is not None else current_gesture

                # Only send non-UNKNOWN gestures to flight controller
                if gesture_to_use != "UNKNOWN":

                    # Pass depth_frame and frame_shape to flight controller for 3D spatial anchor
                    roll, pitch, throttle, yaw = flight_ctrl.process_hand(gesture_to_use, kpts, depth_frame, frame.shape)

                # Draw hand skeleton with bones AND joints (ONLY when hand detected)
                for (i, j) in HAND_CONNECTIONS:
                    if i < len(kpts) and j < len(kpts):
                        x1 = int(kpts[i][0] * w)
                        y1 = int(kpts[i][1] * h)
                        x2 = int(kpts[j][0] * w)
                        y2 = int(kpts[j][1] * h)
                        cv2.line(frame, (x1, y1), (x2, y2), COLOR_BONE, 2, cv2.LINE_AA)

                # Draw joints (circles at each landmark)
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, COLOR_JOINT, -1)

                # big HUD at top center (only one gesture label)
                hud = gesture_to_use if gesture_to_use else current_gesture
                cv2.putText(frame, hud, (max(10, w//2 - 80), 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 220, 220), 2, cv2.LINE_AA)

            # Send to Arduino every frame (flight control is real-time)
            if arduino:
                arduino.send(roll, pitch, throttle, yaw)

            # PERSISTENT HUD: Always show debug info in top right
            cv2.putText(frame, f"Gesture: {gesture_to_use if 'gesture_to_use' in locals() else 'NONE'}", (w - 280, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
            cv2.putText(frame, f"R:{roll} P:{pitch} T:{throttle} Y:{yaw}", (w - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"D:{last_depth_mm:.0f}mm", (w - 280, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)

            # Display every 2nd frame - ONLY call cv2.waitKey() here to avoid latency
            display_counter += 1
            if display_counter % 2 == 0:
                cv2.imshow("Hand Pose Ground Station", frame)
                # ONLY call waitKey on display frames to avoid latency on every loop
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    flight_ctrl.recalibrate_yaw()

            frame_count += 1

    finally:
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()

