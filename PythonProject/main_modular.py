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

import argparse
import cv2
import numpy as np
import depthai as dai
import warnings

try:
    import keyboard  # For reliable held-key detection (pip install keyboard)
except ImportError:
    keyboard = None

from hand_pose.gesture import recognize_gesture
from hand_pose.flight_control import DroneGestureController
from hand_pose.serial_output import ArduinoSerial
from hand_pose.config import (
    HAND_CONNECTIONS, COLOR_JOINT, COLOR_BONE,
    THROTTLE_NEAR_MM, THROTTLE_FAR_MM
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
    print("[OK] HandLandmarker initialized (IMAGE mode - synchronous, ~20-25ms latency)", flush=True)

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
    # 1. Use FAST_DENSITY for highest fill-rate (less holes in depth map)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())

    # 2. CRITICAL: Enable Subpixel for smooth, continuous Z values
    stereo.setSubpixel(True)

    # 3. Extended disparity helps track hand when very close to lens
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    # 4. Correctly apply hardware post-processing filters via initialConfig
    # Spatial filter: Fills holes and smooths depth map horizontally/vertically
    stereo.initialConfig.postProcessing.spatialFilter.enable = True
    stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 2
    stereo.initialConfig.postProcessing.spatialFilter.numIterations = 1

    # Temporal filter: Smooths Z-axis over time so the drone doesn't twitch
    stereo.initialConfig.postProcessing.temporalFilter.enable = True

    # Threshold filter: Only keep depth in valid range
    stereo.initialConfig.postProcessing.thresholdFilter.minRange = 100
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 5000

    # Add SpatialLocationCalculator for accurate 3D depth sampling (like DotP_test.py)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    # Configure ROI for wrist area (will sample depth at hand landmarks)
    spatialConfig = dai.SpatialLocationCalculatorConfigData()
    spatialConfig.depthThresholds.lowerThreshold = 100
    spatialConfig.depthThresholds.upperThreshold = 5000
    spatialConfig.roi = dai.Rect(dai.Point2f(0.4, 0.4), dai.Point2f(0.6, 0.6))
    spatialConfig.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(spatialConfig)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    # Create queues
    print("Creating queues...", flush=True)
    q_video = cam_out.createOutputQueue(maxSize=1, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    q_spatial = spatialLocationCalculator.out.createOutputQueue(maxSize=1, blocking=False)
    print("Queues created, starting pipeline...", flush=True)

    pipeline.start()
    print("Pipeline started", flush=True)

    # Enable laser projector for MORE ACCURATE stereo depth (exactly like DotP_test.py)
    try:
        device.setIrLaserDotProjectorIntensity(0.4)  # 40% intensity
        device.setIrFloodLightIntensity(0.0)
        print("[OK] Active Stereo ENABLED (Laser Dot Projector + SpatialLocationCalculator)", flush=True)
    except Exception as e:
        print(f"Laser Dot Projector not available on this device: {e}", flush=True)

    print("Running — SPACE=arm (in KB mode)  P=keyboard mode  G=gesture mode  W/S=throttle  A/D=yaw  I/J/K/L=tilt/pitch/roll  ONE=hover  FIST=descend  q=quit  r=recalibrate", flush=True)

    last_depth_mm = float((THROTTLE_NEAR_MM + THROTTLE_FAR_MM) / 2)

    # Control mode: GESTURE is the default; P switches to manual keyboard control.
    control_mode = "GESTURE"
    keyboard_roll = 1500
    keyboard_pitch = 1500
    keyboard_throttle = 1000  # start low for safety in keyboard mode
    keyboard_yaw = 1500
    KEY_STEP = 10
    THROTTLE_MIN_SAFE = 1000
    THROTTLE_MAX_SAFE = 2000

    # Space key arming: only in KEYBOARD mode
    space_pressed = False
    ARM_COMMAND = (1500, 1500, 1000, 2000)

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

            gesture_to_use = "NONE"

            # NON-BLOCKING depth - grab latest available or None
            in_depth = q_depth.tryGet()
            depth_frame = None
            if in_depth is not None:
                depth_frame = in_depth.getFrame()

            # NON-BLOCKING spatial data - for more accurate depth sampling
            in_spatial = q_spatial.tryGet()
            spatial_data = None
            if in_spatial is not None:
                spatial_locations = in_spatial.getSpatialLocations()
                if len(spatial_locations) > 0:
                    spatial_data = spatial_locations[0]

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
                    # PRIORITY 1: Use SpatialLocationCalculator data (most accurate from DotP_test.py)
                    if spatial_data is not None:
                        try:
                            last_depth_mm = float(spatial_data.spatialCoordinates.z)
                        except:
                            pass  # Fallback to next method

                    # FALLBACK: Sample depth directly at wrist location
                    if spatial_data is None:
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

                # Only run the gesture controller while in gesture mode.
                if control_mode == "GESTURE" and gesture_to_use != "UNKNOWN":
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

            # Manual keyboard control overlay
            # P = keyboard mode, G = gesture mode
            # W/S = throttle, A/D = yaw, arrows = roll/pitch
            key_code = cv2.waitKey(1)
            key = key_code & 0xFF if key_code != -1 else 0
            space_pressed = False
            if key == ord("q"):
                break
            elif key == ord("r"):
                flight_ctrl.recalibrate_yaw()
            elif key == ord("p") and control_mode != "KEYBOARD":
                control_mode = "KEYBOARD"
                keyboard_roll = 1500
                keyboard_pitch = 1500
                keyboard_throttle = THROTTLE_MIN_SAFE
                keyboard_yaw = 1500
                print("[Control] Switched to KEYBOARD mode", flush=True)
            elif key == ord("g"):
                control_mode = "GESTURE"
                print("[Control] Switched to GESTURE mode", flush=True)

            # Keyboard control: use keyboard library for held-key detection
            if keyboard is not None and control_mode == "KEYBOARD":
                space_pressed = keyboard.is_pressed('space')
                w_pressed = keyboard.is_pressed('w'); s_pressed = keyboard.is_pressed('s')
                a_pressed = keyboard.is_pressed('a'); d_pressed = keyboard.is_pressed('d')
                i_pressed = keyboard.is_pressed('i')
                j_pressed = keyboard.is_pressed('j')
                k_pressed = keyboard.is_pressed('k')
                l_pressed = keyboard.is_pressed('l')

                if space_pressed:
                    roll, pitch, throttle, yaw = ARM_COMMAND
                else:
                    if w_pressed: keyboard_throttle = min(THROTTLE_MAX_SAFE, keyboard_throttle + KEY_STEP)
                    if s_pressed: keyboard_throttle = max(THROTTLE_MIN_SAFE, keyboard_throttle - KEY_STEP)
                    if a_pressed: keyboard_yaw = max(1000, keyboard_yaw - KEY_STEP)
                    if d_pressed: keyboard_yaw = min(2000, keyboard_yaw + KEY_STEP)
                    if i_pressed: keyboard_pitch = max(1000, keyboard_pitch - KEY_STEP)
                    if j_pressed: keyboard_roll = max(1000, keyboard_roll - KEY_STEP)
                    if k_pressed: keyboard_pitch = min(2000, keyboard_pitch + KEY_STEP)
                    if l_pressed: keyboard_roll = min(2000, keyboard_roll + KEY_STEP)
                    roll, pitch, throttle, yaw = keyboard_roll, keyboard_pitch, keyboard_throttle, keyboard_yaw

            elif control_mode == "KEYBOARD":
                space_pressed = (key == 32)
                if space_pressed:
                    roll, pitch, throttle, yaw = ARM_COMMAND
                else:
                    if key == ord("w"): keyboard_throttle = min(THROTTLE_MAX_SAFE, keyboard_throttle + KEY_STEP)
                    elif key == ord("s"): keyboard_throttle = max(THROTTLE_MIN_SAFE, keyboard_throttle - KEY_STEP)
                    elif key == ord("a"): keyboard_yaw = max(1000, keyboard_yaw - KEY_STEP)
                    elif key == ord("d"): keyboard_yaw = min(2000, keyboard_yaw + KEY_STEP)
                    elif key == ord("i"): keyboard_pitch = max(1000, keyboard_pitch - KEY_STEP)
                    elif key == ord("j"): keyboard_roll = max(1000, keyboard_roll - KEY_STEP)
                    elif key == ord("l"): keyboard_roll = min(2000, keyboard_roll + KEY_STEP)
                    roll, pitch, throttle, yaw = keyboard_roll, keyboard_pitch, keyboard_throttle, keyboard_yaw

            # Gesture safety overrides: ONE / FIST always take priority.
            if gesture_to_use in ("ONE", "FIST"):
                control_mode = "GESTURE"
                keyboard_roll = 1500
                keyboard_pitch = 1500
                keyboard_yaw = 1500
                if gesture_to_use == "ONE":
                    keyboard_throttle = 1000
                    roll, pitch, throttle, yaw = 1500, 1500, 1500, 1500
                elif gesture_to_use == "FIST":
                    roll, pitch, throttle, yaw = 1500, 1500, 1400, 1500

            # Send to Arduino every frame (flight control is real-time)
            if arduino:
                arduino.send(roll, pitch, throttle, yaw, force=(control_mode == "KEYBOARD" and space_pressed))

            # PERSISTENT HUD: Always show debug info in top right
            cv2.putText(frame, f"Gesture: {gesture_to_use}", (w - 280, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
            cv2.putText(frame, f"R:{roll} P:{pitch} T:{throttle} Y:{yaw}", (w - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Mode: {control_mode}", (w - 280, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
            arm_status = "ARM!" if (space_pressed and control_mode == "KEYBOARD") else "ready"
            arm_color = (0, 0, 255) if space_pressed else (0, 255, 0)
            cv2.putText(frame, f"ARM: {arm_status}", (w - 280, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
            cv2.putText(frame, f"D:{last_depth_mm:.0f}mm", (w - 280, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1)

            # Display every 2nd frame - ONLY call cv2.waitKey() here to avoid latency
            display_counter += 1
            if display_counter % 2 == 0:
                cv2.imshow("Hand Pose Ground Station", frame)

            frame_count += 1

    finally:
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()

