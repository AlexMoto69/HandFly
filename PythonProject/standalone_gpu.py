"""
standalone_gpu.py
-----------------
PC-based hand pose estimation + drone flight control with OAK-D Pro using MediaPipe (not depthai NN).
Uses MediaPipe Hands on CPU/GPU and OAK-D Pro stereo depth for Z-axis (throttle).

Requirements:
    depthai>=3.0.0
    opencv-python
    numpy
    pyserial
    mediapipe

Run:
    python standalone_gpu.py
    python standalone_gpu.py --port COM3  # with Arduino
"""

from typing import List, Tuple, Optional
import argparse
import cv2
import numpy as np
import depthai as dai
import time
import sys
import warnings

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.5
THROTTLE_NEAR_MM = 200    # 20 cm -> T:2000
THROTTLE_FAR_MM = 700     # 70 cm -> T:1000
YAW_ANGLE_MAX = 30.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

COLOR_JOINT = (0, 255, 0)
COLOR_BONE = (255, 200, 0)
COLOR_BOX = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)

# ══════════════════════════════════════════════════════════════════════════════
# GESTURE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def _dist(a, b):
    return np.linalg.norm(a - b)

def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def recognize_gesture(kpts: List[Tuple[float, float]]) -> Optional[str]:
    kpts = np.array(kpts)
    d_3_5 = _dist(kpts[3], kpts[5])
    d_2_3 = _dist(kpts[2], kpts[3])
    a0 = _angle(kpts[0], kpts[1], kpts[2])
    a1 = _angle(kpts[1], kpts[2], kpts[3])
    a2 = _angle(kpts[2], kpts[3], kpts[4])
    thumb = 1 if (a0 + a1 + a2 > 460 and d_3_5 / d_2_3 > 1.2) else 0

    def finger(tip, mid, base):
        if kpts[tip][1] < kpts[mid][1] < kpts[base][1]:
            return 1
        if kpts[base][1] < kpts[tip][1]:
            return 0
        return -1

    combo = (thumb, finger(8, 7, 6), finger(12, 11, 10), finger(16, 15, 14), finger(20, 19, 18))
    return {
        (1, 1, 1, 1, 1): "FIVE",
        (0, 0, 0, 0, 0): "FIST",
        (1, 0, 0, 0, 0): "OK",
        (0, 1, 1, 0, 0): "PEACE",
        (0, 1, 0, 0, 0): "ONE",
        (1, 1, 0, 0, 0): "TWO",
        (1, 1, 1, 0, 0): "THREE",
        (0, 1, 1, 1, 1): "FOUR",
    }.get(combo)

# ══════════════════════════════════════════════════════════════════════════════
# YAW MATH
# ══════════════════════════════════════════════════════════════════════════════

def knuckle_yaw_angle(keypoints: List[Tuple[float, float]]) -> float:
    p0 = np.array(keypoints[0])
    p9 = np.array(keypoints[9])
    p5 = np.array(keypoints[5])
    p17 = np.array(keypoints[17])

    up_angle = np.degrees(np.arctan2(p9[1] - p0[1], p9[0] - p0[0]))
    knuckle_angle = np.degrees(np.arctan2(p17[1] - p5[1], p17[0] - p5[0]))

    roll = knuckle_angle - up_angle + 90.0
    roll = (roll + 180.0) % 360.0 - 180.0
    return float(roll)

# ══════════════════════════════════════════════════════════════════════════════
# FLIGHT CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class DroneGestureController:
    def __init__(self, smoothing=0.2, deadzone=40, calibration_frames=30):
        self.alpha = smoothing
        self.deadzone = deadzone
        self.smooth_roll = 1500
        self.smooth_pitch = 1500
        self.smooth_throttle = 1000
        self.smooth_yaw = 1500
        self.cam_min = 0.25
        self.cam_max = 0.75
        self._cal_frames = calibration_frames
        self._cal_samples = []
        self._yaw_neutral = None

    def _dz(self, v, center=1500):
        return center if abs(v - center) < self.deadzone else v

    def _ema(self, raw, prev):
        return int(self.alpha * raw + (1.0 - self.alpha) * prev)

    def recalibrate_yaw(self):
        self._cal_samples = []
        self._yaw_neutral = None
        print("[FlightCtrl] Yaw recalibration started — hold FIVE flat.")

    def process_hand(self, gesture, keypoints, depth_mm):
        if gesture in ("FIST", "PEACE"):
            self.smooth_roll = self.smooth_pitch = 1500
            self.smooth_throttle = 1000
            self.smooth_yaw = 1500
            return 1500, 1500, 1000, 1500

        if gesture == "FIVE":
            raw_roll = self._dz(int(np.interp(
                keypoints[9][0], [self.cam_min, self.cam_max], [1000, 2000])))
            raw_pitch = self._dz(int(np.interp(
                keypoints[9][1], [self.cam_min, self.cam_max], [2000, 1000])))
            raw_throttle = int(np.interp(
                depth_mm, [THROTTLE_NEAR_MM, THROTTLE_FAR_MM], [2000, 1000]))

            raw_angle = knuckle_yaw_angle(keypoints)
            if self._yaw_neutral is None:
                self._cal_samples.append(raw_angle)
                if len(self._cal_samples) >= self._cal_frames:
                    self._yaw_neutral = float(np.mean(self._cal_samples))
                    print(f"[FlightCtrl] Yaw neutral: {self._yaw_neutral:.1f} deg")
                raw_yaw = 1500
            else:
                deviation = (raw_angle - self._yaw_neutral + 180.0) % 360.0 - 180.0
                if abs(deviation) < 2.0:
                    deviation = 0.0
                raw_yaw = int(np.interp(
                    deviation, [-YAW_ANGLE_MAX, YAW_ANGLE_MAX], [1200, 1800]))

            self.smooth_roll = self._ema(raw_roll, self.smooth_roll)
            self.smooth_pitch = self._ema(raw_pitch, self.smooth_pitch)
            self.smooth_throttle = self._ema(raw_throttle, self.smooth_throttle)
            self.smooth_yaw = self._ema(raw_yaw, self.smooth_yaw)

            return (self.smooth_roll, self.smooth_pitch, self.smooth_throttle, self.smooth_yaw)

        return 1500, 1500, 1000, 1500

# ══════════════════════════════════════════════════════════════════════════════
# SERIAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

class ArduinoSerial:
    def __init__(self, port=None, baud=115200):
        import serial
        import serial.tools.list_ports
        if port is None:
            for p in serial.tools.list_ports.comports():
                if "arduino" in p.description.lower() or "ch340" in p.description.lower():
                    port = p.device
                    print(f"[Arduino] Auto-detected: {port} ({p.description})")
                    break
        if port is None:
            raise RuntimeError("No Arduino found. Use --port COM<X> to specify one.")
        self._ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"[Arduino] Connected on {port} @ {baud} baud")

    def send(self, roll, pitch, throttle, yaw):
        r, p, t, y = [max(1000, min(2000, v)) for v in (roll, pitch, throttle, yaw)]
        try:
            self._ser.write(f"R:{r} P:{p} T:{t} Y:{y}\n".encode())
            try:
                self._ser.flush()
            except Exception:
                pass
            time.sleep(0.04)
        except Exception as e:
            print(f"[Arduino] Write error: {e}")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[Arduino] Port closed.")

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_bbox(kpts):
    xs = [k[0] for k in kpts]
    ys = [k[1] for k in kpts]
    m = 0.02
    return (max(min(xs) - m, 0.0), max(min(ys) - m, 0.0),
            min(max(xs) + m, 1.0), min(max(ys) + m, 1.0))

def draw_hand(frame, keypoints_norm, label=""):
    h, w = frame.shape[:2]
    for (i, j) in HAND_CONNECTIONS:
        if i < len(keypoints_norm) and j < len(keypoints_norm):
            cv2.line(frame,
                     (int(keypoints_norm[i][0] * w), int(keypoints_norm[i][1] * h)),
                     (int(keypoints_norm[j][0] * w), int(keypoints_norm[j][1] * h)),
                     COLOR_BONE, 2, cv2.LINE_AA)
    for (x_n, y_n) in keypoints_norm:
        cx, cy = int(x_n * w), int(y_n * h)
        cv2.circle(frame, (cx, cy), 5, COLOR_JOINT, -1)

    x1n, y1n, x2n, y2n = keypoints_bbox(keypoints_norm)
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
    if label:
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("START", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=None, help="Arduino serial port e.g. COM3")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # Init flight controller and optional Arduino
    print("Creating flight_ctrl...", flush=True)
    flight_ctrl = DroneGestureController()
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
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Create queues (remove SpatialLocationCalculator - we'll sample depth directly in Python)
    print("Creating queues...", flush=True)
    q_video = cam_out.createOutputQueue(maxSize=1, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    print("Queues created, starting pipeline...", flush=True)

    pipeline.start()
    print("Pipeline started", flush=True)

    # Enable laser projector for more accurate stereo depth
    # Intensity is in range [0.0, 1.0] (percentage, not 0-800)
    try:
        device.setIrLaserDotProjectorIntensity(0.8)  # 80% intensity
        device.setIrFloodLightIntensity(0.0)  # Disable flood light (optional - cleaner depth)
        print("Laser Dot Projector ENABLED at 80% intensity for accurate depth.", flush=True)
    except Exception as e:
        print(f"Laser Dot Projector not available on this device: {e}", flush=True)

    print("Running — FIVE=fly  FIST/PEACE=stop  'q'=quit  'r'=recalibrate yaw", flush=True)

    last_depth_mm = float((THROTTLE_NEAR_MM + THROTTLE_FAR_MM) / 2)

    try:
        frame_count = 0
        display_counter = 0
        while True:
            # BLOCKING frame retrieval - wait for fresh video frame (stabilizes OS thread)
            in_video = q_video.get()
            if in_video is None:
                continue

            # NON-BLOCKING depth - grab latest available or None
            in_depth = q_depth.tryGet()
            # NOTE: If depth is None, we use last_depth_mm and skip depth lookup

            frame = in_video.getCvFrame()

            # OPTIMIZED: Inline color conversion
            rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_for_mp)

            # Synchronous detection (~16-20ms per frame with model_complexity=0)
            result = mp_landmarker.detect(mp_image_obj)

            # Initialize default commands
            roll, pitch, throttle, yaw = 1500, 1500, 1000, 1500

            # Process detection results - ALL DRAWING INSIDE THIS BLOCK
            if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_landmarks = result.hand_landmarks[0]
                kpts = [(lm.x, lm.y) for lm in hand_landmarks]
                h, w = frame.shape[:2]

                # Get depth from wrist (landmark 0) - ONLY if depth is available
                if in_depth is not None:
                    depth_frame = in_depth.getFrame()
                    px_x = int(np.clip(kpts[0][0] * w, 0, w - 1))
                    px_y = int(np.clip(kpts[0][1] * h, 0, h - 1))

                    # OPTIMIZED: 4x4 ROI for fastest median calculation
                    roi = depth_frame[max(0, px_y - 2):min(h, px_y + 2), max(0, px_x - 2):min(w, px_x + 2)]
                    valid_depths = roi[roi > 0]
                    if len(valid_depths) > 0:
                        last_depth_mm = float(np.median(valid_depths))

                # Process flight commands
                gesture = recognize_gesture(kpts)
                roll, pitch, throttle, yaw = flight_ctrl.process_hand(gesture, kpts, last_depth_mm)

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

            # Send to Arduino every frame (flight control is real-time)
            if arduino:
                arduino.send(roll, pitch, throttle, yaw)

            # OPTIMIZED HUD: Minimal text
            cv2.putText(frame, f"R:{roll} P:{pitch} T:{throttle} Y:{yaw}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"D:{last_depth_mm:.0f}mm", (10, 45),
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
