"""
hand_pose_standalone.py
-----------------------
100% standalone hand-pose estimation + drone flight control with OAK-D Pro (RVC2/RVC4).
No dependency on the hand_pose/ package — everything is inlined in this single file.

Requirements (pip install):
    depthai>=3.0.0
    depthai-nodes>=0.3.4
    opencv-python
    numpy
    pyserial          <- only needed if using --port

Models are downloaded automatically from the Luxonis Model Zoo on first run.

Run:
    python hand_pose_standalone.py
    python hand_pose_standalone.py --fps_limit 15
    python hand_pose_standalone.py --device <DeviceID-or-IP>
    python hand_pose_standalone.py --port COM3       # with Arduino

Keys:
    q  — quit
    r  — recalibrate yaw neutral
"""

from collections import defaultdict
from typing import List, Tuple, Optional
import argparse
import textwrap

import cv2
import numpy as np
import depthai as dai
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended
from depthai_nodes.node import ParsingNeuralNetwork

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PADDING              = 0.2
CONFIDENCE_THRESHOLD = 0.5

# Stereo depth throttle range (millimetres)
THROTTLE_NEAR_MM = 200    # 20 cm -> T:2000
THROTTLE_FAR_MM  = 700    # 70 cm -> T:1000

# Yaw: max wrist roll deviation from neutral that maps to 1200/1800
YAW_ANGLE_MAX = 30.0

# MediaPipe 21-landmark skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

COLOR_JOINT = (0, 255, 0)
COLOR_BONE  = (255, 200, 0)
COLOR_BOX   = (0, 200, 255)
COLOR_TEXT  = (255, 255, 255)

PALM_MODEL_SLUG = "luxonis/mediapipe-palm-detection:192x192"
HAND_MODEL_SLUG = "luxonis/mediapipe-hand-landmarker:224x224"

SCRIPT_CODE = textwrap.dedent("""\
    while True:
        try:
            frame = node.inputs["frame_input"].get()
            num_configs_message = node.inputs["num_configs_input"].get()
            conf_seq  = num_configs_message.getSequenceNum()
            frame_seq = frame.getSequenceNum()
            raw = num_configs_message.getData()
            num_configs = len(bytearray(raw)) if raw is not None else 0
            while conf_seq > frame_seq:
                frame = node.inputs["frame_input"].get()
                frame_seq = frame.getSequenceNum()
            for i in range(num_configs):
                cfg = node.inputs["config_input"].get()
                node.outputs["output_config"].send(cfg)
                node.outputs["output_frame"].send(frame)
        except Exception as e:
            node.warn(str(e))
""")


# ══════════════════════════════════════════════════════════════════════════════
# GESTURE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def _dist(a, b): return np.linalg.norm(a - b)

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
        if kpts[tip][1] < kpts[mid][1] < kpts[base][1]: return 1
        if kpts[base][1] < kpts[tip][1]:                 return 0
        return -1

    combo = (thumb, finger(8,7,6), finger(12,11,10), finger(16,15,14), finger(20,19,18))
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
    """
    Rotation-invariant wrist roll angle in degrees.

    Uses the angle between:
      - hand up-vector : kp0 (wrist) -> kp9 (middle MCP)
      - knuckle bar    : kp5 (index MCP) -> kp17 (pinky MCP)

    Result is position-invariant — only a genuine wrist twist changes it.
      ~  0 deg = hand flat/neutral
      + 30 deg = rolled clockwise  -> yaw right
      - 30 deg = rolled anti-clock -> yaw left
    """
    p0  = np.array(keypoints[0])
    p9  = np.array(keypoints[9])
    p5  = np.array(keypoints[5])
    p17 = np.array(keypoints[17])

    up_angle      = np.degrees(np.arctan2(p9[1]  - p0[1],  p9[0]  - p0[0]))
    knuckle_angle = np.degrees(np.arctan2(p17[1] - p5[1],  p17[0] - p5[0]))

    roll = knuckle_angle - up_angle + 90.0
    roll = (roll + 180.0) % 360.0 - 180.0
    return float(roll)


# ══════════════════════════════════════════════════════════════════════════════
# FLIGHT CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class DroneGestureController:
    def __init__(self, smoothing: float = 0.2, deadzone: int = 40,
                 calibration_frames: int = 30):
        self.alpha    = smoothing
        self.deadzone = deadzone
        self.smooth_roll     = 1500
        self.smooth_pitch    = 1500
        self.smooth_throttle = 1000
        self.smooth_yaw      = 1500
        self.cam_min = 0.25
        self.cam_max = 0.75

        # Yaw auto-calibration
        self._cal_frames  = calibration_frames
        self._cal_samples = []
        self._yaw_neutral = None

    def _dz(self, v: int, center: int = 1500) -> int:
        return center if abs(v - center) < self.deadzone else v

    def _ema(self, raw: int, prev: int) -> int:
        return int(self.alpha * raw + (1.0 - self.alpha) * prev)

    def recalibrate_yaw(self) -> None:
        self._cal_samples = []
        self._yaw_neutral = None
        print("[FlightCtrl] Yaw recalibration started — hold FIVE flat.")

    def process_hand(self,
                     gesture: str,
                     keypoints: List[Tuple[float, float]],
                     depth_mm: float) -> tuple:
        """
        gesture   : from recognize_gesture()
        keypoints : 21 full-frame normalised (x, y) landmarks
        depth_mm  : stereo Z of the wrist in millimetres
        Returns   : (roll, pitch, throttle, yaw) each int in [1000, 2000]
        """
        if gesture in ("FIST", "PEACE"):
            self.smooth_roll = self.smooth_pitch = 1500
            self.smooth_throttle = 1000
            self.smooth_yaw = 1500
            return 1500, 1500, 1000, 1500

        if gesture == "FIVE":
            # Roll  — kp[9] X
            raw_roll = self._dz(int(np.interp(
                keypoints[9][0], [self.cam_min, self.cam_max], [1000, 2000])))

            # Pitch — kp[9] Y (inverted: high on screen = forward)
            raw_pitch = self._dz(int(np.interp(
                keypoints[9][1], [self.cam_min, self.cam_max], [2000, 1000])))

            # Throttle — real stereo Z in mm
            raw_throttle = int(np.interp(
                depth_mm, [THROTTLE_NEAR_MM, THROTTLE_FAR_MM], [2000, 1000]))

            # Yaw — wrist roll relative to calibrated neutral
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

            self.smooth_roll     = self._ema(raw_roll,     self.smooth_roll)
            self.smooth_pitch    = self._ema(raw_pitch,    self.smooth_pitch)
            self.smooth_throttle = self._ema(raw_throttle, self.smooth_throttle)
            self.smooth_yaw      = self._ema(raw_yaw,      self.smooth_yaw)

            return (self.smooth_roll, self.smooth_pitch,
                    self.smooth_throttle, self.smooth_yaw)

        return 1500, 1500, 1000, 1500


# ══════════════════════════════════════════════════════════════════════════════
# SERIAL OUTPUT  (Arduino)
# ══════════════════════════════════════════════════════════════════════════════

class ArduinoSerial:
    def __init__(self, port: Optional[str] = None, baud: int = 115200):
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
        print(f"[Arduino] Connected on {port} @ {baud} baud")

    def send(self, roll: int, pitch: int, throttle: int, yaw: int) -> None:
        import serial
        r, p, t, y = [max(1000, min(2000, v)) for v in (roll, pitch, throttle, yaw)]
        try:
            self._ser.write(f"R:{r} P:{p} T:{t} Y:{y}\n".encode())
        except serial.SerialException as e:
            print(f"[Arduino] Write error: {e}")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[Arduino] Port closed.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS DETECTIONS HOST NODE
# ══════════════════════════════════════════════════════════════════════════════

class ProcessDetections(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.detections_input   = self.createInput()
        self.config_output      = self.createOutput()
        self.num_configs_output = self.createOutput()
        self.padding   = PADDING
        self._target_w = None
        self._target_h = None

    def build(self, detections_input, padding, target_size):
        self.padding   = padding
        self._target_w = target_size[0]
        self._target_h = target_size[1]
        self.link_args(detections_input)
        return self

    def process(self, img_detections: dai.Buffer) -> None:
        assert isinstance(img_detections, ImgDetectionsExtended)
        detections = img_detections.detections
        num_msg = dai.Buffer(len(detections))
        num_msg.setTimestamp(img_detections.getTimestamp())
        num_msg.setSequenceNum(img_detections.getSequenceNum())
        self.num_configs_output.send(num_msg)
        for det in detections:
            det: ImgDetectionExtended
            rect = det.rotated_rect
            new_rect = dai.RotatedRect()
            new_rect.center.x    = rect.center.x
            new_rect.center.y    = rect.center.y
            new_rect.size.width  = rect.size.width  + self.padding * 2
            new_rect.size.height = rect.size.height + self.padding * 2
            new_rect.angle       = 0
            cfg = dai.ImageManipConfig()
            cfg.addCropRotatedRect(new_rect, normalizedCoords=True)
            cfg.setOutputSize(self._target_w, self._target_h,
                              dai.ImageManipConfig.ResizeMode.STRETCH)
            cfg.setReusePreviousImage(False)
            cfg.setTimestamp(img_detections.getTimestamp())
            cfg.setSequenceNum(img_detections.getSequenceNum())
            self.config_output.send(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL ROI HELPER
# ══════════════════════════════════════════════════════════════════════════════

def push_spatial_roi(spatial_cfg_queue, wrist_x: float, wrist_y: float,
                     roi_half: float = 0.04) -> None:
    """Update the SpatialLocationCalculator ROI to track the wrist each frame."""
    x1 = max(wrist_x - roi_half, 0.0)
    y1 = max(wrist_y - roi_half, 0.0)
    x2 = min(wrist_x + roi_half, 1.0)
    y2 = min(wrist_y + roi_half, 1.0)
    cfg  = dai.SpatialLocationCalculatorConfig()
    data = dai.SpatialLocationCalculatorConfigData()
    data.roi                            = dai.Rect(dai.Point2f(x1, y1),
                                                   dai.Point2f(x2, y2))
    data.calculationAlgorithm           = \
        dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    data.depthThresholds.lowerThreshold = 100
    data.depthThresholds.upperThreshold = 10000
    cfg.addROI(data)
    spatial_cfg_queue.send(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_bbox(kpts):
    xs = [k[0] for k in kpts]; ys = [k[1] for k in kpts]; m = 0.02
    return (max(min(xs)-m,0.0), max(min(ys)-m,0.0),
            min(max(xs)+m,1.0), min(max(ys)+m,1.0))

def draw_hand(frame: np.ndarray, keypoints_norm: List, label: str = "") -> None:
    h, w = frame.shape[:2]
    for (i, j) in HAND_CONNECTIONS:
        if i < len(keypoints_norm) and j < len(keypoints_norm):
            cv2.line(frame,
                     (int(keypoints_norm[i][0]*w), int(keypoints_norm[i][1]*h)),
                     (int(keypoints_norm[j][0]*w), int(keypoints_norm[j][1]*h)),
                     COLOR_BONE, 2, cv2.LINE_AA)
    for (x_n, y_n) in keypoints_norm:
        cx, cy = int(x_n*w), int(y_n*h)
        cv2.circle(frame, (cx, cy), 5, COLOR_JOINT, -1)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1)
    x1n, y1n, x2n, y2n = keypoints_bbox(keypoints_norm)
    x1, y1 = int(x1n*w), int(y1n*h)
    x2, y2 = int(x2n*w), int(y2n*h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
    if label:
        cv2.putText(frame, label, (x1, max(y1-8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# ARG PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="OAK-D Pro hand pose + drone control — standalone")
    p.add_argument("-d",   "--device",    default=None)
    p.add_argument("-fps", "--fps_limit", default=None, type=int)
    p.add_argument("-p",   "--port",      default=None,
                   help="Arduino serial port e.g. COM3")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    device   = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    fps      = args.fps_limit or (30 if platform == "RVC4" else 15)
    print(f"Connected — Platform: {platform}  |  FPS: {fps}")

    frame_type = (dai.ImgFrame.Type.BGR888p if platform == "RVC2"
                  else dai.ImgFrame.Type.BGR888i)

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading models …")
    det_desc          = dai.NNModelDescription(PALM_MODEL_SLUG)
    det_desc.platform = platform
    det_archive       = dai.NNArchive(dai.getModelFromZoo(det_desc))
    pose_desc          = dai.NNModelDescription(HAND_MODEL_SLUG)
    pose_desc.platform = platform
    pose_archive       = dai.NNArchive(dai.getModelFromZoo(pose_desc))
    print("Models ready.")

    # ── Flight controller + Arduino ───────────────────────────────────────────
    flight_ctrl = DroneGestureController(smoothing=0.2, deadzone=40,
                                         calibration_frames=30)
    arduino = None
    if args.port:
        arduino = ArduinoSerial(port=args.port)
    else:
        print("[main] No --port — dry-run mode (no serial output)")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = dai.Pipeline(device)

    # RGB camera
    cam     = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((768, 768), frame_type, fps=fps)

    # Stereo depth — mono left/right -> StereoDepth -> SpatialLocationCalculator
    mono_left  = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(fps)
    mono_right.setFps(fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(),
                         mono_left.getResolutionHeight())
    stereo.setSubpixel(False)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial_calc.inputConfig.setWaitForMessage(False)
    initial_cfg = dai.SpatialLocationCalculatorConfigData()
    initial_cfg.roi                             = dai.Rect(dai.Point2f(0.45, 0.45),
                                                           dai.Point2f(0.55, 0.55))
    initial_cfg.calculationAlgorithm           = \
        dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    initial_cfg.depthThresholds.lowerThreshold = 100
    initial_cfg.depthThresholds.upperThreshold = 10000
    spatial_calc.initialConfig.addROI(initial_cfg)
    stereo.depth.link(spatial_calc.inputDepth)

    # Palm detector
    resize = pipeline.create(dai.node.ImageManip)
    resize.setMaxOutputFrameSize(
        det_archive.getInputWidth() * det_archive.getInputHeight() * 3)
    resize.initialConfig.setOutputSize(
        det_archive.getInputWidth(), det_archive.getInputHeight(),
        mode=dai.ImageManipConfig.ResizeMode.STRETCH)
    resize.initialConfig.setFrameType(frame_type)
    cam_out.link(resize.inputImage)

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize.out, det_archive)

    proc = pipeline.create(ProcessDetections).build(
        detections_input=det_nn.out,
        padding=PADDING,
        target_size=(pose_archive.getInputWidth(), pose_archive.getInputHeight()))

    script = pipeline.create(dai.node.Script)
    script.setScript(SCRIPT_CODE)
    script.inputs["frame_input"].setMaxSize(30)
    script.inputs["config_input"].setMaxSize(30)
    script.inputs["num_configs_input"].setMaxSize(30)
    det_nn.passthrough.link(script.inputs["frame_input"])
    proc.config_output.link(script.inputs["config_input"])
    proc.num_configs_output.link(script.inputs["num_configs_input"])

    pose_manip = pipeline.create(dai.node.ImageManip)
    pose_manip.initialConfig.setOutputSize(
        pose_archive.getInputWidth(), pose_archive.getInputHeight())
    pose_manip.inputConfig.setMaxSize(30)
    pose_manip.inputImage.setMaxSize(30)
    pose_manip.setNumFramesPool(30)
    pose_manip.inputConfig.setWaitForMessage(True)
    script.outputs["output_config"].link(pose_manip.inputConfig)
    script.outputs["output_frame"].link(pose_manip.inputImage)

    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        pose_manip.out, pose_archive)

    # All queues BEFORE pipeline.start()
    q_video       = cam_out.createOutputQueue(maxSize=4,  blocking=False)
    q_dets        = det_nn.out.createOutputQueue(maxSize=8, blocking=False)
    q_kp          = pose_nn.getOutput(0).createOutputQueue(maxSize=8, blocking=False)
    q_conf        = pose_nn.getOutput(1).createOutputQueue(maxSize=8, blocking=False)
    q_hand        = pose_nn.getOutput(2).createOutputQueue(maxSize=8, blocking=False)
    q_spatial     = spatial_calc.out.createOutputQueue(maxSize=8, blocking=False)
    q_spatial_cfg = spatial_calc.inputConfig.createInputQueue()

    pipeline.start()
    print("Running — FIVE=fly  FIST/PEACE=stop  'q'=quit  'r'=recalibrate yaw")

    # ── State ─────────────────────────────────────────────────────────────────
    last_hands:      List  = []
    det_data_by_seq: dict  = {}
    pose_buffer:     dict  = defaultdict(dict)
    hands_by_seq:    dict  = defaultdict(list)
    last_depth_mm:   float = float((THROTTLE_NEAR_MM + THROTTLE_FAR_MM) / 2)

    try:
        while pipeline.isRunning():
            in_video = q_video.tryGet()

            # ── Palm detections ───────────────────────────────────────────────
            in_dets = q_dets.tryGet()
            if in_dets is not None:
                try:
                    seq = in_dets.getSequenceNum()
                    if isinstance(in_dets, ImgDetectionsExtended) and in_dets.detections:
                        rects = []
                        for d in in_dets.detections:
                            cx = d.rotated_rect.center.x
                            cy = d.rotated_rect.center.y
                            rw = d.rotated_rect.size.width
                            rh = d.rotated_rect.size.height
                            hw = rw / 2 + PADDING
                            hh = rh / 2 + PADDING
                            rects.append((max(cx-hw,0.0), max(cy-hh,0.0),
                                          min(cx+hw,1.0), min(cy+hh,1.0)))
                        det_data_by_seq[seq] = {"rects": rects}
                    else:
                        det_data_by_seq[seq] = {"rects": []}
                        last_hands = []
                except Exception:
                    pass

            # ── Stereo depth ──────────────────────────────────────────────────
            spatial_msg = q_spatial.tryGet()
            if spatial_msg is not None:
                try:
                    locs = spatial_msg.getSpatialLocations()
                    if locs:
                        z = locs[0].spatialCoordinates.z
                        if 50 < z < 15000:
                            last_depth_mm = z
                except Exception:
                    pass

            # ── Landmark messages ─────────────────────────────────────────────
            for slot, q in ((0, q_kp), (1, q_conf), (2, q_hand)):
                msg = q.tryGet()
                if msg is not None:
                    pose_buffer[msg.getSequenceNum()][slot] = msg

            # ── Assemble complete triples ─────────────────────────────────────
            for seq in sorted(s for s, v in pose_buffer.items() if len(v) == 3):
                entry = pose_buffer.pop(seq)
                try:
                    if entry[1].prediction < CONFIDENCE_THRESHOLD:
                        continue
                    handness_label = "Right" if entry[2].prediction >= 0.5 else "Left"
                    kpts_local = [(kp.x, kp.y) for kp in entry[0].keypoints]
                    data     = det_data_by_seq.get(seq, {"rects": []})
                    hand_idx = len(hands_by_seq[seq])
                    det_rect = data["rects"][hand_idx] if hand_idx < len(data["rects"]) else None
                    if det_rect is not None:
                        xmin, ymin, xmax, ymax = det_rect
                        sx, sy = xmax - xmin, ymax - ymin
                        kpts = [(min(max(xmin + sx*kp[0], 0.0), 1.0),
                                 min(max(ymin + sy*kp[1], 0.0), 1.0))
                                for kp in kpts_local]
                    else:
                        kpts = kpts_local
                    gesture = recognize_gesture(kpts)
                    label   = f"{handness_label} | {gesture}" if gesture else handness_label
                    hands_by_seq[seq].append({"keypoints": kpts, "label": label})
                except Exception:
                    pass

            # ── Commit assembled frames ───────────────────────────────────────
            for seq in sorted(s for s in hands_by_seq if s not in pose_buffer):
                assembled = hands_by_seq.pop(seq)
                if assembled:
                    last_hands = assembled

            # ── Prune stale buffers ───────────────────────────────────────────
            for buf in (pose_buffer, det_data_by_seq, hands_by_seq):
                if len(buf) > 30:
                    for old in sorted(buf.keys())[:-15]:
                        buf.pop(old, None)

            # ── Flight control + draw + display ───────────────────────────────
            if in_video is not None:
                frame = in_video.getCvFrame()
                roll, pitch, throttle, yaw = 1500, 1500, 1000, 1500

                if last_hands:
                    primary   = last_hands[0]
                    label_str = primary["label"]
                    gesture   = label_str.split("|")[-1].strip() if "|" in label_str else ""
                    kpts      = primary["keypoints"]

                    # Track wrist with stereo ROI
                    push_spatial_roi(q_spatial_cfg, kpts[0][0], kpts[0][1])

                    roll, pitch, throttle, yaw = flight_ctrl.process_hand(
                        gesture, kpts, last_depth_mm)

                    if arduino is not None:
                        arduino.send(roll, pitch, throttle, yaw)

                for hand in last_hands:
                    draw_hand(frame, hand["keypoints"], label=hand["label"])

                # HUD
                cv2.putText(frame, f"Hands: {len(last_hands)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame,
                            f"CMD  R:{roll}  P:{pitch}  T:{throttle}  Y:{yaw}",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if last_hands:
                    raw_a    = knuckle_yaw_angle(last_hands[0]["keypoints"])
                    neutral  = flight_ctrl._yaw_neutral
                    cal_left = max(0, flight_ctrl._cal_frames - len(flight_ctrl._cal_samples))
                    if neutral is None:
                        yaw_str = f"Yaw: calibrating... ({cal_left} frames left)"
                    else:
                        dev = (raw_a - neutral + 180.0) % 360.0 - 180.0
                        dz  = "DZ" if abs(dev) < 2.0 else f"{dev:+.1f}deg"
                        yaw_str = f"Yaw: {dz}  (neutral={neutral:.1f}  range=1200-1800)"
                    cv2.putText(frame,
                                f"Depth: {last_depth_mm:.0f}mm"
                                f"  ({THROTTLE_FAR_MM}mm=T:1000"
                                f"  {THROTTLE_NEAR_MM}mm=T:2000)"
                                f"  |  {yaw_str}",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "q=quit  r=recal yaw", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow("Hand Pose — OAK-D Pro", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit.")
                pipeline.stop()
                break
            elif key == ord("r"):
                flight_ctrl.recalibrate_yaw()

    finally:
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()
        print("Done.")


if __name__ == "__main__":
    main()

