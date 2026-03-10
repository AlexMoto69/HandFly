"""
hand_pose_standalone.py
-----------------------
100% standalone hand-pose estimation + drone flight control with OAK-D (RVC2/RVC4).
No dependency on the hand_pose/ package — everything is inlined in this single file.

Requirements (pip install):
    depthai>=3.0.0
    depthai-nodes>=0.3.4
    opencv-python
    numpy
    pyserial          ← only needed if using --port

Models are downloaded automatically from the Luxonis Model Zoo on first run
and cached locally in .depthai_cached_models/.

Run:
    python hand_pose_standalone.py
    python hand_pose_standalone.py --fps_limit 15
    python hand_pose_standalone.py --device <DeviceID-or-IP>
    python hand_pose_standalone.py --port COM3          # with Arduino
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

# Padding added around the raw palm bbox before sending to the landmark model.
# 0.2 = covers the full open hand including all fingertips.
PADDING              = 0.2
CONFIDENCE_THRESHOLD = 0.5

# MediaPipe 21-landmark skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm cross
]

# BGR colours
COLOR_JOINT = (0, 255, 0)
COLOR_BONE  = (255, 200, 0)
COLOR_BOX   = (0, 200, 255)
COLOR_TEXT  = (255, 255, 255)

# Throttle range — raw palm bbox width (before padding)
# ~0.32 = hand at ~20 cm → T:2000   ~0.10 = hand at ~70 cm → T:1000
THROTTLE_NEAR = 0.32
THROTTLE_FAR  = 0.10

# Model slugs
PALM_MODEL_SLUG = "luxonis/mediapipe-palm-detection:192x192"
HAND_MODEL_SLUG = "luxonis/mediapipe-hand-landmarker:224x224"

# On-device Script — fans one frame per detected hand to the landmark model
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
# FLIGHT CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class DroneGestureController:
    def __init__(self, smoothing: float = 0.2, deadzone: int = 40):
        self.alpha    = smoothing
        self.deadzone = deadzone
        self.smooth_roll     = 1500
        self.smooth_pitch    = 1500
        self.smooth_throttle = 1000
        self.cam_min = 0.25
        self.cam_max = 0.75

    def _dz(self, v: int, center: int = 1500) -> int:
        return center if abs(v - center) < self.deadzone else v

    def _ema(self, raw: int, prev: int) -> int:
        return int(self.alpha * raw + (1.0 - self.alpha) * prev)

    def process_hand(self, gesture: str,
                     ctrl_x: float, ctrl_y: float,
                     raw_width: float) -> tuple:
        """
        gesture   : recognized gesture string
        ctrl_x/y  : kp[9] (middle finger MCP) in full-frame normalised coords
        raw_width : raw palm bbox width (before padding) — depth proxy
        Returns   : (roll, pitch, throttle, yaw) each in [1000, 2000]
        """
        if gesture in ("FIST", "PEACE"):
            self.smooth_roll = self.smooth_pitch = 1500
            self.smooth_throttle = 1000
            return 1500, 1500, 1000, 1500

        if gesture == "FIVE":
            raw_roll     = self._dz(int(np.interp(ctrl_x,
                                [self.cam_min, self.cam_max], [1000, 2000])))
            raw_pitch    = self._dz(int(np.interp(ctrl_y,
                                [self.cam_min, self.cam_max], [2000, 1000])))
            raw_throttle = int(np.interp(raw_width,
                                [THROTTLE_FAR, THROTTLE_NEAR], [1000, 2000]))
            self.smooth_roll     = self._ema(raw_roll,     self.smooth_roll)
            self.smooth_pitch    = self._ema(raw_pitch,    self.smooth_pitch)
            self.smooth_throttle = self._ema(raw_throttle, self.smooth_throttle)
            return self.smooth_roll, self.smooth_pitch, self.smooth_throttle, 1500

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
                    print(f"[ArduinoSerial] Auto-detected: {port} ({p.description})")
                    break
        if port is None:
            raise RuntimeError("No Arduino found. Use --port COM<X> to specify one.")
        self._ser = serial.Serial(port, baud, timeout=1)
        print(f"[ArduinoSerial] Connected on {port} @ {baud} baud")

    def send(self, roll: int, pitch: int, throttle: int, yaw: int) -> None:
        import serial
        r,p,t,y = [max(1000, min(2000, v)) for v in (roll, pitch, throttle, yaw)]
        try:
            self._ser.write(f"R:{r} P:{p} T:{t} Y:{y}\n".encode())
        except serial.SerialException as e:
            print(f"[ArduinoSerial] Write error: {e}")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[ArduinoSerial] Port closed.")


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
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_bbox(kpts):
    """Tight bounding box from actual keypoint extents."""
    xs = [k[0] for k in kpts]
    ys = [k[1] for k in kpts]
    m  = 0.02
    return (max(min(xs)-m, 0.0), max(min(ys)-m, 0.0),
            min(max(xs)+m, 1.0), min(max(ys)+m, 1.0))

def draw_hand(frame: np.ndarray, keypoints_norm: List, label: str = "") -> None:
    """Draw skeleton, joints, tight keypoint bbox and label."""
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

    # Box from keypoint extents — covers full hand not just palm
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
        description="OAK-D hand pose estimation + drone control — standalone")
    p.add_argument("-d",   "--device",    default=None,
                   help="DeviceID or IP of the OAK camera (auto-detect if omitted)")
    p.add_argument("-fps", "--fps_limit", default=None, type=int,
                   help="FPS cap (default: 15 for RVC2, 30 for RVC4)")
    p.add_argument("-p",   "--port",      default=None,
                   help="Arduino serial port e.g. COM3 or /dev/ttyUSB0")
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

    # ── Flight controller ─────────────────────────────────────────────────────
    flight_ctrl = DroneGestureController(smoothing=0.2, deadzone=40)

    # ── Arduino (optional) ────────────────────────────────────────────────────
    arduino = None
    if args.port:
        arduino = ArduinoSerial(port=args.port)
    else:
        print("[main] No --port — dry-run mode (no serial output)")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = dai.Pipeline(device)

    cam     = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((768, 768), frame_type, fps=fps)

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
    q_video = cam_out.createOutputQueue(maxSize=4,  blocking=False)
    q_dets  = det_nn.out.createOutputQueue(maxSize=8, blocking=False)
    q_kp    = pose_nn.getOutput(0).createOutputQueue(maxSize=8, blocking=False)
    q_conf  = pose_nn.getOutput(1).createOutputQueue(maxSize=8, blocking=False)
    q_hand  = pose_nn.getOutput(2).createOutputQueue(maxSize=8, blocking=False)

    pipeline.start()
    print("Running — FIVE=fly  FIST/PEACE=stop  'q'=quit")

    # ── State ─────────────────────────────────────────────────────────────────
    last_hands      = []
    det_data_by_seq: dict = {}   # seq → {"rects": [...], "raw_widths": [...]}
    pose_buffer:     dict = defaultdict(dict)
    hands_by_seq:    dict = defaultdict(list)

    try:
        while pipeline.isRunning():
            in_video = q_video.tryGet()

            # ── Palm detections ───────────────────────────────────────────────
            in_dets = q_dets.tryGet()
            if in_dets is not None:
                try:
                    seq = in_dets.getSequenceNum()
                    if isinstance(in_dets, ImgDetectionsExtended) and in_dets.detections:
                        rects, raw_widths = [], []
                        for d in in_dets.detections:
                            cx = d.rotated_rect.center.x
                            cy = d.rotated_rect.center.y
                            rw = d.rotated_rect.size.width   # raw — for throttle
                            rh = d.rotated_rect.size.height
                            hw = rw / 2 + PADDING
                            hh = rh / 2 + PADDING
                            rects.append((max(cx-hw, 0.0), max(cy-hh, 0.0),
                                          min(cx+hw, 1.0), min(cy+hh, 1.0)))
                            raw_widths.append(rw)
                        det_data_by_seq[seq] = {"rects": rects, "raw_widths": raw_widths}
                    else:
                        det_data_by_seq[seq] = {"rects": [], "raw_widths": []}
                        last_hands = []
                except Exception:
                    pass

            # ── Collect landmark head messages ────────────────────────────────
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
                    data      = det_data_by_seq.get(seq, {"rects": [], "raw_widths": []})
                    hand_idx  = len(hands_by_seq[seq])
                    det_rect  = data["rects"][hand_idx]      if hand_idx < len(data["rects"])      else None
                    raw_width = data["raw_widths"][hand_idx] if hand_idx < len(data["raw_widths"]) else 0.15
                    # Unproject keypoints from crop-local → full frame
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
                    hands_by_seq[seq].append({"keypoints": kpts,
                                              "label":     label,
                                              "raw_width": raw_width})
                except Exception:
                    pass

            # ── Commit fully-assembled frames ─────────────────────────────────
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
                raw_width_display = 0.0

                if last_hands:
                    primary   = last_hands[0]
                    label_str = primary["label"]
                    gesture   = label_str.split("|")[-1].strip() if "|" in label_str else ""

                    # kp[9] = middle finger MCP — more central/stable than wrist
                    ctrl_x = primary["keypoints"][9][0]
                    ctrl_y = primary["keypoints"][9][1]

                    raw_width_display = primary["raw_width"]
                    roll, pitch, throttle, yaw = flight_ctrl.process_hand(
                        gesture, ctrl_x, ctrl_y, raw_width_display)

                    if arduino is not None:
                        arduino.send(roll, pitch, throttle, yaw)

                for hand in last_hands:
                    draw_hand(frame, hand["keypoints"], label=hand["label"])

                # HUD
                cv2.putText(frame, f"Hands: {len(last_hands)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"CMD  R:{roll}  P:{pitch}  T:{throttle}  Y:{yaw}",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if last_hands:
                    cv2.putText(frame,
                                f"Palm width: {raw_width_display:.3f}"
                                f"  (70cm~0.10  45cm~0.21  20cm~0.32)",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow("Hand Pose — OAK-D", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit.")
                pipeline.stop()
                break

    finally:
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()
        print("Done.")


if __name__ == "__main__":
    main()

