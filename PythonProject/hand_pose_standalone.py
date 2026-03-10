"""
hand_pose_standalone.py
-----------------------
100% standalone hand-pose estimation with an OAK-D (RVC2 / RVC4).
No dependency on oak-examples or any local utils/ folder.

Requirements (pip install):
    depthai>=3.0.0
    depthai-nodes>=0.3.4
    opencv-python
    numpy

Models are downloaded automatically from the Luxonis Model Zoo on first run
and cached locally in .depthai_cached_models/.

Run:
    python hand_pose_standalone.py
    python hand_pose_standalone.py --fps_limit 15
    python hand_pose_standalone.py --device <DeviceID-or-IP>
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import argparse
import textwrap

# ── third-party ────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import depthai as dai
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended
from depthai_nodes.node import ParsingNeuralNetwork

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PADDING            = 0.1
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

# ── On-device script (inlined as a string — no external file needed) ───────────
# Reads one frame + N crop-configs per detection frame and fans them out
# to the ImageManip node that feeds the landmark model.
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

# ── Model slugs (downloaded from Luxonis Model Zoo) ───────────────────────────
PALM_MODEL  = {
    "RVC2": "luxonis/mediapipe-palm-detection:192x192",
    "RVC4": "luxonis/mediapipe-palm-detection:192x192",
}
HAND_MODEL  = {
    "RVC2": "luxonis/mediapipe-hand-landmarker:224x224",
    "RVC4": "luxonis/mediapipe-hand-landmarker:224x224",
}


# ══════════════════════════════════════════════════════════════════════════════
# GESTURE RECOGNITION  (inlined from utils/gesture_recognition.py)
# ══════════════════════════════════════════════════════════════════════════════

def _dist(a, b):
    return np.linalg.norm(a - b)

def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def recognize_gesture(kpts: List[Tuple[float, float]]):
    kpts = np.array(kpts)
    d_3_5 = _dist(kpts[3], kpts[5])
    d_2_3 = _dist(kpts[2], kpts[3])
    a0 = _angle(kpts[0], kpts[1], kpts[2])
    a1 = _angle(kpts[1], kpts[2], kpts[3])
    a2 = _angle(kpts[2], kpts[3], kpts[4])

    thumb  = 1 if (a0 + a1 + a2 > 460 and d_3_5 / d_2_3 > 1.2) else 0

    def finger(tip, mid, base):
        if kpts[tip][1] < kpts[mid][1] < kpts[base][1]:  return 1
        if kpts[base][1] < kpts[tip][1]:                  return 0
        return -1

    index  = finger(8,  7,  6)
    middle = finger(12, 11, 10)
    ring   = finger(16, 15, 14)
    little = finger(20, 19, 18)

    combo = (thumb, index, middle, ring, little)
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
# PROCESS DETECTIONS HOST NODE  (inlined from utils/process.py)
# ══════════════════════════════════════════════════════════════════════════════

class ProcessDetections(dai.node.HostNode):
    """Converts ImgDetectionsExtended → per-hand ImageManipConfig crops."""

    def __init__(self):
        super().__init__()
        self.detections_input  = self.createInput()
        self.config_output     = self.createOutput()
        self.num_configs_output = self.createOutput()
        self.padding   = PADDING
        self._target_w = None
        self._target_h = None

    def build(self, detections_input: dai.Node.Output,
              padding: float, target_size: Tuple[int, int]) -> "ProcessDetections":
        self.padding   = padding
        self._target_w = target_size[0]
        self._target_h = target_size[1]
        self.link_args(detections_input)
        return self

    def process(self, img_detections: dai.Buffer) -> None:
        assert isinstance(img_detections, ImgDetectionsExtended)
        detections = img_detections.detections

        num_cfgs_msg = dai.Buffer(len(detections))
        num_cfgs_msg.setTimestamp(img_detections.getTimestamp())
        num_cfgs_msg.setSequenceNum(img_detections.getSequenceNum())
        self.num_configs_output.send(num_cfgs_msg)

        for det in detections:
            det: ImgDetectionExtended
            rect = det.rotated_rect

            new_rect = dai.RotatedRect()
            new_rect.center.x     = rect.center.x
            new_rect.center.y     = rect.center.y
            new_rect.size.width   = rect.size.width  + self.padding * 2
            new_rect.size.height  = rect.size.height + self.padding * 2
            new_rect.angle        = 0

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

def draw_hand(frame: np.ndarray, keypoints_norm,
              detection_rect=None, label: str = "") -> None:
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
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1)

    if detection_rect is not None:
        x1, y1 = int(detection_rect[0] * w), int(detection_rect[1] * h)
        x2, y2 = int(detection_rect[2] * w), int(detection_rect[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        if label:
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# ARG PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="OAK-D hand pose estimation — standalone")
    p.add_argument("-d",   "--device",    default=None,
                   help="DeviceID or IP of the OAK camera (auto-detect if omitted)")
    p.add_argument("-fps", "--fps_limit", default=None, type=int,
                   help="FPS cap (default: 15 for RVC2, 30 for RVC4)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    print(f"Connected — Platform: {platform}")

    fps = args.fps_limit or (30 if platform == "RVC4" else 15)
    print(f"FPS limit : {fps}")

    frame_type = (dai.ImgFrame.Type.BGR888p
                  if platform == "RVC2"
                  else dai.ImgFrame.Type.BGR888i)

    # ── Download / load models ────────────────────────────────────────────────
    print("Loading models from Model Zoo …")
    det_desc = dai.NNModelDescription(PALM_MODEL[platform])
    det_desc.platform = platform
    det_archive = dai.NNArchive(dai.getModelFromZoo(det_desc))

    pose_desc = dai.NNModelDescription(HAND_MODEL[platform])
    pose_desc.platform = platform
    pose_archive = dai.NNArchive(dai.getModelFromZoo(pose_desc))
    print("Models ready.")

    # ── Build pipeline ────────────────────────────────────────────────────────
    with dai.Pipeline(device) as pipeline:
        print("Building pipeline …")

        # Camera
        cam     = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput((768, 768), frame_type, fps=fps)

        # Resize → palm detector
        resize = pipeline.create(dai.node.ImageManip)
        resize.setMaxOutputFrameSize(
            det_archive.getInputWidth() * det_archive.getInputHeight() * 3)
        resize.initialConfig.setOutputSize(
            det_archive.getInputWidth(), det_archive.getInputHeight(),
            mode=dai.ImageManipConfig.ResizeMode.STRETCH)
        resize.initialConfig.setFrameType(frame_type)
        cam_out.link(resize.inputImage)

        # Stage 1 — palm detection
        det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
            resize.out, det_archive)

        # Detection → crop configs
        proc = pipeline.create(ProcessDetections).build(
            detections_input=det_nn.out,
            padding=PADDING,
            target_size=(pose_archive.getInputWidth(), pose_archive.getInputHeight()))

        # Script node — fans one frame per detected hand
        script = pipeline.create(dai.node.Script)
        script.setScript(SCRIPT_CODE)                  # inline — no file needed
        script.inputs["frame_input"].setMaxSize(30)
        script.inputs["config_input"].setMaxSize(30)
        script.inputs["num_configs_input"].setMaxSize(30)
        det_nn.passthrough.link(script.inputs["frame_input"])
        proc.config_output.link(script.inputs["config_input"])
        proc.num_configs_output.link(script.inputs["num_configs_input"])

        # Warp each hand crop
        pose_manip = pipeline.create(dai.node.ImageManip)
        pose_manip.initialConfig.setOutputSize(
            pose_archive.getInputWidth(), pose_archive.getInputHeight())
        pose_manip.inputConfig.setMaxSize(30)
        pose_manip.inputImage.setMaxSize(30)
        pose_manip.setNumFramesPool(30)
        pose_manip.inputConfig.setWaitForMessage(True)
        script.outputs["output_config"].link(pose_manip.inputConfig)
        script.outputs["output_frame"].link(pose_manip.inputImage)

        # Stage 2 — hand landmark model (3 heads)
        pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
            pose_manip.out, pose_archive)

        # ── Queues — ALL before pipeline.start() ─────────────────────────────
        q_video = cam_out.createOutputQueue(maxSize=4,  blocking=False)
        q_dets  = det_nn.out.createOutputQueue(maxSize=8, blocking=False)
        # head 0 = Keypoints, head 1 = presence score, head 2 = handedness
        q_kp    = pose_nn.getOutput(0).createOutputQueue(maxSize=8, blocking=False)
        q_conf  = pose_nn.getOutput(1).createOutputQueue(maxSize=8, blocking=False)
        q_hand  = pose_nn.getOutput(2).createOutputQueue(maxSize=8, blocking=False)

        print("Starting pipeline …")
        pipeline.start()
        print("Running — show your hand!  Press 'q' to quit.")

        # ── State ─────────────────────────────────────────────────────────────
        last_hands      = []
        det_rects_by_seq: dict = {}
        pose_buffer:      dict = defaultdict(dict)
        hands_by_seq:     dict = defaultdict(list)

        while pipeline.isRunning():
            in_video = q_video.tryGet()

            # Palm detections — store all bboxes keyed by sequence number
            in_dets = q_dets.tryGet()
            if in_dets is not None:
                try:
                    seq = in_dets.getSequenceNum()
                    if isinstance(in_dets, ImgDetectionsExtended) and in_dets.detections:
                        rects = []
                        for d in in_dets.detections:
                            cx = d.rotated_rect.center.x
                            cy = d.rotated_rect.center.y
                            hw = d.rotated_rect.size.width  / 2 + PADDING
                            hh = d.rotated_rect.size.height / 2 + PADDING
                            rects.append((max(cx - hw, 0.0), max(cy - hh, 0.0),
                                          min(cx + hw, 1.0), min(cy + hh, 1.0)))
                        det_rects_by_seq[seq] = rects
                    else:
                        det_rects_by_seq[seq] = []
                        last_hands = []
                except Exception:
                    pass

            # Collect per-head landmark messages
            for slot, q in ((0, q_kp), (1, q_conf), (2, q_hand)):
                msg = q.tryGet()
                if msg is not None:
                    pose_buffer[msg.getSequenceNum()][slot] = msg

            # Assemble complete triples
            for seq in sorted(s for s, v in pose_buffer.items() if len(v) == 3):
                entry = pose_buffer.pop(seq)
                try:
                    confidence = entry[1].prediction
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    handness_label = "Right" if entry[2].prediction >= 0.5 else "Left"
                    kpts_local     = [(kp.x, kp.y) for kp in entry[0].keypoints]

                    rects    = det_rects_by_seq.get(seq, [])
                    hand_idx = len(hands_by_seq[seq])
                    det_rect = rects[hand_idx] if hand_idx < len(rects) else None

                    if det_rect is not None:
                        xmin, ymin, xmax, ymax = det_rect
                        sx, sy = xmax - xmin, ymax - ymin
                        kpts = [(min(max(xmin + sx * kp[0], 0.0), 1.0),
                                 min(max(ymin + sy * kp[1], 0.0), 1.0))
                                for kp in kpts_local]
                    else:
                        kpts = kpts_local

                    gesture = recognize_gesture(kpts)
                    label   = f"{handness_label} | {gesture}" if gesture else handness_label
                    hands_by_seq[seq].append({"keypoints": kpts,
                                              "rect": det_rect, "label": label})
                except Exception:
                    pass

            # Commit fully-assembled frames
            for seq in sorted(s for s in hands_by_seq if s not in pose_buffer):
                assembled = hands_by_seq.pop(seq)
                if assembled:
                    last_hands = assembled

            # Prune stale buffers
            for buf in (pose_buffer, det_rects_by_seq, hands_by_seq):
                if len(buf) > 30:
                    for old in sorted(buf.keys())[:-15]:
                        buf.pop(old, None)

            # Draw & display
            if in_video is not None:
                frame = in_video.getCvFrame()
                for hand in last_hands:
                    draw_hand(frame, hand["keypoints"],
                              detection_rect=hand["rect"], label=hand["label"])
                cv2.putText(frame, f"Hands: {len(last_hands)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow("Hand Pose — OAK-D", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit.")
                break

        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()

