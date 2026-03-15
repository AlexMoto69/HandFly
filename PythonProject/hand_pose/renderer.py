"""
renderer.py
-----------
Everything that touches OpenCV:
  - draw_hand()        : draws skeleton, tight keypoint bbox, and label
  - run_display_loop() : main host loop — reads queues, updates stereo ROI,
                         assembles hand results, runs flight control,
                         sends to Arduino, shows window
"""
from collections import defaultdict
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import depthai as dai
from depthai_nodes import ImgDetectionsExtended

from .config import (
    PADDING, CONFIDENCE_THRESHOLD,
    HAND_CONNECTIONS,
    COLOR_JOINT, COLOR_BONE, COLOR_BOX, COLOR_TEXT,
    THROTTLE_NEAR_MM, THROTTLE_FAR_MM,
)
from .gesture import recognize_gesture
from .flight_control import DroneGestureController, knuckle_yaw_angle
from .serial_output import ArduinoSerial


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_bbox(keypoints_norm: List) -> tuple:
    """Tight bounding box from keypoint extents with small margin."""
    xs = [kp[0] for kp in keypoints_norm]
    ys = [kp[1] for kp in keypoints_norm]
    m  = 0.02
    return (max(min(xs) - m, 0.0), max(min(ys) - m, 0.0),
            min(max(xs) + m, 1.0), min(max(ys) + m, 1.0))


def draw_hand(frame: np.ndarray,
              keypoints_norm: List,
              label: str = "") -> None:
    """Draw skeleton, joints, tight keypoint bbox and label."""
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

    x1n, y1n, x2n, y2n = keypoints_bbox(keypoints_norm)
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
    if label:
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)


def _push_spatial_roi(spatial_cfg_queue, wrist_x: float, wrist_y: float,
                      roi_half: float = 0.04) -> None:
    """Send updated ROI to SpatialLocationCalculator to track wrist each frame."""
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
# MAIN DISPLAY LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_display_loop(pipeline: dai.Pipeline,
                     queues: Dict[str, Any],
                     flight_ctrl: DroneGestureController,
                     arduino: Optional[ArduinoSerial] = None) -> None:
    """
    Reads all output queues, tracks the wrist ROI for stereo depth,
    assembles hand results, runs flight control, optionally sends to Arduino,
    and shows the OpenCV window until the user presses 'q'.
    """
    q_video       = queues["video"]
    q_dets        = queues["dets"]
    q_kp          = queues["kp"]
    q_conf        = queues["conf"]
    q_hand        = queues["hand"]
    q_spatial     = queues["spatial"]
    q_spatial_cfg = queues["spatial_cfg"]

    last_hands:     List  = []
    det_data_by_seq: dict = {}
    pose_buffer:     dict = defaultdict(dict)
    hands_by_seq:    dict = defaultdict(list)
    last_depth_mm:   float = float((THROTTLE_NEAR_MM + THROTTLE_FAR_MM) / 2)

    while pipeline.isRunning():
        in_video = q_video.tryGet()

        # ── Palm detections ───────────────────────────────────────────────────
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
                        rects.append((max(cx - hw, 0.0), max(cy - hh, 0.0),
                                      min(cx + hw, 1.0), min(cy + hh, 1.0)))
                    det_data_by_seq[seq] = {"rects": rects}
                else:
                    det_data_by_seq[seq] = {"rects": []}
                    last_hands = []
            except Exception:
                pass

        # ── Stereo depth — read latest Z ──────────────────────────────────────
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

        # ── Collect per-head landmark messages ────────────────────────────────
        for slot, q in ((0, q_kp), (1, q_conf), (2, q_hand)):
            msg = q.tryGet()
            if msg is not None:
                pose_buffer[msg.getSequenceNum()][slot] = msg

        # ── Assemble complete triples ─────────────────────────────────────────
        for seq in sorted(s for s, v in pose_buffer.items() if len(v) == 3):
            entry = pose_buffer.pop(seq)
            try:
                if entry[1].prediction < CONFIDENCE_THRESHOLD:
                    continue
                handness_label = "Right" if entry[2].prediction >= 0.5 else "Left"
                kpts_local     = [(kp.x, kp.y) for kp in entry[0].keypoints]
                data     = det_data_by_seq.get(seq, {"rects": []})
                hand_idx = len(hands_by_seq[seq])
                det_rect = data["rects"][hand_idx] if hand_idx < len(data["rects"]) else None
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
                hands_by_seq[seq].append({"keypoints": kpts, "label": label})
            except Exception:
                pass

        # ── Commit fully-assembled frames ─────────────────────────────────────
        for seq in sorted(s for s in hands_by_seq if s not in pose_buffer):
            assembled = hands_by_seq.pop(seq)
            if assembled:
                last_hands = assembled

        # ── Prune stale buffers ───────────────────────────────────────────────
        for buf in (pose_buffer, det_data_by_seq, hands_by_seq):
            if len(buf) > 30:
                for old in sorted(buf.keys())[:-15]:
                    buf.pop(old, None)

        # ── Flight control + draw + display ───────────────────────────────────
        if in_video is not None:
            frame = in_video.getCvFrame()
            roll, pitch, throttle, yaw = 1500, 1500, 1000, 1500

            if last_hands:
                primary   = last_hands[0]
                label_str = primary["label"]
                gesture   = label_str.split("|")[-1].strip() if "|" in label_str else ""
                kpts      = primary["keypoints"]

                # Track wrist with stereo ROI
                _push_spatial_roi(q_spatial_cfg, kpts[0][0], kpts[0][1])

                roll, pitch, throttle, yaw = flight_ctrl.process_hand(
                    gesture, kpts, last_depth_mm)

            # Always send current command values to Arduino when available
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
                kpts  = last_hands[0]["keypoints"]
                raw_a = knuckle_yaw_angle(kpts)
                neutral = flight_ctrl._yaw_neutral
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
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("Hand Pose — OAK-D", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit.")
            pipeline.stop()
            break
        elif key == ord("r"):
            flight_ctrl.recalibrate_yaw()

    cv2.destroyAllWindows()
