"""
renderer.py
-----------
Everything that touches OpenCV:
  - draw_hand()        : draws skeleton, joints, bbox from keypoint extents, and label
  - run_display_loop() : main host loop — reads queues, assembles results,
                         runs flight control, sends to Arduino, draws window

Key design decisions:
  - The padded crop rect (det_rect) is used ONLY for unprojecting keypoints back
    to full-frame normalised coords.
  - The bounding box drawn on screen is computed from the actual keypoint extents
    (min/max x,y across all 21 landmarks), so it covers the real hand shape.
  - The depth metric uses the RAW palm detection width (before padding) which
    gives a clean 0.05–0.35 range that actually changes with distance.
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
)
from .gesture import recognize_gesture
from .flight_control import DroneGestureController
from .serial_output import ArduinoSerial


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def keypoints_bbox(keypoints_norm: List) -> tuple:
    """
    Compute a tight bounding box around all 21 keypoints.
    Returns (xmin, ymin, xmax, ymax) in normalised coords with a small margin.
    """
    xs = [kp[0] for kp in keypoints_norm]
    ys = [kp[1] for kp in keypoints_norm]
    margin = 0.02
    return (max(min(xs) - margin, 0.0),
            max(min(ys) - margin, 0.0),
            min(max(xs) + margin, 1.0),
            min(max(ys) + margin, 1.0))


def draw_hand(frame: np.ndarray,
              keypoints_norm: List,
              label: str = "") -> None:
    """
    Draw the hand skeleton, joints, a tight bbox around keypoints, and label.

    Args:
        frame          : BGR OpenCV frame (modified in place).
        keypoints_norm : list of 21 (x, y) pairs normalised to [0, 1].
        label          : text shown above the bbox, e.g. "Right | PEACE".
    """
    h, w = frame.shape[:2]

    # Skeleton bones
    for (i, j) in HAND_CONNECTIONS:
        if i < len(keypoints_norm) and j < len(keypoints_norm):
            cv2.line(frame,
                     (int(keypoints_norm[i][0] * w), int(keypoints_norm[i][1] * h)),
                     (int(keypoints_norm[j][0] * w), int(keypoints_norm[j][1] * h)),
                     COLOR_BONE, 2, cv2.LINE_AA)

    # Joint dots
    for (x_n, y_n) in keypoints_norm:
        cx, cy = int(x_n * w), int(y_n * h)
        cv2.circle(frame, (cx, cy), 5, COLOR_JOINT, -1)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1)

    # Tight bounding box from actual keypoint extents
    x1n, y1n, x2n, y2n = keypoints_bbox(keypoints_norm)
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
    if label:
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_display_loop(pipeline: dai.Pipeline,
                     queues: Dict[str, Any],
                     flight_ctrl: DroneGestureController,
                     arduino: Optional[ArduinoSerial] = None) -> None:
    """
    Reads all output queues, assembles hand results per frame, runs the
    flight controller, optionally sends PPM commands to the Arduino, and
    shows the OpenCV window until the user presses 'q'.

    Args:
        pipeline     : running dai.Pipeline
        queues       : dict of output queues from build_pipeline()
        flight_ctrl  : DroneGestureController instance
        arduino      : optional ArduinoSerial — None = dry-run (no hardware)
    """
    q_video = queues["video"]
    q_dets  = queues["dets"]
    q_kp    = queues["kp"]
    q_conf  = queues["conf"]
    q_hand  = queues["hand"]

    last_hands:       List = []
    # Store both the padded crop rect AND the raw detection width per seq
    # raw_width is the palm bbox width BEFORE padding — used for depth/throttle
    det_data_by_seq:  dict = {}   # seq → {"rects": [...], "raw_widths": [...]}
    pose_buffer:      dict = defaultdict(dict)
    hands_by_seq:     dict = defaultdict(list)

    while pipeline.isRunning():
        in_video = q_video.tryGet()

        # ── Palm detections ───────────────────────────────────────────────────
        in_dets = q_dets.tryGet()
        if in_dets is not None:
            try:
                seq = in_dets.getSequenceNum()
                if isinstance(in_dets, ImgDetectionsExtended) and in_dets.detections:
                    rects      = []
                    raw_widths = []
                    for d in in_dets.detections:
                        cx = d.rotated_rect.center.x
                        cy = d.rotated_rect.center.y
                        rw = d.rotated_rect.size.width   # raw palm width — for depth
                        rh = d.rotated_rect.size.height
                        hw = rw / 2 + PADDING
                        hh = rh / 2 + PADDING
                        rects.append((max(cx - hw, 0.0), max(cy - hh, 0.0),
                                      min(cx + hw, 1.0), min(cy + hh, 1.0)))
                        raw_widths.append(rw)
                    det_data_by_seq[seq] = {"rects": rects, "raw_widths": raw_widths}
                else:
                    det_data_by_seq[seq] = {"rects": [], "raw_widths": []}
                    last_hands = []
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

                data     = det_data_by_seq.get(seq, {"rects": [], "raw_widths": []})
                hand_idx = len(hands_by_seq[seq])
                det_rect  = data["rects"][hand_idx]      if hand_idx < len(data["rects"])      else None
                raw_width = data["raw_widths"][hand_idx] if hand_idx < len(data["raw_widths"]) else 0.15

                # Unproject keypoints from crop-local coords back to full-frame
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
                hands_by_seq[seq].append({
                    "keypoints":  kpts,
                    "label":      label,
                    "raw_width":  raw_width,   # raw palm detection width for throttle
                })
            except Exception:
                pass

        # ── Commit fully-assembled frames to last_hands ───────────────────────
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

            roll, pitch, throttle, yaw = 1500, 1500, 1000, 1500  # safe defaults

            if last_hands:
                primary = last_hands[0]

                # Parse gesture from "Right | FIVE" style label
                label_str = primary["label"]
                gesture   = label_str.split("|")[-1].strip() if "|" in label_str else ""

                # Control point: middle finger base knuckle (kp9) — more central
                # and stable than wrist (kp0) which drifts with wrist rotation
                wrist_x = primary["keypoints"][9][0]
                wrist_y = primary["keypoints"][9][1]

                # Raw palm width — clean depth proxy, unaffected by padding
                # Confirmed range: ~0.05 (far ~1m) to ~0.35 (close ~20cm)
                raw_width = primary["raw_width"]

                # Compute filtered PPM commands
                roll, pitch, throttle, yaw = flight_ctrl.process_hand(
                    gesture, wrist_x, wrist_y, raw_width)

                # Send to Arduino if connected
                if arduino is not None:
                    arduino.send(roll, pitch, throttle, yaw)

            # Draw skeleton + tight keypoint bbox for every detected hand
            for hand in last_hands:
                draw_hand(frame, hand["keypoints"], label=hand["label"])

            # HUD — hand count
            cv2.putText(frame, f"Hands: {len(last_hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # HUD — live PPM readout
            cv2.putText(frame,
                        f"CMD  R:{roll}  P:{pitch}  T:{throttle}  Y:{yaw}",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # HUD — raw palm width for tuning THROTTLE_NEAR / THROTTLE_FAR
            if last_hands:
                rw = last_hands[0]["raw_width"]
                cv2.putText(frame,
                            f"Palm width: {rw:.3f}  (70cm~0.10  45cm~0.21  20cm~0.32)",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 0), 1, cv2.LINE_AA)

            # HUD — quit hint
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Hand Pose — OAK-D", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            pipeline.stop()
            break

    cv2.destroyAllWindows()

