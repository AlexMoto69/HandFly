"""
renderer.py
-----------
Everything that touches OpenCV:
  - draw_hand()        : draws skeleton, joints, bbox, and label onto a frame
  - run_display_loop() : main host loop — reads queues, assembles results,
                         draws, and shows the window
"""
from collections import defaultdict
from typing import Dict, List, Any

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


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_hand(frame: np.ndarray,
              keypoints_norm: List,
              detection_rect=None,
              label: str = "") -> None:
    """
    Draw the hand skeleton, joints, bounding box, and gesture label.

    Args:
        frame          : BGR OpenCV frame (modified in place).
        keypoints_norm : list of 21 (x, y) pairs normalised to [0, 1].
        detection_rect : optional (xmin, ymin, xmax, ymax) normalised bbox.
        label          : text to show above the bbox (e.g. "Right | PEACE").
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
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1)   # black border

    # Bounding box + label
    if detection_rect is not None:
        x1, y1 = int(detection_rect[0] * w), int(detection_rect[1] * h)
        x2, y2 = int(detection_rect[2] * w), int(detection_rect[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        if label:
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_display_loop(pipeline: dai.Pipeline,
                     queues: Dict[str, Any]) -> None:
    """
    Reads all output queues, assembles hand results per frame, and shows
    them in an OpenCV window until the user presses 'q'.
    """
    q_video = queues["video"]
    q_dets  = queues["dets"]
    q_kp    = queues["kp"]
    q_conf  = queues["conf"]
    q_hand  = queues["hand"]

    last_hands:       List       = []
    det_rects_by_seq: dict       = {}
    pose_buffer:      dict       = defaultdict(dict)
    hands_by_seq:     dict       = defaultdict(list)

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

        # ── Collect per-head landmark messages ────────────────────────────────
        for slot, q in ((0, q_kp), (1, q_conf), (2, q_hand)):
            msg = q.tryGet()
            if msg is not None:
                pose_buffer[msg.getSequenceNum()][slot] = msg

        # ── Assemble complete triples (all 3 heads for same seqnum) ───────────
        for seq in sorted(s for s, v in pose_buffer.items() if len(v) == 3):
            entry = pose_buffer.pop(seq)
            try:
                if entry[1].prediction < CONFIDENCE_THRESHOLD:
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
                                          "rect": det_rect,
                                          "label": label})
            except Exception:
                pass

        # ── Commit fully-assembled frames to last_hands ───────────────────────
        for seq in sorted(s for s in hands_by_seq if s not in pose_buffer):
            assembled = hands_by_seq.pop(seq)
            if assembled:
                last_hands = assembled

        # ── Prune stale buffers ───────────────────────────────────────────────
        for buf in (pose_buffer, det_rects_by_seq, hands_by_seq):
            if len(buf) > 30:
                for old in sorted(buf.keys())[:-15]:
                    buf.pop(old, None)

        # ── Draw and show ─────────────────────────────────────────────────────
        if in_video is not None:
            frame = in_video.getCvFrame()

            for hand in last_hands:
                draw_hand(frame,
                          hand["keypoints"],
                          detection_rect=hand["rect"],
                          label=hand["label"])

            cv2.putText(frame, f"Hands: {len(last_hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("Hand Pose — OAK-D", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

    cv2.destroyAllWindows()

