"""
gesture.py
----------
Gesture recognition from 21 MediaPipe hand keypoints.

Supported gestures: FIST, ONE, TWO, THREE, FOUR, FIVE, PEACE, OK
Returns None if no gesture matches.
"""
from typing import List, Tuple, Optional
import numpy as np


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by rays b→a and b→c, in degrees."""
    ba  = a - b
    bc  = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _finger_state(kpts: np.ndarray, tip: int, mid: int, base: int) -> int:
    """
    Returns:
        1  — finger extended (tip above mid above base in image y)
        0  — finger curled   (base above tip)
       -1  — ambiguous
    """
    if kpts[tip][1] < kpts[mid][1] < kpts[base][1]:
        return 1
    if kpts[base][1] < kpts[tip][1]:
        return 0
    return -1


def recognize_gesture(kpts: List[Tuple[float, float]]) -> Optional[str]:
    """
    Classify a hand gesture from 21 normalised (x, y) keypoints.

    Keypoint index layout (MediaPipe convention):
        0  = wrist
        1-4  = thumb  (CMC → tip)
        5-8  = index  (MCP → tip)
        9-12 = middle (MCP → tip)
        13-16= ring   (MCP → tip)
        17-20= pinky  (MCP → tip)
    """
    pts = np.array(kpts)

    # ── Thumb ─────────────────────────────────────────────────────────────────
    d_3_5 = _dist(pts[3], pts[5])
    d_2_3 = _dist(pts[2], pts[3])
    a0    = _angle(pts[0], pts[1], pts[2])
    a1    = _angle(pts[1], pts[2], pts[3])
    a2    = _angle(pts[2], pts[3], pts[4])
    thumb = 1 if (a0 + a1 + a2 > 460 and d_3_5 / d_2_3 > 1.2) else 0

    # ── Four fingers ──────────────────────────────────────────────────────────
    index  = _finger_state(pts,  8,  7,  6)
    middle = _finger_state(pts, 12, 11, 10)
    ring   = _finger_state(pts, 16, 15, 14)
    little = _finger_state(pts, 20, 19, 18)

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

