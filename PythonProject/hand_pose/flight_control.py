"""
flight_control.py
-----------------
Translates hand pose data into drone flight commands (PPM values).

PPM range convention:
    1000 = minimum  (stick full left / full back / zero throttle)
    1500 = center   (stick neutral)
    2000 = maximum  (stick full right / full forward / full throttle)

Throttle depth metric — palm detection bbox width:
    We use the width of the palm detection bounding box (normalised 0..1)
    as a proxy for hand distance from the camera.

    Confirmed real-world values from debug session:
        ~0.39  — hand very close (~20 cm)  → T:2000 (full throttle)
        ~0.20  — hand mid distance (~40 cm) → T:1500 (hover)
        ~0.10  — hand far (~80 cm)          → T:1000 (min throttle)

    Tune THROTTLE_NEAR / THROTTLE_FAR if your camera distance differs.

Gesture behaviour:
    FIVE  → active flight (roll + pitch + throttle live)
    FIST  → emergency stop (all neutral, throttle cut)
    PEACE → emergency stop (same as FIST)
    other → safe hold, throttle cut
"""
import numpy as np
from typing import List, Tuple

# ── Throttle depth range (raw palm detection bbox width, before padding) ──────
# Calibrated from debug session:
#   ~0.32  hand at ~20 cm  → T:2000  full throttle
#   ~0.10  hand at ~70 cm  → T:1000  min throttle
# The midpoint ~0.21 at ~45 cm maps to T:1500 (hover).
# Adjust THROTTLE_NEAR/FAR here if your camera or hand size differs.
THROTTLE_NEAR = 0.32   # 20 cm
THROTTLE_FAR  = 0.10   # 70 cm


class DroneGestureController:
    def __init__(self, smoothing: float = 0.3, deadzone: int = 50):
        """
        Args:
            smoothing : EMA alpha — 0.0 (max smooth/laggy) to 1.0 (raw/instant)
            deadzone  : PPM units around 1500 treated as exactly 1500 (prevents drift)
        """
        self.alpha    = smoothing
        self.deadzone = deadzone

        # Smoothed PPM state — start at safe hover values
        self.smooth_roll     = 1500
        self.smooth_pitch    = 1500
        self.smooth_throttle = 1000

        # Normalised camera X/Y zone that maps to full stick travel
        self.cam_min = 0.25
        self.cam_max = 0.75

    # ── Internal filters ──────────────────────────────────────────────────────

    def _deadzone(self, value: int, center: int = 1500) -> int:
        """Snap value to center if it falls within the deadzone band."""
        return center if abs(value - center) < self.deadzone else value

    def _ema(self, raw: int, previous: int) -> int:
        """Exponential Moving Average — blends new value with history."""
        return int(self.alpha * raw + (1.0 - self.alpha) * previous)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_hand(self,
                     gesture: str,
                     wrist_x: float,
                     wrist_y: float,
                     depth_metric: float) -> tuple:
        """
        Convert hand data to (roll, pitch, throttle, yaw) PPM values.

        Args:
            gesture      : string from recognize_gesture(), e.g. "FIVE", "FIST"
            wrist_x      : normalised [0..1] X of landmark 0 (wrist)
            wrist_y      : normalised [0..1] Y of landmark 0 (wrist)
            depth_metric : bbox width from bbox_depth_metric()
                           large (~0.35) = close,  small (~0.10) = far away

        Returns:
            (roll, pitch, throttle, yaw) — each an int in [1000, 2000]
        """

        # ── EMERGENCY STOP ────────────────────────────────────────────────────
        if gesture in ("FIST", "PEACE"):
            self.smooth_roll     = 1500
            self.smooth_pitch    = 1500
            self.smooth_throttle = 1000
            return 1500, 1500, 1000, 1500

        # ── ACTIVE FLIGHT ─────────────────────────────────────────────────────
        if gesture == "FIVE":
            # Roll — wrist X: left(1000) ↔ right(2000)
            raw_roll = int(np.interp(wrist_x,
                                     [self.cam_min, self.cam_max],
                                     [1000, 2000]))
            raw_roll = self._deadzone(raw_roll)

            # Pitch — wrist Y: high on screen = forward(1000), low = back(2000)
            # Screen Y increases downward, so invert the mapping
            raw_pitch = int(np.interp(wrist_y,
                                      [self.cam_min, self.cam_max],
                                      [2000, 1000]))
            raw_pitch = self._deadzone(raw_pitch)

            # Throttle — depth metric: close(THROTTLE_NEAR)→2000, far(THROTTLE_FAR)→1000
            # np.interp clamps automatically so no extra clip needed
            raw_throttle = int(np.interp(depth_metric,
                                         [THROTTLE_FAR, THROTTLE_NEAR],
                                         [1000, 2000]))

            # Apply EMA smoothing
            self.smooth_roll     = self._ema(raw_roll,     self.smooth_roll)
            self.smooth_pitch    = self._ema(raw_pitch,    self.smooth_pitch)
            self.smooth_throttle = self._ema(raw_throttle, self.smooth_throttle)

            return self.smooth_roll, self.smooth_pitch, self.smooth_throttle, 1500

        # ── UNRECOGNISED GESTURE — safe hold ──────────────────────────────────
        return 1500, 1500, 1000, 1500
