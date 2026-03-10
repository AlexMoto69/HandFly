"""
flight_control.py
-----------------
Translates hand keypoints + stereo depth into drone PPM commands.

PPM range:
    1000 = min   (full left / full back / zero throttle / full yaw left)
    1500 = center (neutral)
    2000 = max   (full right / full forward / full throttle / full yaw right)

Control matrix:
    Roll     <- kp[9] X in frame          (hand left / right)
    Pitch    <- kp[9] Y in frame          (hand up / down)
    Throttle <- stereo Z depth in mm      (hand closer = climb, further = descend)
                Uses OAK-D Pro stereo -- rotation-invariant, true metric depth.
    Yaw      <- angle of kp[5]->kp[17]   (wrist twist clockwise / anti-clockwise)
                kp5 = index MCP,  kp17 = pinky MCP

Gestures:
    FIVE  -> all four axes live
    FIST  -> emergency stop (all neutral, throttle cut)
    PEACE -> emergency stop (same)
    other -> safe hold (throttle cut)
"""
import numpy as np
from typing import List, Tuple

from .config import THROTTLE_NEAR_MM, THROTTLE_FAR_MM, YAW_ANGLE_MAX


def knuckle_yaw_angle(keypoints: List[Tuple[float, float]]) -> float:
    """
    Computes the wrist ROLL angle — how much the hand is twisted left/right.

    Method:
      1. Build the hand's natural "up" vector: kp0 (wrist) → kp9 (middle MCP)
      2. Build the knuckle "horizontal" vector: kp5 (index MCP) → kp17 (pinky MCP)
      3. The signed angle between them, minus 90°, gives the roll deviation
         from a flat neutral position.

    Returns degrees:
        ~  0°  = hand flat and level (neutral yaw)
        + 30°  = wrist rolled clockwise  → yaw right
        - 30°  = wrist rolled anti-clock → yaw left

    This is position-invariant — moving the hand up/down/left/right on screen
    doesn't change the output, only a true wrist twist does.
    """
    p0  = np.array(keypoints[0])   # wrist
    p9  = np.array(keypoints[9])   # middle MCP
    p5  = np.array(keypoints[5])   # index MCP
    p17 = np.array(keypoints[17])  # pinky MCP

    # Hand's natural up-vector (wrist → middle knuckle)
    up_vec      = p9 - p0
    up_angle    = np.degrees(np.arctan2(up_vec[1], up_vec[0]))

    # Knuckle bar vector (index → pinky knuckle)
    knuckle_vec = p17 - p5
    knuckle_angle = np.degrees(np.arctan2(knuckle_vec[1], knuckle_vec[0]))

    # Roll = how far the knuckle bar deviates from being perpendicular to up-vec
    # If the hand is flat: knuckle_angle ≈ up_angle - 90°
    # Roll deviation = knuckle_angle - (up_angle - 90°) = knuckle_angle - up_angle + 90°
    roll = knuckle_angle - up_angle + 90.0

    # Normalise to [-180, 180]
    roll = (roll + 180.0) % 360.0 - 180.0
    return float(roll)


class DroneGestureController:
    def __init__(self, smoothing: float = 0.2, deadzone: int = 40,
                 calibration_frames: int = 30):
        """
        smoothing           : EMA alpha 0.0 (smooth/laggy) to 1.0 (raw/instant)
        deadzone            : PPM units around center snapped to center
        calibration_frames  : number of FIVE-gesture frames to average for
                              yaw neutral calibration on first use
        """
        self.alpha    = smoothing
        self.deadzone = deadzone
        self.smooth_roll     = 1500
        self.smooth_pitch    = 1500
        self.smooth_throttle = 1000
        self.smooth_yaw      = 1500
        self.cam_min = 0.25
        self.cam_max = 0.75

        # Yaw auto-calibration — first N frames with FIVE gesture
        # are averaged to find the user's natural neutral angle
        self._cal_frames   = calibration_frames
        self._cal_samples  = []
        self._yaw_neutral  = None   # set after calibration

    def _dz(self, value: int, center: int = 1500) -> int:
        return center if abs(value - center) < self.deadzone else value

    def _ema(self, raw: int, prev: int) -> int:
        return int(self.alpha * raw + (1.0 - self.alpha) * prev)

    def recalibrate_yaw(self) -> None:
        """Force a new yaw calibration on the next FIVE gesture frames."""
        self._cal_samples = []
        self._yaw_neutral = None
        print("[DroneGestureController] Yaw recalibration started — hold FIVE flat.")

    def process_hand(self,
                     gesture: str,
                     keypoints: List[Tuple[float, float]],
                     depth_mm: float) -> tuple:
        """
        gesture   : from recognize_gesture()
        keypoints : 21 full-frame normalised (x, y) landmarks
        depth_mm  : stereo Z of the wrist region in millimetres
        Returns   : (roll, pitch, throttle, yaw) each int in [1000, 2000]
        """
        # EMERGENCY STOP
        if gesture in ("FIST", "PEACE"):
            self.smooth_roll = self.smooth_pitch = 1500
            self.smooth_throttle = 1000
            self.smooth_yaw = 1500
            return 1500, 1500, 1000, 1500

        # ACTIVE FLIGHT
        if gesture == "FIVE":
            # Roll -- kp[9] X: left(1000) <-> right(2000)
            raw_roll = self._dz(int(np.interp(
                keypoints[9][0], [self.cam_min, self.cam_max], [1000, 2000])))

            # Pitch -- kp[9] Y: hand high (low Y) = forward(1000), low = back(2000)
            raw_pitch = self._dz(int(np.interp(
                keypoints[9][1], [self.cam_min, self.cam_max], [2000, 1000])))

            # Throttle -- real stereo Z in mm: close->2000, far->1000
            raw_throttle = int(np.interp(
                depth_mm,
                [THROTTLE_NEAR_MM, THROTTLE_FAR_MM],
                [2000, 1000]))

            # Yaw -- relative roll deviation from calibrated neutral
            raw_angle = knuckle_yaw_angle(keypoints)

            # Phase 1: collect calibration samples
            if self._yaw_neutral is None:
                self._cal_samples.append(raw_angle)
                if len(self._cal_samples) >= self._cal_frames:
                    self._yaw_neutral = float(np.mean(self._cal_samples))
                    print(f"[DroneGestureController] Yaw neutral calibrated "
                          f"at {self._yaw_neutral:.1f}°")
                # During calibration hold yaw at center
                raw_yaw = 1500
            else:
                # Phase 2: map deviation from neutral → PPM
                deviation = raw_angle - self._yaw_neutral
                # Normalise to [-180, 180] in case of wrap-around
                deviation = (deviation + 180.0) % 360.0 - 180.0
                # 2° deadzone — snaps small jitter to dead center
                if abs(deviation) < 2.0:
                    deviation = 0.0
                # Map ±YAW_ANGLE_MAX → 1200–1800 (soft range, not full 1000–2000)
                raw_yaw = int(np.interp(
                    deviation,
                    [-YAW_ANGLE_MAX, YAW_ANGLE_MAX],
                    [1200, 1800]))

            self.smooth_roll     = self._ema(raw_roll,     self.smooth_roll)
            self.smooth_pitch    = self._ema(raw_pitch,    self.smooth_pitch)
            self.smooth_throttle = self._ema(raw_throttle, self.smooth_throttle)
            self.smooth_yaw      = self._ema(raw_yaw,      self.smooth_yaw)

            return (self.smooth_roll, self.smooth_pitch,
                    self.smooth_throttle, self.smooth_yaw)

        # UNRECOGNISED -- safe hold
        return 1500, 1500, 1000, 1500

