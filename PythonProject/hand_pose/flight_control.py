"""
flight_control.py
-----------------
Spatial Anchor Flight Controller for Hand Gesture Drone Control.

Uses Landmark 9 (middle finger knuckle) 3D position from OAK-D stereo depth.
Saves anchor when gesture detected, then maps movement deltas to PWM outputs.

Gestures:
    ONE        -> Failsafe / Brake (hover, clear anchor)
    THUMBS_UP  -> Takeoff & Drift (full 3D: throttle=Y, pitch=Z, roll=X)
    FOUR/FIVE  -> Cruise Mode (altitude hold: pitch=Z, roll=X)
    PEACE      -> Yaw Rotation (yaw=X only)
    FIST       -> Land (auto-level descent, clear anchor)
"""
import numpy as np
from typing import List, Tuple, Optional

from .config import (
    THUMBS_UP_THROTTLE_SCALE, THUMBS_UP_PITCH_SCALE, THUMBS_UP_ROLL_SCALE,
    CRUISE_PITCH_SCALE, CRUISE_ROLL_SCALE,
    PEACE_YAW_SCALE
)


class DroneGestureController:
    """
    Spatial Anchor Flight Controller.

    Core Logic:
      - Landmark 9 (middle finger knuckle) is saved as a 3D anchor when gesture detected
      - For every frame, calculate Delta = Current - Anchor in real 3D space (mm)
      - Map deltas to PWM outputs using configurable scale factors
      - Save new anchor whenever gesture changes
      - ONE gesture clears anchor immediately (hover mode)
    """

    def __init__(self, smoothing: float = 0.15, deadzone: int = 40):
        """
        smoothing: EMA alpha (0.0=smooth/laggy, 1.0=raw/instant)
        deadzone: PWM units around 1500 to snap to center
        """
        self.alpha = smoothing
        self.deadzone = deadzone

        # Smoothed outputs
        self.smooth_roll = 1500
        self.smooth_pitch = 1500
        self.smooth_throttle = 1500
        self.smooth_yaw = 1500

        # Spatial Anchor System
        self.current_gesture = None
        self.anchor_x_mm = None      # Landmark 9 X in millimeters
        self.anchor_y_mm = None      # Landmark 9 Y in millimeters
        self.anchor_z_mm = None      # Landmark 9 Z (depth) in millimeters

        # Depth history buffer - keep last 3 valid depth readings for smoothing
        self.depth_history = []
        self.depth_history_size = 3

    def _dz(self, value: int, center: int = 1500) -> int:
        """Apply deadzone: force to center if within ±deadzone"""
        return center if abs(value - center) < self.deadzone else value

    def _ema(self, raw: int, prev: int) -> int:
        """Exponential Moving Average filter"""
        return int(self.alpha * raw + (1.0 - self.alpha) * prev)

    def _apply_deadzone(self, delta_mm: float, deadzone_mm: float) -> float:
        """
        Apply deadzone: if delta is smaller than threshold, snap to 0.

        Args:
            delta_mm: movement in millimeters from anchor
            deadzone_mm: threshold in mm below which to snap to 0

        Returns:
            0 if |delta| < deadzone, else delta
        """
        if abs(delta_mm) < deadzone_mm:
            return 0.0
        return delta_mm

    def _smooth_depth(self, z_mm: float) -> float:
        """
        Smooth depth by averaging with history buffer.

        Args:
            z_mm: current depth reading in mm

        Returns:
            smoothed depth (average of last N readings)
        """
        self.depth_history.append(z_mm)
        if len(self.depth_history) > self.depth_history_size:
            self.depth_history.pop(0)

        return float(np.mean(self.depth_history))

    def _get_landmark_9_3d(self, keypoints_norm, depth_frame, frame_shape):
        """
        Extract Landmark 9 (middle finger knuckle) 3D position from OAK-D.

        Args:
            keypoints_norm: 21 normalized (x, y) landmarks
            depth_frame: OAK-D depth frame (numpy array in mm) or None
            frame_shape: (H, W, C)

        Returns:
            (x_mm, y_mm, z_mm) or (None, None, None) if depth not available
        """
        try:
            # Check if depth_frame is valid FIRST
            if depth_frame is None:
                return None, None, None

            h, w = frame_shape[:2]

            # Landmark 9 normalized position
            lm9_x_norm = keypoints_norm[9][0]
            lm9_y_norm = keypoints_norm[9][1]

            # Convert to pixel coordinates
            px_x = int(np.clip(lm9_x_norm * w, 0, w - 1))
            px_y = int(np.clip(lm9_y_norm * h, 0, h - 1))

            # Get depth at this location - VERY LARGE ROI (20x20) for aggressive filtering
            roi_half = 10
            roi = depth_frame[max(0, px_y - roi_half):min(h, px_y + roi_half),
                             max(0, px_x - roi_half):min(w, px_x + roi_half)]
            valid_depths = roi[roi > 0]

            if len(valid_depths) < 10:  # Need at least 10 valid samples
                return None, None, None

            # Aggressive filtering: use median + percentile filtering
            z_mm = float(np.median(valid_depths))

            # Additional check: reject if depth varies too much (outlier frame)
            q25 = float(np.percentile(valid_depths, 25))
            q75 = float(np.percentile(valid_depths, 75))
            iqr = q75 - q25

            # If median is too far from the middle quartile, it's probably noise
            if z_mm < q25 - iqr or z_mm > q75 + iqr:
                return None, None, None

            # Additional check: reject depth if it seems unreasonable
            # Valid hand depth range: 200mm (very close) to 2000mm (very far)
            if z_mm < 200 or z_mm > 2000:
                return None, None, None

            # Apply depth smoothing buffer (average last 3 readings)
            z_mm = self._smooth_depth(z_mm)

            # OAK-D intrinsics (approximate for 1280x720)
            # Note: For best accuracy, calibrate your specific camera!
            fx = 632.0  # focal length X
            fy = 632.0  # focal length Y
            cx = 640.0  # principal point X
            cy = 360.0  # principal point Y

            # Convert pixel + depth to 3D coordinates in millimeters
            x_mm = (px_x - cx) * z_mm / fx
            y_mm = (px_y - cy) * z_mm / fy

            return x_mm, y_mm, z_mm

        except Exception as e:
            print(f"[FlightCtrl] Error getting Landmark 9 3D: {e}")
            return None, None, None

    def process_hand(self, gesture: str, keypoints_norm, depth_frame, frame_shape):
        """
        Main flight control logic based on spatial anchor deltas.

        Args:
            gesture: from recognize_gesture() (ONE, OK, TWO, FOUR, FIVE, PEACE, FIST)
            keypoints_norm: 21 normalized (x, y) landmarks
            depth_frame: OAK-D depth frame (numpy array in mm)
            frame_shape: (H, W, C)

        Returns:
            (roll, pitch, throttle, yaw) each int in [1000, 2000]
        """

        # Handle None/Unknown gestures - return neutral HOVER (1500 throttle for safety)
        if gesture is None or gesture == "UNKNOWN":
            return (1500, 1500, 1500, 1500)

        # STATE 1: ONE (Failsafe / Brake) - Clear anchor immediately
        if gesture == "ONE":
            self.anchor_x_mm = None
            self.anchor_y_mm = None
            self.anchor_z_mm = None
            self.current_gesture = "ONE"

            # Neutral hover
            self.smooth_roll = 1500
            self.smooth_pitch = 1500
            self.smooth_throttle = 1500
            self.smooth_yaw = 1500

            return (self.smooth_roll, self.smooth_pitch, self.smooth_throttle, self.smooth_yaw)

        # STATE 1B: TWO (Failsafe / Brake) - Also clear anchor immediately (same as ONE)
        if gesture == "TWO":
            self.anchor_x_mm = None
            self.anchor_y_mm = None
            self.anchor_z_mm = None
            self.current_gesture = "TWO"

            # Neutral hover
            self.smooth_roll = 1500
            self.smooth_pitch = 1500
            self.smooth_throttle = 1500
            self.smooth_yaw = 1500

            return (self.smooth_roll, self.smooth_pitch, self.smooth_throttle, self.smooth_yaw)

        # STATE 5: FIST (Land) - Clear anchor
        if gesture == "FIST":
            self.anchor_x_mm = None
            self.anchor_y_mm = None
            self.anchor_z_mm = None
            self.current_gesture = "FIST"

            # Auto-level descent
            self.smooth_roll = 1500
            self.smooth_pitch = 1500
            self.smooth_throttle = 1400
            self.smooth_yaw = 1500

            return (self.smooth_roll, self.smooth_pitch, self.smooth_throttle, self.smooth_yaw)

        # For remaining states, get current 3D position of Landmark 9
        curr_x_mm, curr_y_mm, curr_z_mm = self._get_landmark_9_3d(keypoints_norm, depth_frame, frame_shape)

        # If depth not available, use last known position (don't return neutral)
        if curr_x_mm is None or curr_y_mm is None or curr_z_mm is None:
            # Use last smoothed values instead of returning neutral
            if gesture == "FIST" or gesture == "PEACE":
                # These don't need 3D, they're handled above
                pass
            else:
                # Return last smoothed values for OK/FOUR/FIVE
                return (self.smooth_roll, self.smooth_pitch, self.smooth_throttle, self.smooth_yaw)

        # Check if gesture changed - if so, save new anchor
        # SPECIAL: When switching gestures, RESET Z-AXIS (depth) to recalibrate
        if gesture != self.current_gesture:
            # Always update X and Y to current position
            self.anchor_x_mm = curr_x_mm
            self.anchor_y_mm = curr_y_mm
            # RESET Z-AXIS when gesture changes - critical for accurate throttle
            self.anchor_z_mm = curr_z_mm
            self.current_gesture = gesture
            print(f"[FlightCtrl] Gesture changed to '{gesture}' - Z-axis RESET: X={curr_x_mm:.0f}mm Y={curr_y_mm:.0f}mm Z={curr_z_mm:.0f}mm")

        # Ensure anchor is set (safety check)
        if self.anchor_x_mm is None or self.anchor_y_mm is None or self.anchor_z_mm is None:
            self.anchor_x_mm = curr_x_mm
            self.anchor_y_mm = curr_y_mm
            self.anchor_z_mm = curr_z_mm
            self.current_gesture = gesture
            print(f"[FlightCtrl] Anchor initialized for '{gesture}': X={curr_x_mm:.0f}mm Y={curr_y_mm:.0f}mm Z={curr_z_mm:.0f}mm")

        # Calculate deltas (movement from anchor)
        delta_x = curr_x_mm - self.anchor_x_mm  # Left/Right
        delta_y = curr_y_mm - self.anchor_y_mm  # Up/Down (vertical)
        delta_z = curr_z_mm - self.anchor_z_mm  # Depth (push/pull toward camera)

        # Apply deadzone to eliminate hand shake
        from .config import DEADZONE_X_MM, DEADZONE_Y_MM, DEADZONE_Z_MM
        delta_x = self._apply_deadzone(delta_x, DEADZONE_X_MM)
        delta_y = self._apply_deadzone(delta_y, DEADZONE_Y_MM)
        delta_z = self._apply_deadzone(delta_z, DEADZONE_Z_MM)


        # STATE 2: OK (Takeoff & Drift) - Full 3D control
        if gesture == "OK":
            # Throttle: Vertical hand movement (positive Y = UP = climb)
            # Invert because screen Y goes down as values increase
            # CLAMPED: Never go below 1500 (can only climb, not descend)
            raw_throttle = max(1500, int(1500 - delta_y * THUMBS_UP_THROTTLE_SCALE))

            # Pitch: Depth (positive delta_z = away = pitch UP/back)
            raw_pitch = int(1500 + delta_z * THUMBS_UP_PITCH_SCALE)

            # Roll: Horizontal movement
            raw_roll = int(1500 + delta_x * THUMBS_UP_ROLL_SCALE)

            # Yaw: Locked
            raw_yaw = 1500

        # STATE 3: FOUR/FIVE (Cruise Mode) - Altitude hold, pitch/roll control
        elif gesture in ("FOUR", "FIVE"):
            # Throttle: Locked
            raw_throttle = 1500

            # Pitch: Depth (positive delta_z = away = pitch UP/back)
            raw_pitch = int(1500 + delta_z * CRUISE_PITCH_SCALE)

            # Roll: Horizontal movement
            raw_roll = int(1500 + delta_x * CRUISE_ROLL_SCALE)

            # Yaw: Locked
            raw_yaw = 1500

        # STATE 4: PEACE (Yaw Rotation) - Yaw control only
        elif gesture == "PEACE":
            # Throttle: Locked
            raw_throttle = 1500

            # Pitch: Locked
            raw_pitch = 1500

            # Roll: Locked
            raw_roll = 1500

            # Yaw: Horizontal movement (delta_x)
            raw_yaw = int(1500 + delta_x * PEACE_YAW_SCALE)

        else:
            # Unknown gesture - return neutral
            print(f"[FlightCtrl] Unknown gesture: '{gesture}'")
            return (1500, 1500, 1500, 1500)


        # Clamp all PWM values to valid range
        raw_roll = max(1000, min(2000, raw_roll))
        raw_pitch = max(1000, min(2000, raw_pitch))
        raw_throttle = max(1000, min(2000, raw_throttle))
        raw_yaw = max(1000, min(2000, raw_yaw))

        # Apply deadzone
        raw_roll = self._dz(raw_roll)
        raw_pitch = self._dz(raw_pitch)
        raw_throttle = self._dz(raw_throttle)
        raw_yaw = self._dz(raw_yaw)

        # Apply EMA smoothing
        self.smooth_roll = self._ema(raw_roll, self.smooth_roll)
        self.smooth_pitch = self._ema(raw_pitch, self.smooth_pitch)
        self.smooth_throttle = self._ema(raw_throttle, self.smooth_throttle)
        self.smooth_yaw = self._ema(raw_yaw, self.smooth_yaw)

        return (self.smooth_roll, self.smooth_pitch, self.smooth_throttle, self.smooth_yaw)

