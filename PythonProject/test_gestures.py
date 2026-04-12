#!/usr/bin/env python3
"""Quick test of gesture recognition - helps verify OK gesture is recognized"""

import numpy as np
from hand_pose.gesture import recognize_gesture

# Test data: Hand at center (normalized 0-1)
# Each test creates synthetic keypoints representing the gesture

def test_gesture(name, kpts_list):
    """Test a gesture and print result"""
    result = recognize_gesture(kpts_list)
    status = "✓" if result == name else "✗"
    print(f"{status} {name}: {result}")

print("Testing Gesture Recognition...")
print("=" * 40)

# FIST: All fingers closed (all ~0 extended)
fist_kpts = [(0.5, 0.5)] * 21  # All same point = fist
test_gesture("FIST", fist_kpts)

# FIVE: All fingers extended
# Simple approximation: spread out hand
five_kpts = [
    (0.5, 0.5),   # 0: wrist
    (0.4, 0.3), (0.35, 0.2), (0.3, 0.1), (0.25, 0.05),  # 1-4: thumb
    (0.6, 0.3), (0.65, 0.2), (0.7, 0.1), (0.75, 0.05),  # 5-8: index
    (0.5, 0.2), (0.5, 0.1), (0.5, 0.0), (0.5, -0.05),   # 9-12: middle
    (0.4, 0.3), (0.4, 0.15), (0.4, 0.05), (0.4, -0.05),  # 13-16: ring
    (0.3, 0.3), (0.3, 0.15), (0.3, 0.05), (0.3, -0.05),  # 17-20: pinky
]
test_gesture("FIVE", five_kpts)

# OK: Thumb up, index down, others down
ok_kpts = [
    (0.5, 0.5),   # 0: wrist
    (0.4, 0.3), (0.4, 0.2), (0.4, 0.1), (0.4, 0.0),    # 1-4: thumb (UP)
    (0.6, 0.3), (0.6, 0.5), (0.6, 0.6), (0.6, 0.65),   # 5-8: index (DOWN)
    (0.5, 0.3), (0.5, 0.5), (0.5, 0.6), (0.5, 0.65),   # 9-12: middle (DOWN)
    (0.4, 0.3), (0.4, 0.5), (0.4, 0.6), (0.4, 0.65),   # 13-16: ring (DOWN)
    (0.3, 0.3), (0.3, 0.5), (0.3, 0.6), (0.3, 0.65),   # 17-20: pinky (DOWN)
]
test_gesture("OK", ok_kpts)

print("=" * 40)
print("✓ Gesture tests complete")
print("\nNow run: python main_modular.py --port COM3")

