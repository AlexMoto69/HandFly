"""
config.py
---------
All constants, colour definitions, model slugs, and the on-device
Script node code in one place. Import from here instead of hardcoding
values across files.
"""
import textwrap

# ── Detection / pose thresholds ───────────────────────────────────────────────
PADDING              = 0.2
CONFIDENCE_THRESHOLD = 0.5

# ── Stereo depth throttle range (millimetres from camera) ────────────────────
# Hand THIS close  → T:2000 (full throttle)
THROTTLE_NEAR_MM = 200    # 20 cm
# Hand THIS far    → T:1000 (zero throttle)
THROTTLE_FAR_MM  = 700    # 70 cm

# ── Yaw angle range (degrees of wrist roll, knuckle vector kp5→kp17) ─────────
YAW_ANGLE_MAX = 15.0

# ── SPATIAL ANCHOR SCALE FACTORS (Gesture-based Flight Control) ──────────────
# Landmark 9 (Middle Finger Knuckle) is the spatial anchor in 3D from OAK-D
# All deltas are in millimeters from the anchor point when gesture detected

# State 2: THUMBS UP / OK - Full 3D Control
# These are MUCH larger now because deltas in mm are huge (100-300mm)
THUMBS_UP_THROTTLE_SCALE = 2.0    # 100mm up/down = ±200 PWM (was 0.5, too small!)
THUMBS_UP_PITCH_SCALE = 2.5       # 100mm push/pull = ±250 PWM (increased for responsiveness)
THUMBS_UP_ROLL_SCALE = 2.0        # 100mm left/right = ±200 PWM

# State 3: FOUR/FIVE - Cruise Mode (Altitude Hold)
CRUISE_PITCH_SCALE = 2.0          # 100mm depth = ±200 PWM (increased for responsiveness)
CRUISE_ROLL_SCALE = 2.0           # 100mm horizontal = ±200 PWM

# State 4: PEACE - Yaw Rotation Only
PEACE_YAW_SCALE = 3.0             # 100mm left/right = ±300 PWM (more sensitive for yaw)

# ── DEADZONE (mm) - Ignore small movements to prevent jitter from hand shake ─
# If delta is smaller than this, treat as 0 (snap to center)
DEADZONE_X_MM = 15.0              # Left/Right shake deadzone (mm)
DEADZONE_Y_MM = 15.0              # Up/Down shake deadzone (mm)
DEADZONE_Z_MM = 20.0              # Depth shake deadzone (mm) - larger because depth is noisier

# ── MediaPipe 21-landmark skeleton connections ────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm cross
]

# ── OpenCV BGR colours ────────────────────────────────────────────────────────
COLOR_JOINT = (0, 255, 0)
COLOR_BONE  = (255, 200, 0)
COLOR_BOX   = (0, 200, 255)
COLOR_TEXT  = (255, 255, 255)

# ── Luxonis Model Zoo slugs (same model works on RVC2 and RVC4) ───────────────
PALM_MODEL_SLUG = "luxonis/mediapipe-palm-detection:192x192"
HAND_MODEL_SLUG = "luxonis/mediapipe-hand-landmarker:224x224"

# ── On-device Script code (inlined string — no .py file on disk needed) ───────
# Runs on the OAK chip. Reads one camera frame + N crop configs per detection
# frame, then fans them out to the ImageManip node for the landmark model.
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

