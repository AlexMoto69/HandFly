"""
config.py
---------
All constants, colour definitions, model slugs, and the on-device
Script node code in one place. Import from here instead of hardcoding
values across files.
"""
import textwrap

# ── Detection / pose thresholds ───────────────────────────────────────────────
PADDING              = 0.1
CONFIDENCE_THRESHOLD = 0.5

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

