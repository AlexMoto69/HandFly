"""
Ultra-optimized hand detection test - 720p NO SCALING
Measure each component's latency breakdown
"""

import cv2
import time
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision.core import image as mp_image
import depthai as dai
import numpy as np

# ==================== MEDIAPIPE SETUP ====================
print("Loading MediaPipe HandLandmarker...")

# Create HandLandmarker with IMAGE mode (synchronous)
base_options = base_options_module.BaseOptions(
    model_asset_path='models/hand_landmarker.task'
)

# Use IMAGE mode for true synchronous latency measurement
options = hand_landmarker.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE
)

mp_landmarker = hand_landmarker.HandLandmarker.create_from_options(options)
print("✓ MediaPipe loaded (IMAGE mode - synchronous)\n")

# ==================== DEPTHAI SETUP ====================
print("Setting up OAK-D Pro camera...")

device = dai.Device()
pipeline = dai.Pipeline(device)

# 720p camera - maxSize=1 means only newest frame is kept
cam = pipeline.create(dai.node.Camera).build()
cam_out = cam.requestOutput((1280, 720), dai.ImgFrame.Type.BGR888i, fps=30)
cam_queue = cam_out.createOutputQueue(maxSize=1, blocking=False)

pipeline.start()
print("✓ OAK-D started at 720p\n")

# ==================== MAIN LOOP ====================
print("Starting optimized test loop - press 'q' to quit\n")
print("FPS  | Hand? | Total (ms) | GetFrame | ColorCV | MPImage | Detect | Display\n" + "-" * 85)

frame_count = 0
fps_timer = time.time()
detection_count = 0
latencies = []
display_counter = 0

try:
    while True:
        # TIMING: Frame retrieval
        t_start = time.perf_counter()
        in_frame = cam_queue.tryGet()
        if in_frame is None:
            continue
        t_getframe = (time.perf_counter() - t_start) * 1000

        # TIMING: Get CV frame and color conversion
        t_start = time.perf_counter()
        frame = in_frame.getCvFrame()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.astype(np.uint8)
        t_color = (time.perf_counter() - t_start) * 1000

        # TIMING: Create MediaPipe image object
        t_start = time.perf_counter()
        mp_image_obj = mp_image.Image(mp_image.ImageFormat.SRGB, rgb_frame)
        t_mpimage = (time.perf_counter() - t_start) * 1000

        # TIMING: MediaPipe detection
        t_start = time.perf_counter()
        result = mp_landmarker.detect(mp_image_obj)
        t_detect = (time.perf_counter() - t_start) * 1000

        # TIMING: Drawing and display
        t_start = time.perf_counter()
        hands_detected = False
        if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
            hands_detected = True
            hand_landmarks = result.hand_landmarks[0]
            detection_count += 1

            # Draw hand skeleton - OPTIMIZED: only draw 21 circles, no lines
            h, w = frame.shape[:2]
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Smaller circles (2px instead of 5px)

        # Display metrics on frame
        status = "✓ YES" if hands_detected else "- NO "
        total_latency = t_getframe + t_color + t_mpimage + t_detect
        latencies.append(total_latency)
        if len(latencies) > 30:
            latencies.pop(0)
        avg_latency = sum(latencies) / len(latencies)

        cv2.putText(frame, f"Total: {total_latency:.1f}ms | Avg: {avg_latency:.1f}ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if total_latency < 50 else (0, 0, 255), 1)

        # OPTIMIZED: Display every 2nd frame to avoid cv2.imshow() blocking
        # This keeps the processing loop responsive while still updating display ~15fps
        display_counter += 1
        if display_counter % 2 == 0:
            cv2.imshow("Hand Detection - 720p NO SCALING (Optimized Display)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        t_display = (time.perf_counter() - t_start) * 1000

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"{fps:5.1f} | {status} | {avg_latency:9.2f} | {t_getframe:7.2f} | {t_color:6.2f} | {t_mpimage:6.2f} | {t_detect:6.2f} | {t_display:7.2f}")
            frame_count = 0
            detection_count = 0
            fps_timer = time.time()


finally:
    cv2.destroyAllWindows()
    if latencies:
        print(f"\n✓ Test complete - Average total latency: {sum(latencies)/len(latencies):.2f}ms")
    else:
        print("\n✓ Test complete")



