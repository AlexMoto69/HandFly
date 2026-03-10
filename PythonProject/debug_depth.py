"""Quick debug: print actual depth metric + bbox values live for 20 seconds."""
import time, sys
from collections import defaultdict
import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from hand_pose.pipeline import build_pipeline
from hand_pose.flight_control import hand_depth_metric
from hand_pose.config import CONFIDENCE_THRESHOLD

# write to file AND console
_log = open("debug_depth_out.txt", "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    _log.write(msg + "\n")
    _log.flush()

device   = dai.Device()
platform = device.getPlatform().name
pipeline, queues = build_pipeline(device, 6)

q_video = queues["video"]
q_dets  = queues["dets"]
q_kp    = queues["kp"]
q_conf  = queues["conf"]
q_hand  = queues["hand"]

pose_buffer      = defaultdict(dict)
det_info_by_seq  = {}

print("PUT HAND IN FRONT — running 20 seconds — move it closer and further")
log("PUT HAND IN FRONT — running 20 seconds — move it closer and further")
start = time.time()

while time.time() - start < 20:
    q_video.tryGet()

    d = q_dets.tryGet()
    if d is not None:
        seq = d.getSequenceNum()
        if isinstance(d, ImgDetectionsExtended) and d.detections:
            det = d.detections[0]
            det_info_by_seq[seq] = {
                "w": det.rotated_rect.size.width,
                "h": det.rotated_rect.size.height,
            }
            log(f"  DET  seq={seq}  "
                f"bbox_w={det.rotated_rect.size.width:.3f}  "
                f"bbox_h={det.rotated_rect.size.height:.3f}  "
                f"conf={det.confidence:.2f}")

    for slot, q in ((0, q_kp), (1, q_conf), (2, q_hand)):
        msg = q.tryGet()
        if msg is not None:
            pose_buffer[msg.getSequenceNum()][slot] = msg

    for seq in sorted(s for s, v in pose_buffer.items() if len(v) == 3):
        entry = pose_buffer.pop(seq)
        if entry[1].prediction >= CONFIDENCE_THRESHOLD:
            kpts  = [(kp.x, kp.y) for kp in entry[0].keypoints]
            depth = hand_depth_metric(kpts)
            info  = det_info_by_seq.get(seq, {})
            log(f"  POSE seq={seq}  "
                f"depth_metric={depth:.4f}  "
                f"bbox_w={info.get('w', 0):.3f}  "
                f"wrist=({kpts[0][0]:.2f},{kpts[0][1]:.2f})  "
                f"middle_tip=({kpts[12][0]:.2f},{kpts[12][1]:.2f})  "
                f"index_tip=({kpts[8][0]:.2f},{kpts[8][1]:.2f})")

log("Done.")
_log.close()

