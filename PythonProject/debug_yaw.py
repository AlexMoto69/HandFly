"""Debug yaw: print the knuckle angle live for 30 seconds.
Uses the standalone pipeline (no stereo) so NNs have full shave budget."""
import time, textwrap
from collections import defaultdict
import numpy as np
import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.node import ParsingNeuralNetwork

PADDING = 0.2
CONFIDENCE_THRESHOLD = 0.5
PALM_MODEL_SLUG = "luxonis/mediapipe-palm-detection:192x192"
HAND_MODEL_SLUG = "luxonis/mediapipe-hand-landmarker:224x224"
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

from hand_pose.pipeline import ProcessDetections

out = open("debug_yaw_out.txt", "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    out.write(msg + "\n")

device   = dai.Device()
platform = device.getPlatform().name
frame_type = dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i

det_desc = dai.NNModelDescription(PALM_MODEL_SLUG); det_desc.platform = platform
det_archive = dai.NNArchive(dai.getModelFromZoo(det_desc))
pose_desc = dai.NNModelDescription(HAND_MODEL_SLUG); pose_desc.platform = platform
pose_archive = dai.NNArchive(dai.getModelFromZoo(pose_desc))

pipeline = dai.Pipeline(device)
cam = pipeline.create(dai.node.Camera).build()
cam_out = cam.requestOutput((768, 768), frame_type, fps=6)

resize = pipeline.create(dai.node.ImageManip)
resize.setMaxOutputFrameSize(det_archive.getInputWidth() * det_archive.getInputHeight() * 3)
resize.initialConfig.setOutputSize(det_archive.getInputWidth(), det_archive.getInputHeight(),
    mode=dai.ImageManipConfig.ResizeMode.STRETCH)
resize.initialConfig.setFrameType(frame_type)
cam_out.link(resize.inputImage)

det_nn = pipeline.create(ParsingNeuralNetwork).build(resize.out, det_archive)
proc = pipeline.create(ProcessDetections).build(
    detections_input=det_nn.out, padding=PADDING,
    target_size=(pose_archive.getInputWidth(), pose_archive.getInputHeight()))

script = pipeline.create(dai.node.Script)
script.setScript(SCRIPT_CODE)
script.inputs["frame_input"].setMaxSize(30)
script.inputs["config_input"].setMaxSize(30)
script.inputs["num_configs_input"].setMaxSize(30)
det_nn.passthrough.link(script.inputs["frame_input"])
proc.config_output.link(script.inputs["config_input"])
proc.num_configs_output.link(script.inputs["num_configs_input"])

pose_manip = pipeline.create(dai.node.ImageManip)
pose_manip.initialConfig.setOutputSize(pose_archive.getInputWidth(), pose_archive.getInputHeight())
pose_manip.inputConfig.setMaxSize(30); pose_manip.inputImage.setMaxSize(30)
pose_manip.setNumFramesPool(30); pose_manip.inputConfig.setWaitForMessage(True)
script.outputs["output_config"].link(pose_manip.inputConfig)
script.outputs["output_frame"].link(pose_manip.inputImage)

pose_nn = pipeline.create(ParsingNeuralNetwork).build(pose_manip.out, pose_archive)

q_video = cam_out.createOutputQueue(maxSize=4,  blocking=False)
q_dets  = det_nn.out.createOutputQueue(maxSize=8, blocking=False)
q_kp    = pose_nn.getOutput(0).createOutputQueue(maxSize=8, blocking=False)
q_conf  = pose_nn.getOutput(1).createOutputQueue(maxSize=8, blocking=False)
q_hand  = pose_nn.getOutput(2).createOutputQueue(maxSize=8, blocking=False)

pipeline.start()
pose_buffer = defaultdict(dict)
det_data    = {}
hands_by_seq = defaultdict(list)

log("TWIST YOUR WRIST LEFT AND RIGHT — 30 seconds")
start = time.time()

while time.time() - start < 30:
    q_video.tryGet()
    d = q_dets.tryGet()
    if d is not None and isinstance(d, ImgDetectionsExtended) and d.detections:
        seq = d.getSequenceNum()
        rects = []
        for det in d.detections:
            cx, cy = det.rotated_rect.center.x, det.rotated_rect.center.y
            rw, rh = det.rotated_rect.size.width, det.rotated_rect.size.height
            hw, hh = rw/2+PADDING, rh/2+PADDING
            rects.append((max(cx-hw,0), max(cy-hh,0), min(cx+hw,1), min(cy+hh,1)))
        det_data[seq] = rects

    for slot, q in ((0,q_kp),(1,q_conf),(2,q_hand)):
        msg = q.tryGet()
        if msg is not None:
            pose_buffer[msg.getSequenceNum()][slot] = msg

    for seq in sorted(s for s,v in pose_buffer.items() if len(v)==3):
        entry = pose_buffer.pop(seq)
        if entry[1].prediction >= CONFIDENCE_THRESHOLD:
            kl = [(kp.x, kp.y) for kp in entry[0].keypoints]
            rects = det_data.get(seq, [])
            if rects:
                xmin,ymin,xmax,ymax = rects[0]
                sx,sy = xmax-xmin, ymax-ymin
                kpts = [(xmin+sx*k[0], ymin+sy*k[1]) for k in kl]
            else:
                kpts = kl

            p5  = np.array(kpts[5])
            p17 = np.array(kpts[17])
            p0  = np.array(kpts[0])
            p9  = np.array(kpts[9])
            dx = p17[0]-p5[0]; dy = p17[1]-p5[1]
            angle_atan2 = float(np.degrees(np.arctan2(dy, dx)))

            # Alternative: angle relative to wrist-mid axis
            wrist_vec = p9 - p0
            knuckle_vec = p17 - p5
            wrist_angle = float(np.degrees(np.arctan2(wrist_vec[1], wrist_vec[0])))
            relative_angle = angle_atan2 - wrist_angle

            log(f"atan2={angle_atan2:+6.1f}  relative={relative_angle:+6.1f}  "
                f"dx={dx:+.3f} dy={dy:+.3f}  "
                f"kp5=({p5[0]:.2f},{p5[1]:.2f}) kp17=({p17[0]:.2f},{p17[1]:.2f})")

log("Done.")
out.close()
pipeline.stop()


