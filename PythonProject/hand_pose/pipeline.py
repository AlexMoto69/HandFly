"""
pipeline.py
-----------
Everything that touches DepthAI:
  - ProcessDetections  : HostNode that converts palm detections → crop configs
  - build_pipeline()   : builds the full two-stage pipeline and returns queues
"""
from typing import Tuple, Dict, Any

import depthai as dai
from depthai_nodes import ImgDetectionsExtended, ImgDetectionExtended
from depthai_nodes.node import ParsingNeuralNetwork

from .config import (
    PADDING, SCRIPT_CODE,
    PALM_MODEL_SLUG, HAND_MODEL_SLUG,
)


# ══════════════════════════════════════════════════════════════════════════════
# HOST NODE — converts detections to per-hand ImageManip crop configs
# ══════════════════════════════════════════════════════════════════════════════

class ProcessDetections(dai.node.HostNode):
    """
    Receives ImgDetectionsExtended, emits:
      - num_configs_output : dai.Buffer whose byte-length == number of detections
      - config_output      : one ImageManipConfig per detection (rotated-rect crop)
    """

    def __init__(self):
        super().__init__()
        self.detections_input   = self.createInput()
        self.config_output      = self.createOutput()
        self.num_configs_output = self.createOutput()
        self.padding   = PADDING
        self._target_w = None
        self._target_h = None

    def build(self,
              detections_input: dai.Node.Output,
              padding: float,
              target_size: Tuple[int, int]) -> "ProcessDetections":
        self.padding   = padding
        self._target_w = target_size[0]
        self._target_h = target_size[1]
        self.link_args(detections_input)
        return self

    def process(self, img_detections: dai.Buffer) -> None:
        assert isinstance(img_detections, ImgDetectionsExtended)
        detections = img_detections.detections

        # Send count first so the script node knows how many crops to expect
        num_msg = dai.Buffer(len(detections))
        num_msg.setTimestamp(img_detections.getTimestamp())
        num_msg.setSequenceNum(img_detections.getSequenceNum())
        self.num_configs_output.send(num_msg)

        for det in detections:
            det: ImgDetectionExtended
            rect = det.rotated_rect

            new_rect = dai.RotatedRect()
            new_rect.center.x    = rect.center.x
            new_rect.center.y    = rect.center.y
            new_rect.size.width  = rect.size.width  + self.padding * 2
            new_rect.size.height = rect.size.height + self.padding * 2
            new_rect.angle       = 0

            cfg = dai.ImageManipConfig()
            cfg.addCropRotatedRect(new_rect, normalizedCoords=True)
            cfg.setOutputSize(self._target_w, self._target_h,
                              dai.ImageManipConfig.ResizeMode.STRETCH)
            cfg.setReusePreviousImage(False)
            cfg.setTimestamp(img_detections.getTimestamp())
            cfg.setSequenceNum(img_detections.getSequenceNum())
            self.config_output.send(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(device: dai.Device, fps: int) -> Tuple[dai.Pipeline, Dict[str, Any]]:
    """
    Builds and starts the two-stage hand-pose pipeline with stereo depth.

    Returns a dict of output queues:
        "video"  : camera frames            (ImgFrame)
        "dets"   : palm detections          (ImgDetectionsExtended)
        "kp"     : keypoints                (Keypoints)
        "conf"   : hand presence score      (Predictions)
        "hand"   : handedness score         (Predictions)
        "spatial": spatial location data    (SpatialLocationData) — real Z in mm
    """
    platform   = device.getPlatform().name
    frame_type = (dai.ImgFrame.Type.BGR888p
                  if platform == "RVC2"
                  else dai.ImgFrame.Type.BGR888i)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models …")
    det_desc          = dai.NNModelDescription(PALM_MODEL_SLUG)
    det_desc.platform = platform
    det_archive       = dai.NNArchive(dai.getModelFromZoo(det_desc))

    pose_desc          = dai.NNModelDescription(HAND_MODEL_SLUG)
    pose_desc.platform = platform
    pose_archive       = dai.NNArchive(dai.getModelFromZoo(pose_desc))
    print("Models ready.")

    pipeline = dai.Pipeline(device)

    # ── RGB Camera ────────────────────────────────────────────────────────────
    cam     = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((768, 768), frame_type, fps=fps)

    # ── Stereo depth ──────────────────────────────────────────────────────────
    # Mono left + right → StereoDepth → SpatialLocationCalculator
    mono_left  = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(fps)
    mono_right.setFps(fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(),
                         mono_left.getResolutionHeight())
    stereo.setSubpixel(False)   # subpixel costs extra shaves — keep off for RVC2
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # SpatialLocationCalculator — samples a small ROI around the wrist area.
    # We start with a fixed center ROI; renderer updates it each frame with
    # the real kp[0] (wrist) position so it tracks the hand.
    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial_calc.inputConfig.setWaitForMessage(False)

    # Initial ROI — center of frame, small region (will be overridden at runtime)
    initial_cfg = dai.SpatialLocationCalculatorConfigData()
    initial_cfg.roi                        = dai.Rect(dai.Point2f(0.45, 0.45),
                                                      dai.Point2f(0.55, 0.55))
    initial_cfg.calculationAlgorithm      = \
        dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    initial_cfg.depthThresholds.lowerThreshold = 100
    initial_cfg.depthThresholds.upperThreshold = 10000
    spatial_calc.initialConfig.addROI(initial_cfg)

    stereo.depth.link(spatial_calc.inputDepth)

    # ── Resize → palm detector ────────────────────────────────────────────────
    resize = pipeline.create(dai.node.ImageManip)
    resize.setMaxOutputFrameSize(
        det_archive.getInputWidth() * det_archive.getInputHeight() * 3)
    resize.initialConfig.setOutputSize(
        det_archive.getInputWidth(), det_archive.getInputHeight(),
        mode=dai.ImageManipConfig.ResizeMode.STRETCH)
    resize.initialConfig.setFrameType(frame_type)
    cam_out.link(resize.inputImage)

    # ── Stage 1 — palm detection ──────────────────────────────────────────────
    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize.out, det_archive)

    # ── Detection → crop configs ──────────────────────────────────────────────
    proc = pipeline.create(ProcessDetections).build(
        detections_input=det_nn.out,
        padding=PADDING,
        target_size=(pose_archive.getInputWidth(), pose_archive.getInputHeight()))

    # ── Script node — fans one frame per detected hand ────────────────────────
    script = pipeline.create(dai.node.Script)
    script.setScript(SCRIPT_CODE)
    script.inputs["frame_input"].setMaxSize(30)
    script.inputs["config_input"].setMaxSize(30)
    script.inputs["num_configs_input"].setMaxSize(30)
    det_nn.passthrough.link(script.inputs["frame_input"])
    proc.config_output.link(script.inputs["config_input"])
    proc.num_configs_output.link(script.inputs["num_configs_input"])

    # ── Warp each hand crop ───────────────────────────────────────────────────
    pose_manip = pipeline.create(dai.node.ImageManip)
    pose_manip.initialConfig.setOutputSize(
        pose_archive.getInputWidth(), pose_archive.getInputHeight())
    pose_manip.inputConfig.setMaxSize(30)
    pose_manip.inputImage.setMaxSize(30)
    pose_manip.setNumFramesPool(30)
    pose_manip.inputConfig.setWaitForMessage(True)
    script.outputs["output_config"].link(pose_manip.inputConfig)
    script.outputs["output_frame"].link(pose_manip.inputImage)

    # ── Stage 2 — hand landmark model (3 parser heads) ────────────────────────
    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        pose_manip.out, pose_archive)

    # ── Output queues — ALL created BEFORE pipeline.start() ──────────────────
    queues = {
        "video":   cam_out.createOutputQueue(maxSize=4,  blocking=False),
        "dets":    det_nn.out.createOutputQueue(maxSize=8, blocking=False),
        "kp":      pose_nn.getOutput(0).createOutputQueue(maxSize=8, blocking=False),
        "conf":    pose_nn.getOutput(1).createOutputQueue(maxSize=8, blocking=False),
        "hand":    pose_nn.getOutput(2).createOutputQueue(maxSize=8, blocking=False),
        # Stereo spatial depth — Z in mm aligned to RGB frame
        "spatial": spatial_calc.out.createOutputQueue(maxSize=8, blocking=False),
        # Input queue so renderer can push updated ROI configs each frame
        "spatial_cfg": spatial_calc.inputConfig.createInputQueue(),
    }

    pipeline.start()
    return pipeline, queues

