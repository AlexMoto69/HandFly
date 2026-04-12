#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

dot_intensity = 1.0
DOT_STEP = 0.1

color = (255, 255, 255)
stepSize = 0.05

# 1. Create Device FIRST (This is the secret to the v3.0 API)
device = dai.Device()

# 2. Attach Pipeline to Device
with dai.Pipeline(device) as pipeline:
    # Config
    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.6, 0.6)

    # Define sources and outputs using the v3 Camera build() syntax
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    # Linking
    monoLeftOut = monoLeft.requestOutput((640, 400))
    monoRightOut = monoRight.requestOutput((640, 400))
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)

    # Spatial Calculator Config
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 10
    config.depthThresholds.upperThreshold = 10000
    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    config.roi = dai.Rect(topLeft, bottomRight)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    # Create Output Queues directly on the nodes (v3 syntax)
    xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
    outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()
    inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

    stereo.depth.link(spatialLocationCalculator.inputDepth)

    # --- START PIPELINE ---
    pipeline.start()

    # Turn on the Laser! (Using the exact method from your working script)
    pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(dot_intensity)
    pipeline.getDefaultDevice().setIrFloodLightIntensity(0.0)
    print(f"✓ Active Stereo ENABLED! Laser Intensity: {dot_intensity}")
    print("Use W/S for laser intensity, A/D to move ROI, 1-5 for algorithms. 'q' to quit.")

    while pipeline.isRunning():
        spatialDataObj = xoutSpatialQueue.get()
        spatialData = spatialDataObj.getSpatialLocations()

        outputDepthIMage = outputDepthQueue.get()
        frameDepth = outputDepthIMage.getFrame()

        # Colorize depth map
        depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20),
                        fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35),
                        fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50),
                        fontType, 0.5, color)

        cv2.imshow("Active Stereo Depth", depthFrameColor)

        key = cv2.waitKey(1) & 0xFF
        newConfig = False

        if key == ord('q'):
            pipeline.stop()
            break

        # --- LASER INTENSITY CONTROLS ---
        elif key == ord('w'):
            dot_intensity = min(1.0, dot_intensity + DOT_STEP)
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(dot_intensity)
            print(f"Laser INCREASED to {dot_intensity:.1f}")
        elif key == ord('s'):
            dot_intensity = max(0.0, dot_intensity - DOT_STEP)
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(dot_intensity)
            print(f"Laser DECREASED to {dot_intensity:.1f}")

        # --- ROI MOVEMENT CONTROLS ---
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True

        # --- ALGORITHM CONTROLS ---
        elif key == ord('1'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
            newConfig = True
        elif key == ord('2'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
            newConfig = True
        elif key == ord('3'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MAX
            newConfig = True
        elif key == ord('4'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
            newConfig = True
        elif key == ord('5'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
            newConfig = True

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            config.calculationAlgorithm = calculationAlgorithm
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            inputConfigQueue.send(cfg)
            newConfig = False