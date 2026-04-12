#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

dot_intensity = 1
DOT_STEP = 0.1

flood_intensity = 1
FLOOD_STEP = 0.1

color = (255, 255, 255)

# Create pipeline
device = dai.Device()
with dai.Pipeline(device) as pipeline:
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)

    # Spatial location calculator for depth ROI
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    # For Camera.build(), we need to get the mono output first
    monoLeftOut = monoLeft.requestOutput((640, 400))
    monoRightOut = monoRight.requestOutput((640, 400))

    # Link mono cameras to stereo
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    # Linking stereo to spatial calculator
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    # Config for ROI
    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.6, 0.6)

    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 10
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft, bottomRight)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    # Output queues for display
    leftQueue = monoLeftOut.createOutputQueue()
    rightQueue = monoRightOut.createOutputQueue()

    xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
    outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()

    inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

    pipeline.start()
    pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(dot_intensity)
    pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)

    print("✓ Active Stereo ENABLED (Laser Dot Projector + Depth Map)")
    print("W/S: Laser intensity | A/D: Flood light | Arrow keys: Move ROI | Q: Quit")

    stepSize = 0.05
    newConfig = False

    while pipeline.isRunning():
        leftSynced = leftQueue.get()
        rightSynced = rightQueue.get()
        assert isinstance(leftSynced, dai.ImgFrame)
        assert isinstance(rightSynced, dai.ImgFrame)

        # Get depth data
        spatialData = xoutSpatialQueue.get().getSpatialLocations()
        outputDepthIMage = outputDepthQueue.get()
        frameDepth = outputDepthIMage.getFrame()

        valid_depths = frameDepth[frameDepth > 0]
        if len(valid_depths) > 0:
            print(f"Median depth: {np.median(valid_depths):.0f}mm")

        # Process depth frame for visualization
        depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        # Draw ROI and depth data
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

        cv2.imshow("left", leftSynced.getCvFrame())
        cv2.imshow("right", rightSynced.getCvFrame())
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break
        elif key == ord("w"):
            dot_intensity += DOT_STEP
            if dot_intensity > 1:
                dot_intensity = 1
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(dot_intensity)
            print(f"Dot intensity: {dot_intensity*100:.0f}%")
        elif key == ord("s"):
            dot_intensity -= DOT_STEP
            if dot_intensity < 0:
                dot_intensity = 0
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(dot_intensity)
            print(f"Dot intensity: {dot_intensity*100:.0f}%")
        elif key == ord("a"):
            flood_intensity += FLOOD_STEP
            if flood_intensity > 1:
                flood_intensity = 1
            pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)
            print(f"Flood intensity: {flood_intensity*100:.0f}%")
        elif key == ord("d"):
            flood_intensity -= FLOOD_STEP
            if flood_intensity < 0:
                flood_intensity = 0
            pipeline.getDefaultDevice().setIrFloodLightIntensity(flood_intensity)
            print(f"Flood intensity: {flood_intensity*100:.0f}%")
        elif key == 82:  # Up arrow
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == 84:  # Down arrow
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == 81:  # Left arrow
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == 83:  # Right arrow
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            inputConfigQueue.send(cfg)
            newConfig = False

