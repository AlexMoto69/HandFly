# HandFly - Hand Gesture Drone Control System

**Real-time hand gesture recognition for drone flight control using OAK-D Pro stereo depth camera and MediaPipe Hands.**

Control your drone with hand gestures! This project uses an OAK-D Pro camera to detect hand landmarks, recognize gestures, and send PWM commands to a Pixhawk drone via Arduino.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [How It Works](#how-it-works)
6. [Hand Gestures](#hand-gestures)
7. [Configuration](#configuration)
8. [File Structure](#file-structure)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

HandFly is a complete hand gesture-based drone control system with three main components:

```
Hand Gesture Recognition (MediaPipe)
           ↓
    3D Spatial Tracking (OAK-D Pro Depth)
           ↓
    Flight Control Processing
           ↓
    Arduino Serial Output (PWM Commands)
           ↓
    Pixhawk Flight Controller
```

**Pipeline:**
1. OAK-D Pro captures RGB video + stereo depth
2. MediaPipe Hands detects 21 hand landmarks in real-time
3. Gesture recognition identifies hand position/shape
4. 3D spatial anchors calculate hand movement deltas
5. Flight controller maps gestures to PWM commands (1000-2000μs)
6. Arduino receives commands and generates PPM signal
7. Pixhawk receives PPM and controls the drone

---

## 🔧 Hardware Requirements

| Component | Purpose |
|-----------|---------|
| **OAK-D Pro** | Stereo depth camera with laser dot projector |
| **Arduino (CH340/Uno)** | Serial bridge to generate PPM signal |
| **Pixhawk Flight Controller** | Drone autopilot (receives PPM signal) |
| **5.8GHz Radio Transmitter** | Sends control commands from Arduino to Pixhawk |
| **PC/Laptop** | Runs hand pose estimation and control logic |

### USB Connection
```
PC (USB 3.1) ←→ OAK-D Pro
PC (USB 2.0) ←→ Arduino (CH340)
Arduino (GPIO Pin 10) ←→ Radio Transmitter (PPM IN)
Radio Transmitter (5.8GHz) ←→ Pixhawk (RC IN)
```

---

## 📦 Installation

### 1. Clone or Download the Project

```bash
cd F:\Github\HandFly\PythonProject
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
```
depthai>=3.0.0          # OAK-D Pro SDK
mediapipe>=0.10.0       # Hand landmark detection
opencv-python>=4.9.0    # Computer vision
numpy>=2.4.0            # Numerical computing
pyserial>=3.5           # Arduino serial communication
```

### 4. Download MediaPipe Hand Model

```bash
# The hand_landmarker.task file should be in:
F:\Github\HandFly\PythonProject\models\hand_landmarker.task

# If missing, download from:
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
```

### 5. Upload Arduino Code

Flash your Arduino with the PPM encoder code:
- Open `Arduino IDE`
- Load the provided PPM encoder sketch
- Select board: "Arduino Uno" or "CH340"
- Select correct COM port
- Upload

---

## 🚀 Quick Start

### Test 1: Verify OAK-D Camera

```bash
python probe_device.py
```

Should show:
```
Connected — Platform: RVC2  |  FPS: 15
```

### Test 2: Run Hand Pose Estimation (No Arduino)

```bash
python standalone_gpu.py
```

- Shows live hand skeleton overlay
- No Arduino required (dry-run mode)
- Press 'Q' to quit
- Press 'R' to recalibrate yaw neutral

### Test 3: Connect Arduino and Test Flight Control

```bash
python standalone_gpu.py --port COM3
```

Replace `COM3` with your Arduino's port. The script will:
1. Detect your hand
2. Send PWM commands to Arduino
3. Display: `R:1500 P:1423 T:1350 Y:1500` (Roll, Pitch, Throttle, Yaw)

### Test 4: Run Full System with Main Modular

```bash
python main_modular.py --port COM3
```

- Loads MediaPipe hand model
- Initializes OAK-D Pro with laser dot projector
- Connects to Arduino
- Begins drone control

---

## 🧠 How It Works

### 1. Hand Detection & Landmarks

MediaPipe Hands identifies **21 landmarks** on your hand:
```
0  = Wrist
1-4 = Thumb
5-8 = Index finger
9-12 = Middle finger
13-16 = Ring finger
17-20 = Pinky
```

### 2. Gesture Recognition

Gestures are recognized by analyzing **finger angles and distances**:

| Gesture | Recognition Logic | Control |
|---------|-------------------|---------|
| **FIST** | All fingers folded | Emergency stop (descend) |
| **ONE** | Index only raised | Failsafe / hover |
| **TWO** | Index + middle raised | Failsafe / hover |
| **FIVE** | All fingers open | Direct manual control |
| **OK** | Thumb-index circle | Takeoff & 3D movement |
| **PEACE** | Index + middle V shape | Yaw rotation only |

### 3. 3D Spatial Anchoring

When you make a gesture, **Landmark 9 (middle finger knuckle)** position is saved as an anchor:

```
Anchor = (X_mm, Y_mm, Z_mm) at gesture start

Every frame:
  Delta_X = Current_X - Anchor_X  (Left/Right movement)
  Delta_Y = Current_Y - Anchor_Y  (Up/Down movement)
  Delta_Z = Current_Z - Anchor_Z  (Toward/Away camera - depth)

PWM_Output = Map(Delta, Range) + Smoothing + Deadzone
```

### 4. Depth Sensing (Active Stereo)

OAK-D Pro provides **stereo depth** enhanced by:
- ✅ **Laser dot projector** (40% intensity) - Adds patterns for texture-poor areas
- ✅ **Spatial filtering** - Removes noise holes
- ✅ **Temporal filtering** - Smooths depth across frames
- ✅ **Subpixel mode** - Fine-grained depth resolution

**Depth → Throttle Mapping:**
```
20cm (close) = 2000 PWM (climb)
70cm (far)   = 1000 PWM (descend)
```

### 5. Flight Control Processing

**Exponential Moving Average (EMA) Filter:**
```python
Smoothed = alpha * Raw_Value + (1 - alpha) * Previous_Smoothed
alpha = 0.15 (adjustable in config.py)
```

**Deadzone Application:**
```python
if |PWM_Value - 1500| < deadzone:
    PWM_Value = 1500  (snap to center)
```

**Arduino Output:**
```
Format: "R:1500 P:1423 T:1350 Y:1500\n"
         Roll  Pitch  Throttle Yaw
         (1000-2000 PWM microseconds)
```

---

## ✋ Hand Gestures

### Gesture Overview

```
FIST         ONE          TWO          THREE        FOUR
(stop)       (hover)      (hover)      (unused)     (cruise)
   👊           ☝️           ✌️           (unused)     🖐️ (4 fingers)
              
FIVE         OK           PEACE
(manual)     (takeoff)    (yaw rotation)
   🖐️          👌            ✌️
```

### Detailed Control Map

| Gesture | State | Controls | Behavior |
|---------|-------|----------|----------|
| **FIST** | Landing | T:1400 R:1500 P:1500 Y:1500 | Descends slowly, auto-levels |
| **ONE** | Failsafe | T:1500 R:1500 P:1500 Y:1500 | Hover, motors armed |
| **TWO** | Failsafe | T:1500 R:1500 P:1500 Y:1500 | Hover, motors armed |
| **OK** | Takeoff | T:Dynamic R:Dynamic P:Dynamic Y:1500 | Move hand to control all axes |
| **FOUR** | Cruise | T:1500 R:Dynamic P:Dynamic Y:1500 | Altitude hold, move horizontally |
| **FIVE** | Manual | T:Dynamic R:Dynamic P:Dynamic Y:1500 | Full 3D manual control |
| **PEACE** | Yaw | T:1500 R:1500 P:1500 Y:Dynamic | Rotate drone in place |

### OK Gesture (Takeoff & 3D Control)

```
Hand Position → PWM Output
───────────────────────────
Raise hand (↑)     → Throttle UP (2000)
Lower hand (↓)     → Throttle DOWN (1000)
Move left (←)      → Roll LEFT (1000)
Move right (→)     → Roll RIGHT (2000)
Push toward cam (↑) → Pitch FORWARD (low PWM)
Pull from cam (↓)  → Pitch BACKWARD (high PWM)
```

---

## ⚙️ Configuration

Edit `hand_pose/config.py` to fine-tune behavior:

```python
# Depth-to-throttle mapping
THROTTLE_NEAR_MM = 200    # 20cm → 2000 PWM (climb)
THROTTLE_FAR_MM = 700     # 70cm → 1000 PWM (descend)

# Control sensitivity
THUMBS_UP_PITCH_SCALE = 1.0    # 100mm depth = ±100 PWM
CRUISE_PITCH_SCALE = 1.0       # 100mm depth = ±100 PWM
CRUISE_ROLL_SCALE = 2.0        # 100mm horizontal = ±200 PWM

# Hand shake filtering
DEADZONE_X_MM = 15.0   # Ignore left/right shake < 15mm
DEADZONE_Y_MM = 15.0   # Ignore up/down shake < 15mm
DEADZONE_Z_MM = 40.0   # Ignore depth shake < 40mm (more noisy)

# EMA smoothing
# Lower alpha = smoother but slower response
# Higher alpha = faster but jittery
alpha = 0.15
```

---

## 📁 File Structure

```
F:\Github\HandFly\PythonProject\
├── standalone_gpu.py              ← Main standalone script
├── main_modular.py                ← Main with modular architecture
├── main.py                        ← Alternative entry point
│
├── hand_pose/                     ← Core flight control package
│   ├── __init__.py
│   ├── config.py                  ← All constants & tuning parameters
│   ├── gesture.py                 ← Gesture recognition logic
│   ├── flight_control.py          ← Spatial anchor controller
│   ├── pipeline.py                ← OAK-D pipeline setup
│   ├── renderer.py                ← Drawing & HUD
│   ├── serial_output.py           ← Arduino serial communication
│   └── __pycache__/
│
├── models/
│   └── hand_landmarker.task       ← MediaPipe hand model
│
├── depth_test.py                  ← Depth testing script
├── DotP_test.py                   ← Laser projector testing
├── Ac_depth_test.py               ← Active stereo testing
│
├── test_gestures.py               ← Gesture recognition testing
├── test_mediapipe_latency.py      ← Performance benchmarking
│
└── requirements.txt               ← Python dependencies
```

---

## 🐛 Troubleshooting

### Camera Not Detected

```bash
python probe_device.py
```

If no device found:
1. Check USB 3.1 cable (must be USB 3.0+)
2. Try different USB port
3. Unplug/replug camera
4. Update depthai: `pip install --upgrade depthai`

### Arduino Not Found

```bash
# List available COM ports
python -m serial.tools.list_ports

# Connect to specific port
python standalone_gpu.py --port COM3
```

### Hand Detection Not Working

- Ensure **good lighting** (at least 500 lux)
- Keep hand **fully visible** in frame
- Hand should be **30-100cm from camera**
- Avoid **shiny/reflective surfaces** behind hand

### Jerky/Unstable Control

**Problem:** PWM values jumping around

**Solutions:**
1. Increase `DEADZONE_*_MM` values in config.py
2. Decrease `alpha` (EMA smoothing factor) - make it slower
3. Ensure **laser projector is enabled** (visible red dots)
4. Check **lighting conditions**

### High Latency

**Problem:** Noticeable delay between hand movement and drone response

**Solutions:**
1. Use `standalone_gpu.py` instead of `main_modular.py`
2. Reduce USB hub chains - connect directly to PC
3. Check CPU usage (should be < 50%)
4. Ensure **mediapipe model is using GPU** (look for GPU acceleration message)

### Laser Dot Projector Not Visible

**Problem:** No red dots visible on hand

**Solutions:**
1. Check intensity: should show "Laser Dot Projector ENABLED at 40%"
2. Ensure room is **not too bright** (works better in dim light)
3. Try moving hand closer to camera
4. Verify device is **OAK-D Pro** (laser requires Pro model)

### Arduino Not Receiving Commands

**Problem:** `[Arduino] Write error` messages

**Solutions:**
1. Check USB cable is data cable (not power-only)
2. Verify Arduino board type in Arduino IDE settings
3. Test with: `python -m serial.tools.miniterm COM3 115200`
4. Add 100μF capacitor across Arduino GND/VCC for power stability

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Hand Detection Latency | 16-25ms (MediaPipe IMAGE mode) |
| Depth Sensing Latency | ~8ms (stereo processing) |
| Total Loop Time | ~30-40ms (25-33 FPS) |
| Serial Output Rate | 25Hz (40ms interval) |
| CPU Usage | 20-40% (RTX 3070 Ti with GPU) |
| Memory Usage | ~500-800MB |

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **R** | Recalibrate yaw neutral (show FIVE gesture) |
| **Esc** | Emergency stop (close all windows) |

---

## 🔗 Integration with Pixhawk

### PPM Signal Format

Arduino generates **8-channel PPM signal**:
```
Channel 1 = Roll
Channel 2 = Pitch
Channel 3 = Throttle
Channel 4 = Yaw
Channels 5-8 = Reserved/Aux
```

### Pixhawk Configuration

In Mission Planner:
1. Set RC input protocol to **PPM**
2. Set receiver servo output to **GPIO Pin 10** (Arduino)
3. Calibrate RC stick ranges
4. Set flight mode to **Stabilize** or **AltHold**

---

## 📝 License & Attribution

- **MediaPipe** - Google (Apache 2.0)
- **DepthAI** - Luxonis (MIT)
- **OpenCV** - Open Source (Apache 2.0)

---

## 🤝 Contributing

Found a bug? Have a feature request?

1. Test the issue with `standalone_gpu.py`
2. Check `requirements.txt` for dependency versions
3. Review `hand_pose/config.py` for tuning suggestions
4. Open an issue with:
   - OS (Windows/Linux/Mac)
   - Device model (OAK-D Pro/Lite)
   - Error message & stack trace

---

## 📚 Quick Reference

### Start Hand Pose Estimation
```bash
python standalone_gpu.py
```

### Start with Arduino
```bash
python standalone_gpu.py --port COM3
```

### Run Full System
```bash
python main_modular.py --port COM3
```

### Test Gestures
```bash
python test_gestures.py
```

### Benchmark Performance
```bash
python test_mediapipe_latency.py
```

### Debug Depth Sensor
```bash
python DotP_test.py
```

---

## ❓ FAQ

**Q: Can I use OAK-D Lite instead of Pro?**
A: Yes, but without laser dot projector. Depth will be less accurate in low-texture areas.

**Q: How accurate is the depth?**
A: ±5-10mm at 50cm with laser enabled. Accuracy degrades beyond 1m.

**Q: Can I control multiple drones?**
A: Currently supports 1 drone. Multi-drone would require channel switching logic.

**Q: Is this compatible with DJI drones?**
A: No, designed for Pixhawk-based drones only. DJI drones require proprietary APIs.

**Q: Can I run this on Jetson Nano?**
A: Yes, but MediaPipe inference will be slower (~50-100ms latency).

---

**Last Updated:** April 2026  
**Status:** Stable & Production Ready

