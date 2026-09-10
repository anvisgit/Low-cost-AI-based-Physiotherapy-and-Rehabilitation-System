# Complete Walkthrough of the `esp_code` Folder

This document provides a **detailed, code-block-level** guide to the unified ESP32 firmware — what each function does, how data flows between the ESP32 ↔ Backend ↔ ML Model, and exactly which backend code sections to check when troubleshooting.

---

## Folder Structure

```
esp_code/
├── unified_esp/                      # [FOLDER] Single unified ESP32 Arduino Sketch
│   └── unified_esp.ino              # All-in-one: sensors, motors, WiFi, HTTP server
├── test_esp_communication.py         # Python script to test connection & feedback loop
├── TESTING_GUIDE.md                  # Step-by-step guide to run & test
├── esp_code_walkthrough.md           # This file
├── install_driver.bat                # One-click CP2102 USB driver installer
├── silabser.inf / silabser.cat       # Driver configuration files
└── [arm/ arm64/ x64/ x86/]           # Driver binaries for different Windows architectures
```

---

## Architecture — Single ESP32

Unlike the previous two-ESP architecture (Exo Unit + Bridge Unit), the current system uses **one single ESP32** that does everything:

- **Reads sensors** — two MPU6050 IMUs (thigh + shank) for joint angles, one FSR for foot pressure
- **Drives motors** — one BTS7960 motor driver for knee assistance (hip motor actuator and drive pins are disabled)
- **Connects to WiFi** — via WiFiManager captive portal (Samarth-branded UI)
- **Serves HTTP** — web server with endpoints for the backend to poll data and send commands

No ESP-NOW radio. No bridge device. Just one board connected directly to your WiFi network.

---

## Closed-Loop Data Flow

```
═══════════════════════════════════════════════════════════════════════════
  CLOSED LOOP: Sensors → ESP → Backend → ML Model → ESP → Motors → repeat
═══════════════════════════════════════════════════════════════════════════

  ESP32 (Unified)                       Backend (Python FastAPI)
  ┌───────────────────────┐             ┌────────────────────────────┐
  │ MPU6050 IMUs          │             │                            │
  │ FSR foot sensor       │──GET /data──▶ RealSensorHub polls data   │
  │                       │             │                            │
  │                       │             │ PS1 (camera) + ESP angles  │
  │                       │             │          ↓                 │
  │                       │             │ ML Model (RehabNet)        │
  │                       │             │ → prediction: ASSIST/RESIST│
  │                       │             │          ↓                 │
  │ BTS7960 motor drivers │◀─POST /cmd──│ Sends mode_command back    │
  │ (apply torque)        │             │                            │
  └───────────────────────┘             └────────────────────────────┘

  The loop runs continuously during a live exercise session.
═══════════════════════════════════════════════════════════════════════════
```

### Step-by-Step Flow

1. **ESP32 reads sensors** — IMUs compute knee/hip angles via complementary filter, FSR computes foot force
2. **Backend polls `GET /data`** — `RealSensorHub._background_poll_loop()` fetches JSON every 100ms
3. **Camera + sensor fusion** — `session_ws.py` overlays ESP32 angles onto camera pose data
4. **ML model analyzes rep** — `ModelExerciseAnalyzer.analyze_rep()` runs RehabNet inference
5. **Model prediction** — outputs `mode_command: {mode_id: 1=assist/2=resist, target_torque: N}`
6. **Backend sends `POST /command`** — prediction is sent to ESP32's HTTP server
7. **ESP32 applies torque** — `applyControl()` drives motors at the predicted mode/torque
8. **Loop repeats** — sensors read again, and the cycle continues

---

## The JSON Packet (sensor data format)

```json
{
  "knee_angle": 75.3,
  "hip_angle": 45.1,
  "foot_force": 3.42,
  "stance": true,
  "knee_pwm": 95,
  "knee_dir": "ext",
  "hip_pwm": 60,
  "hip_dir": "ext",
  "motors": "on",
  "battery_percent": 100,
  "calibration_status": "calibrated",
  "connected": true
}
```

---

## unified_esp.ino — Function-by-Function Breakdown

### Hardware Initialization

| Function | What it does |
|----------|-------------|
| `setupPWM()` | Configures PWM channels (20kHz, 8-bit) for the knee motor (hip disabled) |
| `calibrateIMU(mpu, offset)` | Samples 100 gyro readings while stationary to find baseline offset |
| `initWiFi()` | WiFiManager captive portal — connects to saved WiFi or starts "Samarth-Exo-Setup" AP |
| `readMPU(mpu, pitch, offset)` | Reads one IMU's raw data, applies complementary filter (98% gyro + 2% accel) |
| `readIMUs()` | Calls `readMPU()` for both thigh and shank, computes knee/hip angles |
| `readFSR()` | Reads FSR analog voltage → resistance → force (Newtons), applies low-pass filter |
| `readBatteryPercent()` | Placeholder — returns 100% (replace with ADC divider when battery monitoring is added) |
| `driveKneeMotor(pwm, forward)` | Sets knee motor speed and direction via BTS7960 |
| `driveHipMotor(pwm, forward)` | Bypassed / disabled on this hardware configuration |
| `stopAllMotors()` | Instantly stops knee motor (PWM = 0, hip disabled) |
| `applyControl()` | Main control logic — reads `current_mode_id` and `current_target_torque` (set by ML prediction via HTTP) and drives motors accordingly |

### Motor Control Modes

| Mode ID | Name | Behavior |
|---------|------|----------|
| 0 | Off | Motors stopped, safe state |
| 1 | Assist | Gait-triggered — motors push only during stance phase when joint angle exceeds threshold |
| 2 | Resist | Constant torque — motors push at the target torque regardless of gait phase |

### HTTP Endpoints

| Endpoint | Method | Purpose | Who calls it |
|----------|--------|---------|-------------|
| `/data` | GET | Returns sensor JSON (angles, forces, motor state) | Backend `RealSensorHub` polls this |
| `/status` | GET | Returns device info (IP, uptime, battery, mode) | Backend on connect |
| `/command` | POST | Receives ML prediction `{mode_id, target_torque}` | Backend `session_ws.py` |
| `/exercise` | POST | Legacy alias for `/command` | Backwards compat |
| `/telemetry` | GET | Raw telemetry with boolean values | Legacy clients |
| `/calibrate` | POST | Returns calibration status | Backend calibrate button |

### Main Loop

```
loop() {
  1. readIMUs()          — Read joint angles (every iteration)
  2. readFSR()           — Read foot pressure (every iteration)
  3. applyControl()      — Apply ML prediction to motors (every 5ms / 200Hz)
  4. server.handleClient() — Process HTTP requests (polls + commands)
  5. delay(1)            — Yield to WiFi/system tasks
}
```

---

## Backend Code — Where to Look

### Sensor Hub (polls ESP32 for data)

| File | Key Section |
|------|-------------|
| `services/sensor_hub/__init__.py` | `get_sensor_hub()` — creates `RealSensorHub` when `PS3_USE_REAL_SENSOR=true` |
| `services/sensor_hub/real_sensor.py` | `_background_poll_loop()` — polls `GET /data` every 100ms in a background thread |
| `services/sensor_hub/real_sensor.py` | `_parse_exo_data()` — parses JSON into `ExoSensorData` model |
| `services/sensor_hub/real_sensor.py` | `send_command()` — sends `POST /command` with mode/torque |
| `services/sensor_hub/schemas.py` | `ExoSensorData`, `SensorReading`, `SensorCommand` — data models |

### ML Model (analyzes movement, predicts assist/resist)

| File | Key Section |
|------|-------------|
| `services/exercise_analysis/model_analyzer.py` | `analyze_rep()` — runs RehabNet inference on joint angles |
| `services/exercise_analysis/model_analyzer.py` | `PS2ModeCommand` output — `mode_id=1` (Assist) or `mode_id=2` (Resist) |
| `services/exercise_analysis/schemas.py` | `PS2ModeCommand` — `{mode_id, mode_name, target_torque}` |

### WebSocket Session (orchestrates the closed loop)

| File | Key Section |
|------|-------------|
| `ws_handlers/session_ws.py` | Lines 147-157 — Polls ESP32 sensor data via `poll_once()` |
| `ws_handlers/session_ws.py` | Lines 169-172 — Overrides camera angles with ESP32 hardware sensor data |
| `ws_handlers/session_ws.py` | Lines 220-229 — Runs ML analysis when a rep is detected |
| `ws_handlers/session_ws.py` | Lines 287-329 — Sends ML prediction back to ESP32 via `POST /exercise` |

---

## WiFi Setup

On first boot (or after BOOT button reset):

1. ESP32 creates a WiFi access point: **"Samarth-Exo-Setup"**
2. Connect your phone/laptop to this network
3. A captive portal opens with the Samarth-branded dark UI
4. Select your WiFi network and enter the password
5. ESP32 saves credentials and connects automatically on future boots
6. Hold the **BOOT button** (GPIO 0) during power-on to clear saved WiFi and reconfigure

---

## GPIO Pin Map

| GPIO | Function | Hardware |
|------|----------|----------|
| 0 | BOOT button (WiFi reset) | Built-in |
| 4 | KNEE_RPWM | BTS7960 Knee Motor Forward |
| 5 | KNEE_L_EN | BTS7960 Knee Motor Enable (Backward) |
| 12 | HIP_L_EN | Unused (Hip Motor Enable Backward disabled) |
| 13 | HIP_RPWM | Unused (Hip Motor Forward disabled) |
| 14 | HIP_LPWM | Unused (Hip Motor Backward disabled) |
| 15 | HIP_R_EN | Unused (Hip Motor Enable Forward disabled) |
| 16 | KNEE_LPWM | BTS7960 Knee Motor Backward |
| 17 | KNEE_R_EN | BTS7960 Knee Motor Enable (Forward) |
| 21 | SDA | I2C Data (MPU6050 x2) |
| 22 | SCL | I2C Clock (MPU6050 x2) |
| 34 | FSR_PIN | Foot Force Sensor (Analog) |

### I2C Addresses

| Address | Device |
|---------|--------|
| 0x68 | MPU6050 Thigh (AD0 = LOW) |
| 0x69 | MPU6050 Shank (AD0 = HIGH) |
