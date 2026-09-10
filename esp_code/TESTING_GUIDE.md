# Samarth Unified ESP32 — Testing Guide

This guide walks you through flashing, connecting, and testing the **single unified ESP32** that handles sensors, motors, WiFi, and HTTP communication with the backend.

---

## Prerequisites

### Hardware
- **1× ESP32 DevKit V1** (or compatible)
- **2× MPU6050 IMU** (thigh + shank, I2C addresses 0x68 and 0x69)
- **1× BTS7960 Motor Driver** (knee motor only, hip is disabled in this hardware profile)
- **1× FSR (Force Sensitive Resistor)** on GPIO 34
- USB cable for flashing

### Software
- **Arduino IDE 2.x** (or PlatformIO)
- ESP32 board package installed (`esp32` by Espressif in Board Manager)
- Required libraries (install via Library Manager):
  - `WiFiManager` by tzapu
  - `ArduinoJson` by Benoit Blanchon (v7+)
  - `MPU6050` by Electronic Cats
  - `I2Cdev` by Jeff Rowberg

### CP2102 USB Driver
If your PC doesn't detect the ESP32, install the USB driver:
```
Double-click install_driver.bat (in the esp_code folder)
```

---

## Step 1: Flash the Firmware

1. Open `esp_code/unified_esp/unified_esp.ino` in Arduino IDE
2. Select Board: **ESP32 Dev Module**
3. Select the correct COM port
4. Click **Upload** (→ button)
5. Open Serial Monitor at **115200 baud**

You should see:
```text
======================================
  SAMARTH Unified Exoskeleton ESP32  
======================================
[INFO] Calibrating IMU gyros... keep sensors still
[SUCCESS] IMU calibration complete
[SUCCESS] Motors initialized (stopped)
Connecting to WiFi (or starting setup portal)...
```

> **Important**: Keep the device perfectly still during the IMU calibration phase (~1 second).

---

## Step 2: Connect to WiFi

### First Time Setup
1. On your phone/laptop, connect to WiFi network: **`Samarth-Exo-Setup`**
2. A captive portal will open automatically (dark Samarth-branded UI)
3. Select your WiFi network from the list
4. Enter your WiFi password
5. The ESP32 saves credentials and connects

### Subsequent Boots
The ESP32 auto-connects to the saved WiFi network.

### Reset WiFi Settings
Hold the **BOOT button** (GPIO 0) while powering on the ESP32 to clear saved WiFi credentials and re-enter setup mode.

---

## Step 3: Find the ESP32 IP Address

After connecting, the Serial Monitor shows:
```text
[SUCCESS] Connected to WiFi. IP: 192.168.x.x
```

Note this IP address — you'll need it for the backend `.env` configuration.

---

## Step 4: Test HTTP Endpoints

### Quick test with curl or browser

**Check status:**
```bash
curl http://192.168.x.x/status
```
Expected response:
```json
{
  "device_id": "ESP32-SAMARTH-EXO",
  "battery_percent": 100,
  "calibration_status": "calibrated",
  "connected": true,
  "ip": "192.168.x.x",
  "channel": 6,
  "uptime_ms": 12345,
  "current_mode": 0,
  "current_torque": 0.0
}
```

**Get sensor data:**
```bash
curl http://192.168.x.x/data
```
Expected response:
```json
{
  "knee_angle": 75.3,
  "hip_angle": 45.1,
  "foot_force": 3.42,
  "stance": true,
  "knee_pwm": 0,
  "knee_dir": "ext",
  "hip_pwm": 0,
  "hip_dir": "ext",
  "motors": "off",
  "battery_percent": 100,
  "calibration_status": "calibrated",
  "connected": true
}
```

**Send ML prediction (simulate assist mode):**
```bash
curl -X POST http://192.168.x.x/command -H "Content-Type: application/json" -d "{\"mode_id\": 1, \"target_torque\": 1.0}"
```
Expected response:
```json
{"status": "ok"}
```

The Serial Monitor should show:
```text
[ML] prediction received: Mode=ASSIST, Torque=1.00Nm
```

**Stop motors:**
```bash
curl -X POST http://192.168.x.x/command -H "Content-Type: application/json" -d "{\"mode_id\": 0, \"target_torque\": 0.0}"
```

---

## Step 5: Test with Python Script

The `test_esp_communication.py` script can test the full feedback loop:

```bash
cd esp_code
python test_esp_communication.py
```

Make sure to update the ESP_URL in the script to match your ESP32's IP address.

---

## Step 6: Connect Backend

1. Edit `backend/.env`:
   ```
   PS3_USE_REAL_SENSOR=true
   PS3_ESP_URL=http://192.168.x.x
   ```
   Replace `192.168.x.x` with your ESP32's actual IP.

2. Start the backend:
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```

3. The backend will:
   - Poll `GET /data` every 100ms for sensor readings
   - During live sessions, send ML predictions back via `POST /command`
   - The closed loop runs automatically

---

## Step 7: Verify the Closed Loop

During a live exercise session:

1. Start a session from the frontend
2. The camera captures your movement (PS1 pose estimation)
3. ESP32 sensor data overlays onto camera angles
4. When a rep is detected, the ML model analyzes it
5. The model predicts: **ASSIST** or **RESIST** with a target torque
6. The backend sends this prediction to the ESP32
7. The ESP32 drives the motors accordingly
8. The cycle repeats

Watch the Serial Monitor for:
```text
[ML] prediction received: Mode=ASSIST, Torque=1.20Nm
[ML] prediction received: Mode=RESIST, Torque=2.50Nm
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ESP32 not detected on USB | Install CP2102 driver (`install_driver.bat`) |
| WiFi portal doesn't appear | Hold BOOT button during power-on to reset WiFi |
| `GET /data` times out | Check ESP32 and laptop are on same WiFi network |
| Motors don't respond to commands | Check motor driver wiring, ensure enable pins are HIGH |
| IMU readings are wrong | Ensure sensors are wired to correct I2C addresses (0x68, 0x69) |
| Backend shows "ESP32 not connected" | Verify `PS3_ESP_URL` in `.env` matches ESP32 IP |
| Angles stuck at 0 | Keep device still during boot for gyro calibration |
