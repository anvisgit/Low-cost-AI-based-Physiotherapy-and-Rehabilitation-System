/*
  ===========================================================================
  SAMARTH UNIFIED ESP32 — Single Board Exoskeleton Controller
  ===========================================================================
  
  This single ESP32 handles EVERYTHING:
  1. Reads joint angles from two MPU6050 IMU sensors (thigh + shank).
  2. Reads foot pressure from a Force Sensor (FSR).
  3. Drives two motor controllers (BTS7960) for knee and hip assistance.
  4. Connects to WiFi using a captive portal (WiFiManager).
  5. Hosts an HTTP web server for the backend to:
     - Poll sensor data  (GET /data, GET /telemetry)
     - Check device status (GET /status)
     - Send ML model predictions — assist or resist — (POST /command, POST /exercise)
     - Trigger calibration (POST /calibrate)

  ═══════════════════════════════════════════════════════════════════════
  CLOSED LOOP DATA FLOW:
  
    Sensors (IMU, FSR) ──► ESP32 serves GET /data ──► Backend polls data
                                                          │
                                                          ▼
                                                     ML Model (RehabNet)
                                                     uses camera + sensor data
                                                     predicts: ASSIST or RESIST
                                                          │
                                                          ▼
    Motors apply torque ◄── ESP32 receives POST /command ◄─┘
    
    This loop runs continuously during a live session.
  ═══════════════════════════════════════════════════════════════════════

  How to configure:
  - On first boot, connect to the "Samarth-Exo-Setup" WiFi hotspot.
  - Open 192.168.4.1 in your browser to pick your WiFi network.
  - Hold the BOOT button (GPIO 0) during power-on to reset WiFi settings.
  ===========================================================================
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <WiFiManager.h>
#include <Wire.h>
#include <MPU6050.h>
#include <I2Cdev.h>
#include <ESPmDNS.h>

// ==================== CONFIGURATION ====================
// Pin definition for the BOOT button on the ESP32 (GPIO 0)
// Hold down this button while powering on to reset WiFi settings.
#define WIFI_RESET_PIN 0

// ==================== MOTOR PIN DEFINITIONS ====================
// Pins to control the Knee motor speed and direction (BTS7960 driver)
#define KNEE_RPWM  4   // Forward speed control pin
#define KNEE_LPWM  16  // Backward speed control pin
#define KNEE_R_EN  17  // Enable forward drive (High = Active)
#define KNEE_L_EN  5   // Enable backward drive (High = Active)

// Pins to control the Hip motor speed and direction (BTS7960 driver)
#define HIP_RPWM   13  // Forward speed control pin
#define HIP_LPWM   14  // Backward speed control pin
#define HIP_R_EN   15  // Enable forward drive (High = Active)
#define HIP_L_EN   12  // Enable backward drive (High = Active)

// Sensor pin for the foot force sensor
#define FSR_PIN    34  // Analog pin A0 / GPIO 34

// Motor PWM settings (High frequency to avoid motor whining noise)
#define PWM_FREQ   20000 // 20 kHz
#define PWM_RES    8     // 8-bit resolution (speed values from 0 to 255)
#define PWM_MAX    255   // Max speed value

// ==================== TORQUE -> MOTOR SPEED SCALING ====================
// Target torque is received as a decimal number (0.0 to 2.0 Nm) from the ML model.
// The scale factor converts this torque value into a motor speed (0 to 255 PWM).
#define TORQUE_TO_PWM_SCALE 150.0
#define MAX_TORQUE 2.0

// ==================== SENSOR OBJECTS ====================
// MPU6050 Accelerometer/Gyroscope sensors
MPU6050 mpu_thigh(0x68);    // Thigh sensor (I2C address 0x68)
MPU6050 mpu_shank(0x69);    // Shank/Calf sensor (I2C address 0x69 - AD0 pin set HIGH)

// Raw angles and offset variables
float thigh_pitch = 0, shank_pitch = 0;
float thigh_gyro_offset = 0, shank_gyro_offset = 0;
unsigned long last_imu_time = 0;

// ==================== TIMER INTERVALS ====================
unsigned long lastTelemetryTime = 0;
const int TELEMETRY_INTERVAL = 10;    // Internal telemetry update at 100 Hz (every 10ms)
unsigned long lastControlTime = 0;
const int CONTROL_INTERVAL = 5;       // Adjust motor speeds at 200 Hz (every 5ms)

// ==================== CURRENT EXOSKELETON STATE ====================
float knee_angle = 0;         // Calculated knee angle in degrees (0 to 130)
float hip_angle = 0;          // Calculated hip angle in degrees (-30 to 120)
float foot_force = 0;         // Foot force in Newtons
bool stance_detected = false; // True when patient is stepping on the ground
bool calibration_done = false;// True after IMUs are calibrated on boot

// ==================== MOTOR STATE ====================
bool motor_enabled = false;
int knee_pwm = 0;
bool knee_dir_forward = true;
int hip_pwm = 0;
bool hip_dir_forward = true;

int current_mode_id = 0;         // Current Mode: 0 = Off, 1 = Assist, 2 = Resist
float current_target_torque = 0.0;

// ==================== HTTP WEB SERVER ====================
// Create local Web Server on standard HTTP port 80
WebServer server(80);

// ==================== MOTOR CONTROL FUNCTIONS ====================
// Initialize the PWM hardware to control motor speeds (Arduino-ESP32 v3.x compatible)
void setupPWM() {
  ledcAttach(KNEE_RPWM, PWM_FREQ, PWM_RES);
  ledcAttach(KNEE_LPWM, PWM_FREQ, PWM_RES);
  // Hip motor actuator disabled on this hardware configuration
  // ledcAttach(HIP_RPWM,  PWM_FREQ, PWM_RES);
  // ledcAttach(HIP_LPWM,  PWM_FREQ, PWM_RES);
}

// Stop both knee and hip motors instantly
void stopAllMotors() {
  ledcWrite(KNEE_RPWM, 0); ledcWrite(KNEE_LPWM, 0);
  // Hip motor actuator disabled on this hardware configuration
  // ledcWrite(HIP_RPWM,  0); ledcWrite(HIP_LPWM,  0);
  knee_pwm = 0; hip_pwm = 0;
}

// Drive the Knee Motor at a specific speed (0 to 255) and direction
void driveKneeMotor(int pwm, bool forward) {
  pwm = constrain(pwm, 0, PWM_MAX);
  if (forward) { 
    ledcWrite(KNEE_RPWM, pwm); 
    ledcWrite(KNEE_LPWM, 0); 
  } else { 
    ledcWrite(KNEE_RPWM, 0);   
    ledcWrite(KNEE_LPWM, pwm); 
  }
}

// Drive the Hip Motor - disabled on this hardware configuration
void driveHipMotor(int pwm, bool forward) {
  hip_pwm = 0;
}

// ==================== SENSOR: READ joint angles (IMUs) ====================
// Read accelerometer/gyroscope raw data, apply complementary filter to find pitch
void readMPU(MPU6050 &mpu, float &pitch, float &gyro_offset) {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Pitch calculation using trigonometry
  float acc_pitch = atan2(-ax, sqrt((float)ay*ay + (float)az*az)) * 180.0 / M_PI;
  float gyro_rate = (gx - gyro_offset) / 131.0;

  unsigned long now = micros();
  float dt = (now - last_imu_time) / 1000000.0;
  if (dt < 0) dt = 0.001; // Avoid divide by zero
  last_imu_time = now;

  // Complementary filter: combines 98% Gyro (fast) + 2% Accel (stable) to find true angle
  pitch = 0.98 * (pitch + gyro_rate * dt) + 0.02 * acc_pitch;
}

// Keep the sensor perfectly still on boot to record baseline offset values
void calibrateIMU(MPU6050 &mpu, float &gyro_offset) {
  long sum = 0;
  for (int i = 0; i < 100; i++) {
    int16_t gx, gy, gz;
    mpu.getRotation(&gx, &gy, &gz);
    sum += gx;
    delay(5);
  }
  gyro_offset = sum / 100.0;
}

// Read both IMU sensors and calculate the joint angles
void readIMUs() {
  if (!calibration_done) {
    knee_angle = 0.0;
    hip_angle = 0.0;
    return;
  }
  readMPU(mpu_thigh, thigh_pitch, thigh_gyro_offset);
  readMPU(mpu_shank, shank_pitch, shank_gyro_offset);

  // Knee angle is the difference between Thigh angle and Calf/Shank angle
  knee_angle = thigh_pitch - shank_pitch;
  hip_angle = thigh_pitch;
  
  // Safe bounds check
  knee_angle = constrain(knee_angle, 0, 130);
  hip_angle = constrain(hip_angle, -30, 120);
}

// ==================== SENSOR: READ Foot Pressure Sensor (FSR) ====================
// Read FSR voltage, calculate electrical resistance, and convert to force in Newtons
void readFSR() {
  int raw = analogRead(FSR_PIN);
  float voltage = (raw / 4095.0) * 3.3; // Convert 12-bit ADC reading to Voltage
  float fsr_resistance;
  
  if (voltage < 0.01) {
    fsr_resistance = 10000000;
  } else {
    fsr_resistance = (10000.0 * (3.3 - voltage)) / voltage;
  }

  if (fsr_resistance > 1000000) {
    foot_force = 0; // No pressure
  } else {
    float conductance = 1.0 / fsr_resistance;
    foot_force = (conductance * 1e6) / 800.0; // Estimate force
  }
  
  // Simple low-pass filter to smooth out sensor spikes/noise
  static float filtered_force = 0;
  filtered_force = 0.9 * filtered_force + 0.1 * foot_force;
  foot_force = filtered_force;
  
  // If force is greater than 2 Newtons, detect that foot is on the ground
  stance_detected = (foot_force > 2.0);
}

// ==================== BATTERY level (Placeholder) ====================
float readBatteryPercent() {
  return 100.0; // Default to 100%. Replace with an ADC divider reading later.
}

// ==================== MAIN MOTOR CONTROL / MODES ====================
// Apply the assistive/resistive motor force based on ML model prediction.
// The ML model sends: mode_id (1=Assist, 2=Resist) and target_torque via POST /command.
void applyControl() {
  // Mode 0: Safe Stop
  if (current_mode_id == 0) {
    motor_enabled = false;
    stopAllMotors();
    return;
  }

  motor_enabled = true;
  float torque = constrain(current_target_torque, 0.0, MAX_TORQUE);

  // Mode 1: Gait-triggered assist (Assistive)
  // Only push when the user's foot is on the ground (stance phase) and joint exceeds thresholds
  if (current_mode_id == 1) {
    if (stance_detected && knee_angle > 30) {
      // Calculate speed scaled by current angle and target torque
      knee_pwm = constrain((int)(torque * TORQUE_TO_PWM_SCALE * (knee_angle - 30) / 90.0), 0, PWM_MAX);
      knee_dir_forward = true;
    } else {
      knee_pwm = 0;
    }
    // Hip motor is disabled on this hardware configuration
    hip_pwm = 0;
    hip_dir_forward = true;
  } 
  // Mode 2: Constant Assist / Resistive Torque
  // Pushes constantly, ignoring whether foot is on the ground or not
  else if (current_mode_id == 2) {
    knee_pwm = constrain((int)(torque * TORQUE_TO_PWM_SCALE), 0, PWM_MAX);
    knee_dir_forward = true;
    // Hip motor is disabled on this hardware configuration
    hip_pwm = 0;
    hip_dir_forward = true;
  } 
  // Unknown mode: Safe Stop
  else {
    knee_pwm = 0;
    hip_pwm = 0;
  }

  driveKneeMotor(knee_pwm, knee_dir_forward);
  // Hip motor is disabled
  // driveHipMotor(hip_pwm, hip_dir_forward);
}

// ==================== AUTO WiFi CONNECTION (WiFiManager) ====================
void initWiFi() {
  WiFiManager wm;

  const char* custom_html = 
    "<style>"
    "* { margin: 0; padding: 0; box-sizing: border-box; }"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"

    "body { "
      "background: linear-gradient(145deg, #0A1628 0%, #0F2038 40%, #122A45 100%) !important; "
      "font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; "
      "color: #E2E8F0 !important; "
      "padding: 30px 15px !important; "
      "display: flex !important; flex-direction: column !important; "
      "align-items: center !important; justify-content: center !important; "
      "min-height: 100vh !important; "
    "}"

    // Logo area
    "body::before { "
      "content: '⚕'; "
      "display: block; "
      "font-size: 48px; "
      "margin-bottom: 8px; "
      "filter: drop-shadow(0 0 12px rgba(34,144,150,0.4)); "
    "}"

    "h1 { "
      "background: linear-gradient(135deg, #2DD4BF, #229096, #14B8A6) !important; "
      "-webkit-background-clip: text !important; "
      "-webkit-text-fill-color: transparent !important; "
      "font-size: 32px !important; font-weight: 800 !important; "
      "text-align: center !important; "
      "margin-bottom: 4px !important; "
      "letter-spacing: -0.5px !important; "
    "}"

    "h3 { "
      "color: #94A3B8 !important; "
      "font-size: 13px !important; text-align: center !important; "
      "margin-top: 0 !important; margin-bottom: 24px !important; "
      "font-weight: 500 !important; letter-spacing: 2px !important; "
      "text-transform: uppercase !important; "
    "}"

    // Main card
    "div.wrap { "
      "background: rgba(15, 23, 42, 0.8) !important; "
      "backdrop-filter: blur(20px) !important; "
      "-webkit-backdrop-filter: blur(20px) !important; "
      "padding: 35px 28px !important; "
      "border-radius: 24px !important; "
      "box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5), "
        "0 0 0 1px rgba(34,144,150,0.15), "
        "inset 0 1px 0 0 rgba(255,255,255,0.05) !important; "
      "border: 1px solid rgba(34, 144, 150, 0.2) !important; "
      "width: 100% !important; max-width: 420px !important; "
      "margin: 0 auto !important; "
    "}"

    // Buttons
    "button, input[type='submit'] { "
      "background: linear-gradient(135deg, #229096 0%, #1A7A7F 100%) !important; "
      "color: #FFFFFF !important; "
      "border: none !important; "
      "padding: 15px !important; "
      "border-radius: 14px !important; "
      "font-weight: 600 !important; font-size: 15px !important; "
      "cursor: pointer !important; "
      "width: 100% !important; "
      "margin: 12px 0 6px 0 !important; "
      "box-shadow: 0 4px 15px 0 rgba(34, 144, 150, 0.35), "
        "inset 0 1px 0 0 rgba(255,255,255,0.15) !important; "
      "transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important; "
      "letter-spacing: 0.3px !important; "
    "}"
    "button:hover, input[type='submit']:hover { "
      "background: linear-gradient(135deg, #1B7A7F 0%, #156568 100%) !important; "
      "transform: translateY(-2px) !important; "
      "box-shadow: 0 8px 25px 0 rgba(34, 144, 150, 0.45), "
        "inset 0 1px 0 0 rgba(255,255,255,0.15) !important; "
    "}"
    "button:active, input[type='submit']:active { "
      "transform: translateY(0px) !important; "
    "}"

    // Input fields
    "input[type='text'], input[type='password'] { "
      "background: rgba(15, 23, 42, 0.6) !important; "
      "border: 1px solid rgba(148, 163, 184, 0.2) !important; "
      "padding: 14px 16px !important; "
      "border-radius: 12px !important; "
      "font-size: 14px !important; "
      "color: #E2E8F0 !important; "
      "margin-bottom: 14px !important; "
      "width: 100% !important; "
      "outline: none !important; "
      "transition: all 0.2s !important; "
    "}"
    "input[type='text']:focus, input[type='password']:focus { "
      "border-color: #229096 !important; "
      "box-shadow: 0 0 0 3px rgba(34, 144, 150, 0.2), "
        "0 0 20px rgba(34, 144, 150, 0.1) !important; "
      "background: rgba(15, 23, 42, 0.8) !important; "
    "}"
    "input[type='text']::placeholder, input[type='password']::placeholder { "
      "color: #64748B !important; "
    "}"

    // WiFi network list items
    "div.q { "
      "border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important; "
      "padding: 14px 10px !important; "
      "display: flex !important; "
      "justify-content: space-between !important; "
      "align-items: center !important; "
      "transition: background 0.2s !important; "
      "border-radius: 8px !important; "
      "margin: 2px 0 !important; "
    "}"
    "div.q:hover { "
      "background: rgba(34, 144, 150, 0.08) !important; "
    "}"
    "div.q a { "
      "color: #E2E8F0 !important; "
      "text-decoration: none !important; "
      "font-weight: 600 !important; "
      "font-size: 14px !important; "
      "transition: color 0.2s !important; "
    "}"
    "div.q a:hover { color: #2DD4BF !important; }"

    // Info / status messages
    "div.msg { "
      "background: rgba(34, 144, 150, 0.1) !important; "
      "color: #2DD4BF !important; "
      "padding: 14px 16px !important; "
      "border-radius: 12px !important; "
      "font-size: 13px !important; "
      "border: 1px solid rgba(34, 144, 150, 0.2) !important; "
      "margin-bottom: 18px !important; "
      "line-height: 1.6 !important; "
    "}"

    "a { color: #2DD4BF !important; text-decoration: none !important; font-weight: 600 !important; }"

    // Signal strength indicators
    "div.q div { color: #94A3B8 !important; font-size: 12px !important; }"
    "</style>";

  wm.setCustomHeadElement(custom_html);

  // Set up the physical BOOT pin as input with internal pullup resistor
  pinMode(WIFI_RESET_PIN, INPUT_PULLUP);
  delay(100); // debounce delay
  
  // If BOOT button is held down during startup, clear saved WiFi configurations
  if (digitalRead(WIFI_RESET_PIN) == LOW) {
    Serial.println("[WARNING] BOOT button held -- clearing saved WiFi credentials...");
    wm.resetSettings();
  }

  // Tries to auto-connect using last saved credentials.
  // If it can't, it starts an Access Point called "Samarth-Exo-Setup"
  wm.setConfigPortalTimeout(180);  // Close configuration portal after 3 minutes
  wm.setConnectTimeout(8);         // Limit saved Wi-Fi search to 8 seconds before starting setup portal AP

  Serial.println("Connecting to WiFi (or starting setup portal)...");
  if (!wm.autoConnect("Samarth-Exo-Setup")) {
    Serial.println("[ERROR] WiFi setup timed out. Restarting...");
    ESP.restart();
  }

  Serial.println();
  Serial.print("[SUCCESS] Connected to WiFi. IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("WiFi channel: ");
  Serial.println(WiFi.channel());
}

// ==================== HTTP WEB SERVER HANDLERS ====================

// 1. POST /command — Receive ML model prediction (assist/resist + torque)
//    and apply it directly to the motors.
//    This is the primary endpoint for the closed-loop control.
//    Expected JSON: {"mode_id": 1, "target_torque": 1.2}
//    mode_id: 0 = Off, 1 = Assist, 2 = Resist
void handleCommand() {
  if (server.method() != HTTP_POST) {
    server.send(405, "application/json", "{\"error\":\"method not allowed\"}");
    return;
  }

  String body = server.arg("plain");
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) {
    server.send(400, "application/json", "{\"error\":\"invalid json\"}");
    return;
  }

  int mode_id = doc["mode_id"] | 0;
  float target_torque = doc["target_torque"] | 0.0f;

  // Apply ML prediction directly to motor control state
  current_mode_id = mode_id;
  current_target_torque = target_torque;

  Serial.print("[ML] prediction received: Mode=");
  Serial.print(mode_id == 1 ? "ASSIST" : (mode_id == 2 ? "RESIST" : "OFF"));
  Serial.print(", Torque=");
  Serial.print(target_torque);
  Serial.println("Nm");

  server.send(200, "application/json", "{\"status\":\"ok\"}");
}

// 2. POST /exercise — Legacy alias for /command (backwards compatibility)
void handleExercise() {
  handleCommand();
}

// 3. GET /data — Returns latest sensor readings and motor state.
//    This is the primary endpoint polled by the backend RealSensorHub.
void handleData() {
  JsonDocument doc;
  doc["knee_angle"]         = knee_angle;
  doc["hip_angle"]          = hip_angle;
  doc["foot_force"]         = foot_force;
  doc["stance"]             = stance_detected;
  doc["knee_pwm"]           = knee_pwm;
  doc["knee_dir"]           = knee_dir_forward ? "ext" : "flex";
  doc["hip_pwm"]            = hip_pwm;
  doc["hip_dir"]            = hip_dir_forward ? "ext" : "flex";
  doc["motors"]             = motor_enabled ? "on" : "off";
  doc["battery_percent"]    = (int)readBatteryPercent();
  doc["calibration_status"] = calibration_done ? "calibrated" : "uncalibrated";
  doc["connected"]          = true;

  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

// 4. GET /telemetry — Legacy raw telemetry endpoint
void handleTelemetry() {
  JsonDocument doc;
  doc["battery_percent"]    = (int)readBatteryPercent();
  doc["calibration_status"] = calibration_done;
  doc["knee_angle"]         = knee_angle;
  doc["hip_angle"]          = hip_angle;
  doc["foot_force"]         = foot_force;
  doc["stance"]             = stance_detected;
  doc["knee_motor_pwm"]     = knee_pwm;
  doc["knee_motor_dir"]     = knee_dir_forward;
  doc["hip_motor_pwm"]      = hip_pwm;
  doc["hip_motor_dir"]      = hip_dir_forward;
  doc["motors_active"]      = motor_enabled;
  doc["connected"]          = true;

  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

// 5. GET /status — Device info and health check
void handleStatus() {
  JsonDocument doc;
  doc["device_id"]          = "ESP32-SAMARTH-EXO";
  doc["battery_percent"]    = (int)readBatteryPercent();
  doc["calibration_status"] = calibration_done ? "calibrated" : "uncalibrated";
  doc["connected"]          = true;
  doc["ip"]                 = WiFi.localIP().toString();
  doc["channel"]            = WiFi.channel();
  doc["uptime_ms"]          = millis();
  doc["current_mode"]       = current_mode_id;
  doc["current_torque"]     = current_target_torque;

  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

// 6. POST /calibrate — Calibration endpoint
void handleCalibrate() {
  JsonDocument doc;
  doc["calibration_status"] = calibration_done ? "calibrated" : "uncalibrated";
  doc["message"] = "Calibration is performed automatically on boot. Restart the device to recalibrate.";

  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}

// ==================== SETUP (RUNS ONCE ON POWER UP) ====================
void setup() {
  Serial.begin(115200);
  // Wait up to 2 seconds for Serial to initialize (useful for native USB boards)
  for (int i = 0; i < 20 && !Serial; i++) {
    delay(100);
  }
  Serial.println();
  Serial.println("======================================");
  Serial.println("  SAMARTH Unified Exoskeleton ESP32  ");
  Serial.println("======================================");

  // Start I2C communication (Pins: SDA=21, SCL=22)
  Wire.begin(21, 22);
  Wire.setClock(400000); // 400kHz fast I2C mode

  // Initialize both IMUs
  mpu_thigh.initialize();
  mpu_shank.initialize();
  bool thigh_ok = mpu_thigh.testConnection();
  bool shank_ok = mpu_shank.testConnection();
  
  if (!thigh_ok || !shank_ok) {
    Serial.println("[WARNING] MPU6050 connection failed! Check wiring.");
    if (!thigh_ok) Serial.println("[WARNING] Thigh IMU (0x68) not detected.");
    if (!shank_ok) Serial.println("[WARNING] Shank IMU (0x69) not detected.");
    calibration_done = false;
  } else {
    mpu_thigh.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    mpu_shank.setFullScaleGyroRange(MPU6050_GYRO_FS_250);

    // Calibrate Gyro scopes. KEEP DEVICE STILL during boot.
    Serial.println("[INFO] Calibrating IMU gyros... keep sensors still");
    calibrateIMU(mpu_thigh, thigh_gyro_offset);
    calibrateIMU(mpu_shank, shank_gyro_offset);
    last_imu_time = micros();
    calibration_done = true;
    Serial.println("[SUCCESS] IMU calibration complete");
  }

  // Configure Motor speed enable pins as outputs
  setupPWM();
  pinMode(KNEE_R_EN, OUTPUT); pinMode(KNEE_L_EN, OUTPUT);
  // Hip motor pins disabled on this hardware configuration
  // pinMode(HIP_R_EN, OUTPUT);  pinMode(HIP_L_EN, OUTPUT);
  digitalWrite(KNEE_R_EN, HIGH); digitalWrite(KNEE_L_EN, HIGH);
  // digitalWrite(HIP_R_EN, HIGH);  digitalWrite(HIP_L_EN, HIGH);
  stopAllMotors(); // Ensure motors don't spin on boot
  Serial.println("[SUCCESS] Motors initialized (stopped)");

  // Connect to WiFi (captive portal if needed)
  initWiFi();

  // Start mDNS responder
  if (MDNS.begin("samarth-exo")) {
    Serial.println("[SUCCESS] mDNS responder started: http://samarth-exo.local");
  } else {
    Serial.println("[WARNING] Error setting up MDNS responder!");
  }

  // Register HTTP endpoint routes
  server.on("/command",   HTTP_POST, handleCommand);
  server.on("/exercise",  HTTP_POST, handleExercise);
  server.on("/data",      HTTP_GET,  handleData);
  server.on("/telemetry", HTTP_GET,  handleTelemetry);
  server.on("/status",    HTTP_GET,  handleStatus);
  server.on("/calibrate", HTTP_POST, handleCalibrate);
  server.begin();

  Serial.println();
  Serial.println("======================================");
  Serial.print("  IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("  MAC Address: ");
  Serial.println(WiFi.macAddress());
  Serial.println("  HTTP server on port 80");
  Serial.println("  Waiting for ML predictions...");
  Serial.println("======================================");
}

// ==================== LOOP (RUNS REPEATEDLY) ====================
void loop() {
  unsigned long now = millis();

  // 1. Read sensors (runs every iteration — fast)
  readIMUs();
  readFSR();

  // 2. Apply motor control based on latest ML prediction (200 Hz)
  if (now - lastControlTime >= CONTROL_INTERVAL) {
    applyControl();
    lastControlTime = now;
  }

  // 3. Handle incoming HTTP requests from the backend
  //    (GET /data polls, POST /command with ML predictions)
  server.handleClient();

  delay(1); // Give ESP32 CPU a break for internal WiFi/system processes
}
