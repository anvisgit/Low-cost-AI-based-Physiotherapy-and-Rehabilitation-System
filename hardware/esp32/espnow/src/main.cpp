#include <esp_now.h>
#include <WiFi.h>
#include <Wire.h>
#include <MPU6050.h> // i2cdevlib MPU6050
#include <I2Cdev.h>


// Function Prototypes
void setupPWM();
void stopAllMotors();
void driveKneeMotor(int pwm, bool forward);
void driveHipMotor(int pwm, bool forward);
void readIMUs();
void readFSR();
void autonomousAssist();
void initESPNow();
void sendTelemetry();


// ==================== PIN DEFINITIONS ====================
#define KNEE_RPWM 4
#define KNEE_LPWM 16
#define KNEE_R_EN 17
#define KNEE_L_EN 5

#define HIP_RPWM 13
#define HIP_LPWM 14
#define HIP_R_EN 15
#define HIP_L_EN 12

#define FSR_PIN 34

#define PWM_FREQ 20000
#define PWM_RES 8
#define PWM_MAX 255

// ==================== MPU6050 OBJECTS ====================
MPU6050 mpu_thigh(0x68); // AD0 low
MPU6050 mpu_shank(0x69); // AD0 high

// Complementary filter coefficients
float thigh_pitch = 0, shank_pitch = 0;
float thigh_gyro_offset = 0, shank_gyro_offset = 0;
unsigned long last_imu_time = 0;

// ==================== TIMING ====================
unsigned long lastTelemetryTime = 0;
const int TELEMETRY_INTERVAL = 10; // 100 Hz
unsigned long lastControlTime = 0;
const int CONTROL_INTERVAL = 5; // 200 Hz

// ==================== SENSOR DATA ====================
float knee_angle = 0;
float hip_angle = 0;
float foot_force = 0;
bool stance_detected = false;

// ==================== MOTOR STATE ====================
bool motor_enabled = true;
int knee_pwm = 0;
bool knee_dir_forward = true;
int hip_pwm = 0;
bool hip_dir_forward = true;

// ==================== ESP-NOW ====================
uint8_t controllerMAC[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF}; // <-- SET THIS!

typedef struct {
  float knee_angle;
  float hip_angle;
  float foot_force;
  bool stance;
  int knee_pwm;
  bool knee_dir;
  int hip_pwm;
  bool hip_dir;
  bool motors_enabled;
} telemetry_t;

telemetry_t telemetry;

typedef struct {
  char cmd[32];
  int knee_pwm;
  bool knee_dir;
  int hip_pwm;
  bool hip_dir;
} command_t;

command_t incomingCommand;
void stopAllMotors() {
  ledcWrite(0, 0); ledcWrite(1, 0);
  ledcWrite(2, 0); ledcWrite(3, 0);
  knee_pwm = 0; hip_pwm = 0;
}

void driveKneeMotor(int pwm, bool forward) {
  pwm = constrain(pwm, 0, PWM_MAX);
  if (forward) {
    ledcWrite(0, pwm);
    ledcWrite(1, 0);
  } else {
    ledcWrite(0, 0);
    ledcWrite(1, pwm);
  }
}

void driveHipMotor(int pwm, bool forward) {
  pwm = constrain(pwm, 0, PWM_MAX);
  if (forward) {
    ledcWrite(2, pwm);
    ledcWrite(3, 0);
  } else {
    ledcWrite(2, 0);
    ledcWrite(3, pwm);
  }
}

void setupPWM() {
  ledcSetup(0, PWM_FREQ, PWM_RES);
  ledcAttachPin(KNEE_RPWM, 0);

  ledcSetup(1, PWM_FREQ, PWM_RES);
  ledcAttachPin(KNEE_LPWM, 1);

  ledcSetup(2, PWM_FREQ, PWM_RES);
  ledcAttachPin(HIP_RPWM, 2);

  ledcSetup(3, PWM_FREQ, PWM_RES);
  ledcAttachPin(HIP_LPWM, 3);
}

// Callbacks
void onDataSent(const uint8_t *mac, esp_now_send_status_t status) {}
void onDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
  memcpy(&incomingCommand, data, sizeof(incomingCommand));
  String cmd = String(incomingCommand.cmd);
  if (cmd == "disable") {
    motor_enabled = false;
    stopAllMotors();
  }
  else if (cmd == "enable") {
    motor_enabled = true;
  }
  else if (cmd == "stop") {
    stopAllMotors();
  }
  else if (cmd == "direct") {
    if (motor_enabled) {
      knee_pwm = incomingCommand.knee_pwm;
      knee_dir_forward = incomingCommand.knee_dir;
      hip_pwm = incomingCommand.hip_pwm;
      hip_dir_forward = incomingCommand.hip_dir;
      driveKneeMotor(knee_pwm, knee_dir_forward);
      driveHipMotor(hip_pwm, hip_dir_forward);
    }
  }
}

// ==================== IMU READING (i2cdevlib) ====================
// Simple complementary filter: pitch = 0.98 * (pitch + gyro*dt) + 0.02 * accel_pitch
void readMPU(MPU6050 &mpu, float &pitch, float &gyro_offset) {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Accelerometer pitch (roll around X axis, assuming Y is forward/up depending on mount)
  // We assume the sensor's Y axis points up along the leg, X forward, Z lateral.
  // Pitch (tilt forward/backward) = atan2(-ax, sqrt(ay^2 + az^2)) or atan2(ax, az) depending orientation.
  // We'll use atan2(ax, az) * 180/PI for pitch when sensor is mounted with Y vertical.
  // Standard: pitch = atan2(-ax, sqrt(ay*ay + az*az)) * RAD_TO_DEG;
  float acc_pitch = atan2(-ax, sqrt((float)ay*ay + (float)az*az)) * 180.0 / M_PI;

  // Gyroscope rate around X axis (deg/s)
  float gyro_rate = (gx - gyro_offset) / 131.0; // LSB/°/s for ±250°/s scale (default)

  // Time delta
  unsigned long now = micros();
  float dt = (now - last_imu_time) / 1000000.0;
  if (dt < 0) dt = 0.001;
  last_imu_time = now;

  // Complementary filter
  pitch = 0.98 * (pitch + gyro_rate * dt) + 0.02 * acc_pitch;
}

void calibrateIMU(MPU6050 &mpu, float &gyro_offset) {
  // Simple calibration: read 100 samples of gyro X and average
  long sum = 0;
  for (int i = 0; i < 100; i++) {
    int16_t gx, gy, gz;
    mpu.getRotation(&gx, &gy, &gz);
    sum += gx;
    delay(5);
  }
  gyro_offset = sum / 100.0;
}

void readIMUs() {
  readMPU(mpu_thigh, thigh_pitch, thigh_gyro_offset);
  readMPU(mpu_shank, shank_pitch, shank_gyro_offset);

  knee_angle = thigh_pitch - shank_pitch;
  hip_angle = thigh_pitch; // Simplified; ideally compute from pelvis IMU
  knee_angle = constrain(knee_angle, 0, 130);
  hip_angle = constrain(hip_angle, -30, 120);
}

// ==================== FSR ====================
void readFSR() {
  int raw = analogRead(FSR_PIN);
  float voltage = (raw / 4095.0) * 3.3;
  float fsr_resistance;
  if (voltage < 0.01) fsr_resistance = 10000000;
  else fsr_resistance = (10000.0 * (3.3 - voltage)) / voltage;
  if (fsr_resistance > 1000000) foot_force = 0;
  else {
    float conductance = 1.0 / fsr_resistance;
    foot_force = (conductance * 1e6) / 800.0;
  }
  static float filtered_force = 0;
  filtered_force = 0.9 * filtered_force + 0.1 * foot_force;
  foot_force = filtered_force;
  stance_detected = (foot_force > 2.0);
}

// ==================== AUTONOMOUS ASSIST ====================
void autonomousAssist() {
  if (!motor_enabled) return;
  if (stance_detected && knee_angle > 30) {
    knee_pwm = constrain((int)(2.0 * (knee_angle - 30)), 0, 150);
    knee_dir_forward = true;
  } else {
    knee_pwm = 0;
  }
  if (stance_detected && hip_angle > 20) {
    hip_pwm = constrain((int)(2.0 * (hip_angle - 20)), 0, 150);
    hip_dir_forward = true;
  } else {
    hip_pwm = 0;
  }
  driveKneeMotor(knee_pwm, knee_dir_forward);
  driveHipMotor(hip_pwm, hip_dir_forward);
}

// ==================== ESP-NOW INIT ====================
void initESPNow() {
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }
  esp_now_register_send_cb(onDataSent);
  esp_now_register_recv_cb(onDataRecv);
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, controllerMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
  }
}

void sendTelemetry() {
  telemetry.knee_angle = knee_angle;
  telemetry.hip_angle = hip_angle;
  telemetry.foot_force = foot_force;
  telemetry.stance = stance_detected;
  telemetry.knee_pwm = knee_pwm;
  telemetry.knee_dir = knee_dir_forward;
  telemetry.hip_pwm = hip_pwm;
  telemetry.hip_dir = hip_dir_forward;
  telemetry.motors_enabled = motor_enabled;
  esp_now_send(controllerMAC, (uint8_t*)&telemetry, sizeof(telemetry));
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000); // Fast I2C

  // Initialize IMUs
  mpu_thigh.initialize();
  mpu_shank.initialize();
  if (!mpu_thigh.testConnection() || !mpu_shank.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while(1);
  }
  // Set full scale gyro range ±250°/s (default) – keep as is
  mpu_thigh.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
  mpu_shank.setFullScaleGyroRange(MPU6050_GYRO_FS_250);

  // Calibrate gyro offsets
  Serial.println("Calibrating IMU gyros... keep sensors still");
  calibrateIMU(mpu_thigh, thigh_gyro_offset);
  calibrateIMU(mpu_shank, shank_gyro_offset);
  last_imu_time = micros();

  setupPWM();
  pinMode(KNEE_R_EN, OUTPUT); pinMode(KNEE_L_EN, OUTPUT);
  pinMode(HIP_R_EN, OUTPUT); pinMode(HIP_L_EN, OUTPUT);
  digitalWrite(KNEE_R_EN, HIGH); digitalWrite(KNEE_L_EN, HIGH);
  digitalWrite(HIP_R_EN, HIGH); digitalWrite(HIP_L_EN, HIGH);
  stopAllMotors();

  initESPNow();
  Serial.println("Exoskeleton ready. MAC: " + WiFi.macAddress());
}

// ==================== LOOP ====================
void loop() {
  unsigned long now = millis();
  readIMUs();
  readFSR();
  if (motor_enabled) {
    autonomousAssist();
  } else {
    stopAllMotors();
  }
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    sendTelemetry();
    lastTelemetryTime = now;
  }
  delay(1);
}
