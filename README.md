<div align="center">

<img src="logo.jpeg" alt="SAMARTH Logo" width="160" />

# SAMARTH

### Smart Augmented Mobility & Adaptive Rehabilitation Technologies for Humans

**An AI-guided physiotherapy rehabilitation platform for real-time biomechanical analysis, kinematic monitoring, and clinical outcome tracking**

*Developed at IIT (BHU) Varanasi - Physiotherapy · Machine Learning · Impact*

---

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb)](https://mongodb.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-0097A7?style=flat-square&logo=google)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [How It Works](#how-it-works)
4. [Technology Stack](#technology-stack)
5. [Database Design](#database-design)
6. [WebSocket Protocol](#websocket-protocol)
7. [Pipeline Modules (PS1 / PS2 / PS3)](#pipeline-modules)
8. [API Reference](#api-reference)
9. [Repository Structure](#repository-structure)
10. [Installation & Setup](#installation--setup)
11. [Environment Variables](#environment-variables)
12. [Docker Deployment](#docker-deployment)
13. [Integrating PS2 and PS3](#integrating-ps2-and-ps3)

---

## Overview

SAMARTH is a full-stack, real-time physiotherapy rehabilitation platform that fuses **computer vision**, **biomechanical kinematics**, and **clinical data management** into a single cohesive system. It is designed for use in hospitals, clinics, and home-based telerehabilitation settings.

The system enables:

- **Patients** to perform assigned rehabilitation exercises under real-time AI supervision, receiving instant form feedback without any wearable hardware.
- **Therapists** to remotely monitor patient adherence, review biomechanical analytics across sessions, and generate clinical PDF reports.
- **Researchers** to plug in ML models (PS2) and embedded sensor firmware (PS3) through clearly defined integration contracts without touching the core platform.

The platform is built around the team's **PS1 KinemaFlow** computer vision pipeline as its primary sensing engine, with purpose-built adapter layers for the upcoming PS2 (exercise analysis ML model) and PS3 (BLE/IMU wearable sensor) modules.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Patient Browser                            │
│  React 18 + TypeScript  ·  Vite 5  ·  Tailwind CSS  ·  Recharts     │
│                                                                     │
│   Camera Feed ──► MediaStream API ──► JPEG Frames                   │
│   WebSocket Client (live session + camera validation)               │
│   REST Client (axios, /api/v1/*)                                    │
└────────────────────────┬───────────────────┬────────────────────────┘
                         │ HTTP / WS         │ HTTP / WS
                         ▼                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Uvicorn/ASGI)                 │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  REST Routes │  │  WS Handlers │  │  Auth (JWT HS256)        │  │
│  │  /api/v1/*   │  │  /ws/session │  │  Access + Refresh Tokens │  │
│  │              │  │  /ws/camera  │  │                          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────┘  │
│         │                 │                                        │
│         ▼                 ▼                                        │
│  ┌──────────────────────────────────────────────┐                  │
│  │           PoseEngineAdapter (Singleton)       │                 │
│  │  The ONLY gateway to the PS1 pipeline.        │                 │
│  │  Manages per-session PoseEstimator lifecycle. │                 │
│  └──────────────────┬───────────────────────────┘                  │
│                     │ sys.path injection                           │
└─────────────────────┼──────────────────────────────────────────────┘
                      ▼
┌────────────────────────────────────────────────────────────────────┐
│                   PS1 - KinemaFlow Pipeline                        │
│                                                                    │
│  BackgroundSegmenter  ·  VideoEnhancer  ·  PoseEstimator (MP)      │
│  KinematicsExtractor  ·  FeatureExtractor  ·  FilterUtils          │
│                                                                    │
│  Real-time mode: per-JPEG-frame inference                          │
│  Batch mode:     run_pipeline_on_video() → CSV + angle timeseries  │
└────────────────────────────────────────────────────────────────────┘
                      │
         ┌────────────┴──────────────┐
         ▼                           ▼
┌─────────────────┐       ┌─────────────────────┐
│ PS2 (stub/real) │       │ PS3 (mock/real)     │
│ ML Exercise     │       │ ESP32 Exo Driver    │
│ Analyzer        │       │                     │
│ model_analyzer  │       │ real_sensor.py      │
│ .py             │       │ (Wi-Fi HTTP status  │
└─────────────────┘       │  /data/command)     │
         │                └─────────────────────┘
         │                           │
         ▼                           ▼
┌───────────────────────────────────────────────┐
│        MongoDB Atlas (Motor + Beanie)         │
│            14 document collections            │
└───────────────────────────────────────────────┘
```

---

## How It Works

### 1. Camera Validation (Pre-Session)

Before every exercise session, the patient enters the **Camera Validation** screen. This screen opens a dedicated WebSocket channel (`/ws/camera/{token}`) and begins streaming JPEG frames from the browser's webcam to the backend at approximately 10–15 fps.

The `PoseEngineAdapter` runs PS1's `PoseEstimator` on each frame and evaluates **10 real-time checks**:

| Check | Description |
|---|---|
| Left hip visible | Landmark visibility ≥ 0.25 threshold (relaxed for indoor webcams) |
| Right hip visible | Landmark visibility ≥ 0.25 threshold |
| Left knee visible | Landmark visibility ≥ 0.25 threshold |
| Right knee visible | Landmark visibility ≥ 0.25 threshold |
| Left ankle visible | Landmark visibility ≥ 0.25 threshold |
| Right ankle visible | Landmark visibility ≥ 0.25 threshold |
| Full lower body visible | All six landmarks above passing simultaneously |
| Inside exercise zone | Hip X-position within 5–95% of frame width; body height fraction ≥ 15% |
| Adequate lighting | Mean luminance of grayscale frame ≥ 20 (lenient for varying room settings) |
| Camera stable | Frame-to-frame landmark drift ≤ 0.20 normalized units |

*Additionally, a safety check requires that at least 3 out of the 6 lower-body landmarks are clearly visible with visibility > 0.5 to prevent low-confidence phantom joints.*

Only when **all 10 checks pass** is the patient allowed to proceed. The frontend renders a live checklist that updates per-frame, with a textual guidance message telling the patient exactly what to correct.

### 2. Live Exercise Session

Once validated, a second WebSocket connection (`/ws/session/{session_id}/{token}`) is opened. The patient performs their assigned exercise in front of the camera.

**Sensor Fusion & Closed Loop:**
If `PS3_USE_REAL_SENSOR=true` is enabled, the backend fuses the real-time sensor data polled from the ESP32 exoskeleton (angles, foot pressure force, motor status) with the computer vision metrics. This sensor telemetry is returned concurrently to the frontend client.

**Data flow per frame:**

### Browser Webcam
```
  → canvas.toDataURL('image/jpeg')
  → base64-encode
  → JSON {type: "frame", data: "<base64>"}
  ## Technology Stack

```

### Backend

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.10+ | Primary backend language |
| **FastAPI** | 0.111 | Async REST + WebSocket framework (ASGI) |
| **Uvicorn** | 0.29 | ASGI server with WebSocket support |
| **Motor** | 3.4.0 | Async MongoDB driver (non-blocking I/O) |
| **Beanie** | 1.26 | MongoDB ODM - document models with Pydantic V2 |
| **Pydantic** | 2.7 | Request/response validation and settings management |
| **python-jose** | 3.3 | JWT token creation and verification (HS256) |
| **passlib[bcrypt]** | 1.7 | Password hashing (bcrypt) |
| **MediaPipe** | 0.10.14 | Google's pose estimation model (33 keypoints) |
| **OpenCV** | 4.9 | Frame decoding, image processing, luminance analysis |
| **NumPy** | 1.26 | Landmark array processing, angle computation |
| **SciPy** | 1.13 | Signal filtering (Butterworth, Savitzky-Golay) |
| **Pandas** | 2.2 | Time-series CSV export |
| **WeasyPrint** | 62.3 | Server-side PDF report generation |
| **Cloudinary** | 1.40 | Optional cloud video/image storage (production) |
| **Loguru** | 0.7 | Structured logging |

### Frontend

| Technology | Version | Role |
|---|---|---|
| **React** | 18 | UI component framework |
| **TypeScript** | 5 | Type-safe application code |
| **Vite** | 5.4 | Build tool with HMR dev server and proxy |
| **Tailwind CSS** | 3 | Utility-first CSS framework |
| **Recharts** | 2 | Declarative chart library (ROM trends, compliance) |
| **Zustand** | 4 | Lightweight global state management (auth store) |
| **React Router** | 6 | Client-side navigation |
| **Axios** | 1.7 | HTTP client with interceptors for token refresh |
| **Lucide React** | 0.395 | Icon system (replaces all emoji) |
| **Sonner** | 1 | Toast notification system |
| **WebSocket API** | Native | Browser-native WebSocket for live sessions |
| **MediaStream API** | Native | Browser webcam capture (`getUserMedia`) |

### Infrastructure

| Technology | Role |
|---|---|
| **MongoDB Atlas** | Managed cloud database (free tier supported) |
| **Docker + Docker Compose** | Containerised development and deployment |
| **Vite Proxy** | Dev-time HTTP/WS forwarding from `:5173` → `:8000` |

---

## Database Design

SAMARTH uses **MongoDB** with **Motor** (async driver) and **Beanie** (ODM). All collections are schema-validated at the application layer via Pydantic V2 models.

### Collections

| Collection | Document Model | Description |
|---|---|---|
| `users` | `User` | Authentication records: email, password_hash, role (`patient`\|`therapist`\|`admin`), first_name, last_name, timestamps |
| `patients` | `Patient` | Patient profile: linked `user_id`, medical history, injury info, surgery date, target ROM limits. *Compliance rate is dynamically calculated.* |
| `therapists` | `Therapist` | Therapist profile: linked `user_id`, specialisation, clinic details, assigned `patient_ids` roster |
| `exercises` | `Exercise` | Exercise library: name, category, target ROM (degrees), target reps/sets, safety instructions, media/GIF URLs |
| `exercise_plans` | `ExercisePlan` | Therapist-assigned plans: list of exercises, frequencies, start/end dates |
| `sessions` | `Session` | Per-exercise-session record: mode (`live`\|`uploaded`), status (`in_progress`\|`completed`\|`abandoned`), duration, total reps, average ROM, symmetry, and **PS3 integration parameters** (`ps3_connected`, `ps3_avg_acceleration`, `ps3_commands_sent`, etc.) |
| `pose_data` | `PoseData` | Raw per-frame landmark dump (33 keypoints with normalized `x,y,z,visibility`) - linked to session |
| `angle_data` | `AngleData` | Computed joint angles per frame: left/right knee, hip, ankle (degrees) - linked to session |
| `uploaded_videos` | `UploadedVideo` | Video upload metadata: filename, storage path, file URL, PS1 batch processing status |
| `feedback_events` | `FeedbackEvent` | Real-time PS2 form-error events: flag types, rep indices, confidence scores, and timestamps |
| `analytics` | `AnalyticsSnapshot` | Aggregated weekly/monthly analytics: ROM averages, compliance rate, rep totals, symmetry index |
| `reports` | `Report` | Generated clinical report records: PDF/CSV files, download URLs, and metadata |
| `notifications` | `Notification` | In-app notifications: type, message, read status, linked user |
| `system_logs` | `SystemLog` | Structured security & audit log: actions, actors, resource targets, and timestamps |

### Key Relationships

```
User ──< Patient ──< Session ──< AngleData
                            ──< PoseData
                            ──< FeedbackEvent
                            ──< UploadedVideo
     ──< Therapist ──< ExercisePlan ──< Exercise
                   ──< Patient (assigned)
Patient ──< AnalyticsSnapshot
        ──< Report
        ──< Notification
```

### Indexing Strategy

- `users`: unique index on `email`
- `sessions`: compound index on `(patient_id, created_at DESC)`
- `angle_data`: index on `session_id`
- `analytics`: compound index on `(patient_id, week_start)`

---

## WebSocket Protocol

SAMARTH uses **two persistent WebSocket channels**, both authenticated via JWT token in the URL path (avoiding cookie complexity in WebSocket handshakes).

### Channel 1 - Camera Validation

```
Endpoint: ws://localhost:8000/ws/camera/{jwt_access_token}
Purpose:  Pre-session camera positioning check
Timeout:  15 seconds of inactivity
```

**Client → Server:**
```json
{ "type": "frame",  "data": "<base64-encoded JPEG>" }
{ "type": "ping" }
{ "type": "stop" }
```

**Server → Client:**
```json
{
  "type": "validation",
  "frame_index": 42,
  "pose_confidence": 0.87,
  "validation": {
    "left_hip_visible": true,
    "right_hip_visible": true,
    "left_knee_visible": true,
    "right_knee_visible": true,
    "left_ankle_visible": false,
    "right_ankle_visible": false,
    "full_lower_body_visible": false,
    "inside_zone": false,
    "adequate_lighting": true,
    "camera_stable": true,
    "all_valid": false,
    "guidance_message": "Move back - ankles not visible. Ensure feet are in frame"
  },
  "landmarks": {
    "LEFT_HIP":   { "x_norm": 0.48, "y_norm": 0.52, "visibility": 0.97 },
    "RIGHT_HIP":  { "x_norm": 0.55, "y_norm": 0.53, "visibility": 0.96 },
    "LEFT_KNEE":  { "x_norm": 0.47, "y_norm": 0.72, "visibility": 0.91 }
  }
}
```

### Channel 2 - Live Exercise Session

```
Endpoint: ws://localhost:8000/ws/session/{session_id}/{jwt_access_token}
Purpose:  Per-frame pose inference + rep counting during active exercise
Timeout:  30 seconds of inactivity
```

**Client → Server:**
```json
{ "type": "frame", "data": "<base64-encoded JPEG>" }
{ "type": "ping" }
{ "type": "end" }
```

**Server → Client (per frame):**
```json
{
  "type": "frame_result",
  "frame_index": 150,
  "timestamp_ms": 1717843200000,
  "pose_confidence": 0.91,
  "rep_count": 5,
  "angles": {
    "left_knee":  112.4,
    "right_knee": 109.8,
    "left_hip":   95.2,
    "right_hip":  93.7,
    "left_ankle": 88.1,
    "right_ankle": 87.6
  },
  "validation": { "...": "same structure as camera validation" },
  "landmarks": { "...": "normalised x,y + visibility per joint" }
}
```

**Rep Detection Algorithm:**

The repetition detector dynamically adapts to the kinematics of the active exercise by selecting specific joints, movement directions, and thresholds inside `SessionAnalysisContext`:
```python
# 1. Flexion Exercises (Squats, Lunges, SLR, Hip Abduction, Hurdle Step):
#    - Signal tracked: max(180.0 - left_joint, 180.0 - right_joint)
#    - Flexion peaks are local maximums (flexion_history[-2] > both neighbors)
#    - Thresholds range from 15.0° (SLR, Abduction) to 45.0° (Deep Squat).
# 
# 2. Extension Exercises (Knee Extension Seated, Sit to Stand):
#    - Signal tracked: max(left_knee, right_knee)
#    - Extension peaks are local maximums (straightening)
#    - Thresholds are 110.0° (Knee Extension) and 140.0° (Sit to Stand).
# 
# 3. Debounce: Cooldown period of 4.5 seconds between repetitions.
```

---

## Pipeline Modules

SAMARTH is architected around three decoupled processing modules (PS1, PS2, PS3). Each has a defined interface contract exposed through the backend service layer.

### PS1 - KinemaFlow (Computer Vision)

**Status: Integrated and active**

The PS1 pipeline (`/pipeline/`) is the team's existing computer vision engine. It is **never imported directly** by any backend route. All access goes through the `PoseEngineAdapter` singleton, which:

1. Injects the `/pipeline/` directory into `sys.path` at application startup
2. Exposes two operating modes:
   - **Real-time mode**: `process_frame(jpeg_bytes, frame_index, ts_ms, pose_estimator, kinematics)` → `RealtimeFrameResult`
   - **Batch mode**: `process_video(video_path, session_id, output_dir)` → `BatchProcessingResult`
3. Manages per-session `PoseEstimator` and `KinematicsExtractor` lifecycle (create/close)

**PS1 modules used:**
- `modules.pose_estimator.PoseEstimator` - MediaPipe Pose wrapper
- `modules.kinematics.KinematicsExtractor` - Joint angle computation from landmark coordinates
- `modules.background_seg.BackgroundSegmenter` - Background removal (batch mode)
- `modules.video_enhance.VideoEnhancer` - Brightness/contrast correction (batch mode)
- `modules.filter_utils` - Butterworth signal filtering
- `modules.feature_extractor` - Feature extraction utilities

### PS2 - Exercise Analysis ML Model

**Status: Integrated with RehabNet checkpoint**

The PS2 module analyses PS1 biomechanical time-series data to classify movement quality and detect error patterns. Set `PS2_MODEL_PATH=../pipeline/ps2/rehabnet_best.pth` and `PS2_USE_REAL_MODEL=true` in `.env` to load the real RehabNet analyzer. If the checkpoint cannot be loaded, the backend falls back to the mock analyzer instead of reporting fake real-model output.

**Error flags tracked (6 categories):**

| Flag | Description |
|---|---|
| `insufficient_ROM` | Knee flexion peak below target angle |
| `too_fast` | Rep duration below minimum threshold |
| `too_slow` | Rep duration above maximum threshold |
| `knee_valgus` | Inward knee collapse (medial deviation) |
| `asymmetric` | Left/right bilateral ROM difference > threshold |
| `trunk_comp` | Excessive trunk lateral flexion |

### PS3 - Wearable Sensor Hub

**Status: Stub (ready for integration)**

The PS3 module connects to BLE/IMU sensors (accelerometer, gyroscope) worn by the patient. A sensor HUD is rendered on the live session screen showing connection status and real-time inertial signals.

**To integrate your sensor firmware:**

1. Open [`backend/services/sensor_hub/`](backend/services/sensor_hub/)
2. Replace `mock_sensor.py` with BLE serial/bluetooth integration using `PS3_SENSOR_PORT` and `PS3_BAUD_RATE` from settings
3. Set `PS3_USE_REAL_SENSOR=true` in `.env`

---

## API Reference

All REST endpoints are prefixed with `/api/v1`. Full interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register new user (patient, therapist, or admin) |
| `POST` | `/auth/login` | Login and receive access + refresh JWT tokens |
| `POST` | `/auth/refresh` | Refresh access token using refresh token |
| `GET` | `/auth/me` | Get authenticated user profile |
| `GET` | `/exercises/` | List all exercises in the library |
| `GET` | `/exercises/{id}` | Get single exercise details |
| `POST` | `/exercises/seed` | Seed default exercise library into database |
| `POST` | `/sessions/` | Create a new exercise session |
| `GET` | `/sessions/` | List sessions for authenticated patient |
| `GET` | `/sessions/{id}` | Get session details |
| `DELETE` | `/sessions/{id}` | Delete session and associated data |
| `GET` | `/sessions/{id}/processing-status` | Get progress of batch video processing |
| `PUT` | `/sessions/{id}/complete` | Mark session as completed |
| `POST` | `/sessions/{id}/upload-video` | Upload recorded video for PS1 batch analysis |
| `GET` | `/sessions/{id}/angle-data` | Retrieve per-frame angle timeseries |
| `GET` | `/sessions/{id}/ps2-results` | Retrieve PS2 error analysis results |
| `GET` | `/analytics/patient/{id}/summary` | Patient-level aggregate analytics |
| `GET` | `/analytics/patient/{id}/weekly` | Week-by-week ROM and rep trends |
| `GET` | `/analytics/patient/{id}/rom-trends` | ROM trajectory over N days |
| `GET` | `/analytics/patient/{id}/exercise-breakdown` | Joint and ROM performance metrics |
| `POST` | `/reports/generate` | Generate PDF/CSV clinical report |
| `GET` | `/reports/patient/{id}` | List generated reports for patient |
| `GET` | `/patients/` | List all patients (therapist-only) |
| `GET` | `/patients/{id}` | Retrieve profile details of a single patient |
| `GET` | `/patients/{id}/sessions` | Get all sessions for a specific patient |
| `GET` | `/patients/{id}/analytics` | Retrieve deep analytics history for a patient |
| `GET` | `/patients/{id}/plan` | Retrieve assigned exercise plan for a patient |
| `GET` | `/notifications/` | List notifications for current user |
| `GET` | `/notifications/unread-count` | Get total count of unread notifications |
| `PUT` | `/notifications/{id}/read` | Mark notification as read |
| `PUT` | `/notifications/mark-all-read` | Mark all notifications as read |
| `GET` | `/sensor/status` | Retrieve connection, calibration, battery info |
| `GET` | `/sensor/data` | Retrieve latest cached IMU/FSR readings |
| `POST` | `/sensor/connect` | Manually connect to ESP32 URL |
| `POST` | `/sensor/disconnect` | Disconnect the active sensor hub |
| `POST` | `/sensor/calibrate` | Dispatch homing/zeroing calibration command |
| `POST` | `/sensor/command` | Dispatch manual motor speed/mode commands |
| `GET` | `/health` | System health check |

---

## Repository Structure

```
dashboard/
│
├── pipeline/                        # PS1 KinemaFlow pipeline (DO NOT MODIFY)
│   ├── modules/
│   │   ├── pose_estimator.py        # MediaPipe Pose wrapper
│   │   ├── kinematics.py            # Joint angle extraction
│   │   ├── background_seg.py        # Background segmentation
│   │   ├── video_enhance.py         # Frame quality enhancement
│   │   └── filter_utils.py          # Signal filtering
│   └── main.py                      # run_pipeline_on_video() entry point
│
├── backend/
│   ├── main.py                      # FastAPI app, router registration, lifespan
│   ├── backend_config.py            # Pydantic settings (env-driven)
│   ├── database.py                  # Motor + Beanie initialisation
│   │
│   ├── api/                         # REST route handlers
│   │   ├── auth.py                  # Register / Login / Refresh / Me
│   │   ├── sessions.py              # Session CRUD + video upload
│   │   ├── exercises.py             # Exercise library + seed
│   │   ├── analytics.py             # Patient analytics queries
│   │   ├── patients.py              # Patient management (therapist view)
│   │   ├── reports.py               # PDF/CSV report generation
│   │   └── notifications.py         # Notification management
│   │
│   ├── ws_handlers/
│   │   ├── session_ws.py            # /ws/session/{id}/{token} - live exercise
│   │   └── camera_ws.py             # /ws/camera/{token} - camera validation
│   │
│   ├── models/                      # Beanie document models (14 collections)
│   │   ├── user.py                  # User (auth)
│   │   ├── patient.py               # Patient profile
│   │   ├── therapist.py             # Therapist profile
│   │   ├── session.py               # Exercise session
│   │   ├── angle_data.py            # Per-frame joint angles
│   │   ├── pose_data.py             # Raw landmark dump
│   │   └── ...                      # (analytics, report, notification, etc.)
│   │
│   └── services/
│       ├── auth_service.py          # JWT creation/verification, password hashing
│       ├── pose_engine/
│       │   ├── adapter.py           # PoseEngineAdapter - gateway to PS1
│       │   └── schemas.py           # RealtimeFrameResult, BatchProcessingResult, etc.
│       ├── exercise_analysis/       # PS2 ML service - model_analyzer.py
│       └── sensor_hub/              # PS3 sensor driver - real_sensor.py / mock_sensor.py
│
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts            # Axios instance + JWT interceptor + auto-refresh
│   │   │   └── index.ts             # Typed API call functions
│   │   ├── components/
│   │   │   └── Layout.tsx           # Sidebar navigation + dark mode toggle
│   │   ├── pages/
│   │   │   ├── auth/                # Login + Register
│   │   │   ├── patient/
│   │   │   │   ├── PatientDashboard.tsx     # Session cards, quick stats
│   │   │   │   ├── ExerciseSelectionPage.tsx # Exercise library browser
│   │   │   │   ├── CameraValidationPage.tsx  # 10-check webcam validation
│   │   │   │   ├── LiveSessionPage.tsx       # Real-time angle gauges + rep counter
│   │   │   │   ├── UploadSessionPage.tsx     # Video upload + batch PS1
│   │   │   │   ├── SessionSummaryPage.tsx    # Post-session results
│   │   │   │   ├── AllSessionsPage.tsx       # Historical session browser
│   │   │   │   ├── AnalyticsPage.tsx         # ROM trends (Recharts)
│   │   │   │   └── ReportsPage.tsx           # PDF report generation
│   │   │   └── therapist/
│   │   │       ├── TherapistDashboard.tsx    # Patient roster + compliance
│   │   │       └── PatientDetailPage.tsx     # Per-patient ROM + session history
│   │   ├── stores/
│   │   │   └── authStore.ts         # Zustand store (tokens, user, logout)
│   │   ├── styles/
│   │   │   └── globals.css          # Design tokens, Tailwind components
│   │   └── types/                   # TypeScript interfaces
│   │
│   ├── vite.config.ts               # Vite + dev proxy (/api → :8000, /ws → :8000)
│   └── tailwind.config.ts           # Brand colours (#2A5BC4, #229096, #0F172A)
│
├── docker-compose.yml               # Backend + Frontend containers
├── .env                             # Root environment variables
└── logo.jpeg                        # SAMARTH brand mark
```


---

## Installation & Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Node.js | 20 LTS or higher |
| Python | 3.10 or higher |
| MongoDB Atlas | Free tier (M0) or local MongoDB |

### Step 1 - Clone and configure environment

```bash
git clone <repository-url>
cd dashboard
cp .env.example .env
```

Edit `.env` and set at minimum:
```bash
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<dbname>
JWT_SECRET=<generate with: python -c "import secrets; print(secrets.token_hex(64))">
```

### Step 2 - Backend setup

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --env-file "../.env" --reload
```

> **Note:** If you encounter a `motor`/`pymongo` version conflict, run:
> ```bash
> pip install "motor==3.4.0" "beanie==1.26.0"
> ```

The Swagger API docs will be available at: **http://localhost:8000/docs**

### Step 3 - Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The application will be available at: **http://localhost:5173**

### Step 4 - Seed exercise library

On first run, seed the exercise database by calling:
```bash
curl -X POST http://localhost:8000/api/v1/exercises/seed
```

Or simply click **Start Exercise** on the patient dashboard - the UI auto-seeds on empty library.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `APP_NAME` | No | `Samarth` | The name of the application |
| `APP_ENV` | No | `development` | Environment mode (`development`\|`production`) |
| `DEBUG` | No | `true` | Enable debug logs and detailed error reports |
| `LOG_LEVEL` | No | `info` | Logging verbosity (`info`\|`debug`\|`warning`\|`error`) |
| `HOST` | No | `0.0.0.0` | Host IP address of backend server |
| `PORT` | No | `8000` | Port of backend server |
| `FRONTEND_ORIGIN` | No | `http://localhost:5173` | Allowed origin for CORS validation |
| `MONGO_URI` | Yes | - | MongoDB connection string (Atlas or local) |
| `MONGO_DB_NAME` | No | `samarth` | MongoDB database name |
| `JWT_SECRET` | Yes | - | HS256 secret for signing tokens (min 64 chars) |
| `JWT_ALGORITHM` | No | `HS256` | Token signature verification algorithm |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | `60` | Lifespan of JWT access token |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | No | `30` | Lifespan of JWT refresh token |
| `STORAGE_BACKEND` | No | `local` | Media storage driver (`local`\|`cloudinary`) |
| `LOCAL_UPLOAD_DIR` | No | `./uploads` | Path for saving video and overlay outputs locally |
| `CLOUDINARY_CLOUD_NAME` | No | - | Cloudinary namespace (production CDN) |
| `CLOUDINARY_API_KEY` | No | - | Cloudinary public API key |
| `CLOUDINARY_API_SECRET` | No | - | Cloudinary private secret key |
| `PS1_PIPELINE_PATH` | No | `../pipeline` | Path to kinematics pipeline directory |
| `PS1_STRIDE` | No | `2` | Step stride for reading pre-recorded video frames |
| `PS1_ENHANCE` | No | `true` | Toggle frame histogram correction |
| `PS1_ENHANCE_LEVEL` | No | `auto` | Level of enhancement (`auto`\|`light`\|`aggressive`) |
| `PS1_SAVE_ANNOTATED_VIDEO` | No | `false` | Save visual overlay output video to storage |
| `PS2_MODEL_PATH` | No | `""` | Path to local trained RehabNet weights (`.pth` checkpoint) |
| `PS2_USE_REAL_MODEL` | No | `false` | Automatically enabled if valid `PS2_MODEL_PATH` exists |
| `PS3_ESP_URL` | No | `http://samarth-exo.local` | Direct Wi-Fi URL or mDNS address of the Exoskeleton ESP32 device |
| `PS3_USE_REAL_SENSOR` | No | `true` | Toggle physical sensor collection |
| `PS3_POLL_INTERVAL_MS` | No | `100` | Polling rate of hardware daemon thread |
| `PS3_RECONNECT_INTERVAL` | No | `5` | Retry delay (seconds) if connection is lost |
| `PS3_HEARTBEAT_TIMEOUT` | No | `10` | Disconnect threshold (seconds) without `/data` |
| `PS3_COMMAND_TIMEOUT` | No | `2.0` | Timeout threshold (seconds) for motor POST requests |
| `PS3_SENSOR_PORT` | No | `""` | Serial interface port (fallback USB usage) |
| `PS3_BAUD_RATE` | No | `115200` | USB Serial baud rate |
| `SMTP_HOST` | No | `""` | Mail server SMTP hostname |
| `SMTP_PORT` | No | `587` | Mail server SMTP port |
| `SMTP_USER` | No | `""` | SMTP sender username |
| `SMTP_PASSWORD` | No | `""` | SMTP sender password |
| `EMAIL_FROM` | No | `noreply@samarth.health` | Outgoing email address for system notifications |
| `WEASYPRINT_ENABLED` | No | `true` | Toggle server-side PDF report compilation |
| `REPORTS_DIR` | No | `./reports` | Path for saving PDF exports |

---

## Docker Deployment

A `docker-compose.yml` is provided for containerised deployment:

```bash
# From project root (dashboard/)
docker-compose up --build
```

This starts:
- **Backend** on port `8000` - FastAPI + Uvicorn, with `/pipeline/` mounted read-only
- **Frontend** on port `5173` - Vite dev server (or build for Nginx in production)

For production, set `STORAGE_BACKEND=cloudinary` and configure Cloudinary credentials to avoid relying on local filesystem for video storage.

---

## Integrating PS2 and PS3

### PS2 - RehabNet Exercise Analyzer

The system is wired to `backend/services/exercise_analysis/model_analyzer.py`, which loads the PyTorch / iTransformer weights (e.g. `rehabnet_best.pth`). Real-time and batch-upload flows pass joint angles into RehabNet, returning structured evaluations that map to the Beanie MongoDB documents.

1. Place your model weights file (e.g., `models/rehabnet_best.pth`)
2. Set `PS2_MODEL_PATH=models/rehabnet_best.pth` and `PS2_USE_REAL_MODEL=true` in `.env`
3. Implement `analyze_rep()` in `backend/services/exercise_analysis/model_analyzer.py` matching the Pydantic schema:

```python
def analyze_rep(
    self,
    angle_data: Dict,
    rep_id: int,
    rep_number: int = 1,
    fps: float = 30.0,
) -> PS2RepResult:
    # angle_data: {"left_knee": [float, ...], "right_knee": [...], "left_hip": [...], ...}
    # Returns a validated PS2RepResult object:
    return PS2RepResult(
        timestamp=datetime.utcnow().timestamp(),
        rep_id=rep_id,
        error_flags=PS2ErrorFlags(
            insufficient_ROM=0,
            too_fast=0,
            too_slow=1,
            knee_valgus=0,
            asymmetric=0,
            trunk_comp=0,
        ),
        confidence=PS2Confidence(too_slow=0.9),
        mode_command=PS2ModeCommand(
            mode_id=1,
            mode_name="Assistive",
            target_torque=3.0,
        ),
        session=PS2SessionMetrics(
            rep_number=rep_number,
            session_score=0.85,
            quality_trend="stable",
        )
    )
```

The `PS2_USE_REAL_MODEL` flag is **automatically set** to `true` when `PS2_MODEL_PATH` points to an existing file - no other config change needed.


### Training & Fine-Tuning with Collected User Data

As patients perform exercises, the system records raw 3D pose coordinates in the `PoseData` collection. To export this dataset for retraining deep learning models:
1. Run the database exporter utility:
   ```bash
   python scripts/export_user_data.py --output ../../data/user_pose_data.csv
   ```
2. The script processes completed repetitions, assigns classification labels (based on whether PS2 flagged errors), extracts joint coordinates frame-by-frame, and outputs a formatted, training-ready dataset to `data/user_pose_data.csv`. This can be fed directly to the training scripts under `ml-model/` or `POC/notebooks/`.

### PS3 - Wearable Sensor Hub (ESP32 Integration)

SAMARTH is directly integrated with a single unified ESP32 exoskeleton hardware driver (Wi-Fi based). The ESP32 board runs a local web server to read dual MPU6050 IMU joint angles (thigh and shank) and foot force sensors (FSRs) to control a BTS7960 motor driver for knee assistance. 

> [!NOTE]
> **Knee-Motor-Only Configuration**: The current hardware configuration is optimized for a knee-motor-only setup. Hip motor pins and drive logic are disabled in the ESP32 firmware. To maintain API consistency, the ESP32 continues to report default placeholder values for the hip motor (`hip_pwm: 0`, `hip_dir: "ext"`, `hip_angle: 0.0`) in the `/data` and `/telemetry` endpoints, ensuring the backend operates without changes.

> [!IMPORTANT]
> **Serial Interface**: The firmware utilizes standard ASCII characters for its serial output messages (using tags like `[INFO]`, `[SUCCESS]`, `[WARNING]`, `[ERROR]`, and `[ML]`) to avoid garbled output or encoding issues on terminal clients that do not support Unicode characters. Additionally, if the IMUs fail to initialize, the firmware reports a warning instead of halting in a blocking loop, preventing watchdog resets.

#### Exoskeleton Control Loop Design & Sensor Fusion

For research and replication, the firmware implements the following low-level control loop and data processing paradigms:

1. **Sensor Fusion via Complementary Filter**
   To filter out high-frequency accelerometer vibrations and low-frequency gyroscope drift, a complementary filter runs onboard at the main loop rate:
   $$\theta_{filtered} = \alpha \cdot (\theta_{filtered} + \omega \cdot dt) + (1 - \alpha) \cdot \theta_{acc}$$
   Where the filter gain is configured to $\alpha = 0.98$ (98% gyroscope angular velocity integration and 2% absolute accelerometer pitch reference).

2. **Differential Joint Angle Calculation**
   The active knee joint flexion/extension angle is computed dynamically as the difference between the absolute thigh and shank pitch:
   $$\theta_{knee} = \theta_{thigh} - \theta_{shank}$$
   The calculated angle is software-constrained to physiological safety bounds: $\theta_{knee} \in [0^{\circ}, 130^{\circ}]$.

3. **Gait Stance Phase Classification**
   Foot pressure signals are acquired via an analog Force Sensitive Resistor (FSR) on GPIO 34. To eliminate transient spikes and contact bounce, a low-pass filter is applied before binary classification:
   $$F_{filtered} = 0.9 \cdot F_{filtered} + 0.1 \cdot F_{raw}$$
   $$\text{Stance} = \begin{cases} \text{true} & \text{if } F_{filtered} > 2.0\text{ N} \\ \text{false} & \text{otherwise} \end{cases}$$

4. **Gait-Triggered Assistance (PWM Scaling)**
   When operating in Gait-Triggered Assistive mode (`mode_id == 1`), torque is applied proportionally to the joint flexion angle exceeding a $30^{\circ}$ threshold during the stance phase:
   $$PWM_{knee} = \text{constrain}\left( \tau_{target} \cdot K_{scale} \cdot \frac{\theta_{knee} - 30^{\circ}}{90^{\circ}}, 0, 255 \right)$$
   Where $\tau_{target}$ is the target torque (0.0 to 2.0 Nm) predicted by the RehabNet ML model, and $K_{scale} = 150$ maps the physical torque to duty cycle. If $\text{Stance} = \text{false}$ or $\theta_{knee} \le 30^{\circ}$, the motor is de-energized ($PWM = 0$).

#### Exoskeleton Hardware Pin Mapping & Schematic

The single-ESP32 architecture utilizes the following pin assignments to interface with the actuators, IMUs, and pressure sensors:

```text
                      ┌────────────────────────┐
                      │    ESP32 DEVKIT V1     │
                      │                        │
        [Boot Reset]  │ 00 (GPIO 0)      GND   │───────── Ground (GND Bus)
  BTS7960 Knee RPWM  ◀│ 04 (GPIO 4)      3V3   │───────── Power (3.3V Bus)
  BTS7960 Knee L_EN  ◀│ 05 (GPIO 5)      EN    │
                      │ 12 (GPIO 12)*    VP    │
                      │ 13 (GPIO 13)*    VN    │
                      │ 14 (GPIO 14)*    34    │◀──────── FSR Signal (Analog)
                      │ 15 (GPIO 15)*    35    │
  BTS7960 Knee LPWM  ◀│ 16 (GPIO 16)     32    │
  BTS7960 Knee R_EN  ◀│ 17 (GPIO 17)     33    │
                      │ 25               26    │
                      │ 27               23    │
    I2C Bus SDA Pin  ◀│ 21 (GPIO 21)     19    │
    I2C Bus SCL Pin  ◀│ 22 (GPIO 22)     18    │
                      │ Tx0              Rx0   │
                      └────────────────────────┘
                       * Note: Hip pins are disabled/unused.
```

##### Pin Connection Table

| Peripheral | ESP32 Pin | Component Pin | Function | Status |
|---|---|---|---|---|
| **Knee Actuator** | GPIO 4 | RPWM | Forward Speed PWM (20kHz) | **Active** |
| | GPIO 16 | LPWM | Backward Speed PWM (20kHz) | **Active** |
| | GPIO 17 | R_EN | Forward Drive Enable (High) | **Active** |
| | GPIO 5 | L_EN | Backward Drive Enable (High) | **Active** |
| **Hip Actuator** | GPIO 13 | RPWM | Forward Speed PWM | *Disabled* |
| | GPIO 14 | LPWM | Backward Speed PWM | *Disabled* |
| | GPIO 15 | R_EN | Forward Drive Enable | *Disabled* |
| | GPIO 12 | L_EN | Backward Drive Enable | *Disabled* |
| **Dual MPU6050** | GPIO 21 | SDA | Shared I2C Data | **Active** |
| | GPIO 22 | SCL | Shared I2C Clock | **Active** |
| **Foot FSR** | GPIO 34 | A0 / Signal | Analog Voltage Input | **Active** |

##### Sensor Configuration Notes
*   **Thigh IMU (I2C 0x68)**: The hardware AD0 address select pin must be pulled **LOW** (connected to GND).
*   **Shank IMU (I2C 0x69)**: The hardware AD0 address select pin must be pulled **HIGH** (connected to 3.3V).
*   **FSR Sensor divider**: Connect a $10\text{ k}\Omega$ pull-down resistor from GPIO 34 to GND, with the FSR connected between GPIO 34 and 3.3V.

#### 1. Configuration & Connection
1. Set the following environment variables in `backend/.env`:
   ```env
   PS3_ESP_URL=http://samarth-exo.local
   PS3_USE_REAL_SENSOR=true
   PS3_POLL_INTERVAL_MS=100
   ```
   *(Note: The ESP32 starts an mDNS responder on boot. By setting `PS3_ESP_URL=http://samarth-exo.local`, the backend will automatically discover the ESP32 dynamically on the local network, removing the need to track dynamic IP allocations or inspect the Serial Monitor).*
2. On boot, if the ESP32 cannot locate the saved access point (e.g. mobile hotspot), it times out after 8 seconds and spawns a Wi-Fi captive setup portal named `Samarth-Exo-Setup` (IP: `192.168.4.1`) for local Wi-Fi provisioning.

#### 2. Backend Polling Daemon
To prevent network request latency from blocking the live camera pose evaluation loop (which requires high frame-rate rendering), the backend `RealSensorHub` launches a background polling thread upon calling `.connect()`. This thread continuously queries the ESP32 at `/data` at the configured `PS3_POLL_INTERVAL_MS` rate and caches the state, allowing the live WebSocket pipeline to fetch sensor telemetry in `0ms` without blocking the main event loop.

#### 3. ESP32 HTTP API Contract

* **`GET /data`**: Returns real-time telemetry frame:
  ```json
  {
    "knee_angle": 75.3,           // Active knee joint angle (deg)
    "hip_angle": 45.1,            // Active hip joint angle (deg)
    "foot_force": 3.42,           // Foot pressure sensor load (Newtons)
    "stance": true,               // Stance phase (true = on ground)
    "knee_pwm": 95,               // Knee actuator speed (0-255)
    "knee_dir": "ext",            // Knee direction ("ext" or "flex")
    "hip_pwm": 60,                // Hip actuator speed (0-255)
    "hip_dir": "ext",             // Hip direction ("ext" or "flex")
    "motors": "on",               // Motor state ("on" or "off")
    "battery_percent": 100,       // Battery charge level (%)
    "calibration_status": "calibrated", // Homing calibration state
    "connected": true             // Device connection status
  }
  ```

* **`GET /status`**: Returns device status and connection information:
  ```json
  {
    "device_id": "ESP32-SAMARTH-EXO",
    "battery_percent": 100,
    "calibration_status": "calibrated",
    "connected": true,
    "ip": "192.168.1.15",
    "channel": 6,
    "uptime_ms": 125430,
    "current_mode": 0,
    "current_torque": 0.0
  }
  ```

* **`POST /calibrate`**: Dispatches a homing calibration command to trigger motor zeroing. Returns the updated calibration state:
  ```json
  {
    "calibration_status": "calibrated",
    "message": "Calibration is performed automatically on boot. Restart the device to recalibrate."
  }
  ```

* **`POST /command`**: Dispatches mode configuration and target torque to the actuators:
  ```json
  {
    "mode_id": 2,                 // None (0), Assistive (1), Resistive (2)
    "target_torque": 1.2          // Target torque in Nm
  }
  ```
  *(Note: The backend validates command payloads against the `CommandRequest` schema, which filters out direct PWM/direction overrides. Extra fields are ignored by the ESP32.)*

* **`POST /exercise`**: Legacy alias for `POST /command` (identical behavior).

* **`GET /telemetry`**: Legacy telemetry endpoint returning raw values:
  ```json
  {
    "battery_percent": 100,
    "calibration_status": true,
    "knee_angle": 75.3,
    "hip_angle": 45.1,
    "foot_force": 3.42,
    "stance": true,
    "knee_motor_pwm": 95,
    "knee_motor_dir": true,
    "hip_motor_pwm": 60,
    "hip_motor_dir": true,
    "motors_active": true,
    "connected": true
  }
  ```
---

<div align="center">

**SAMARTH** - Physiotherapy · Machine Learning · Impact

*IIT (BHU) Varanasi*

</div>
