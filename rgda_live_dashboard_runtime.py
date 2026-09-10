"""
Live dashboard runtime for exercise tracking.

Use with:
  RGDA_DATA_PIPELINE_ONLY.py
  rgda_rehabnet_architecture.py
  rgda_dashboard_clean.py

Design:
  - Live per-frame output is rule/biomechanics based.
  - Skeleton overlay is drawn on every frame.
  - Keypoints are buffered during the live session.
  - After the session ends, finalize_session() segments reps and returns the
    same clean dashboard summary style as uploaded-video analysis.
  - RGDA model can be passed, but final patient-facing result remains rule/flag
    based because that is currently more reliable for live MediaPipe input.
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from rgda_dashboard_clean import (
    JOINT_COLS,
    BASE_COLS,
    EXERCISE2IDX,
    ROM_TARGETS,
    TARGET_ANGLES,
    UNILATERAL_EXERCISES,
    VEL_THRESHOLD,
    JERK_THRESHOLD,
    FPS,
    ps1_extract_row,
    ps1_root_center,
    ps1_bone_normalize,
    ps1_extract_features,
    ps1_df_to_samples,
    infer_sample_dashboard,
    _safe_angle_arr,
    draw_lower_body_skeleton,
)


LIVE_MP = {
    "HipLeft": 23,
    "KneeLeft": 25,
    "AnkleLeft": 27,
    "HipRight": 24,
    "KneeRight": 26,
    "AnkleRight": 28,
}


def _angle_3pt(a, b, c):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v1 = a - b
    v2 = c - b
    denom = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)
    cos = float(np.dot(v1, v2) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _landmark_xyz(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _frame_joint_angles(landmarks):
    lh = _landmark_xyz(landmarks, LIVE_MP["HipLeft"])
    lk = _landmark_xyz(landmarks, LIVE_MP["KneeLeft"])
    la = _landmark_xyz(landmarks, LIVE_MP["AnkleLeft"])
    rh = _landmark_xyz(landmarks, LIVE_MP["HipRight"])
    rk = _landmark_xyz(landmarks, LIVE_MP["KneeRight"])
    ra = _landmark_xyz(landmarks, LIVE_MP["AnkleRight"])
    return {
        "left_knee_deg": _angle_3pt(lh, lk, la),
        "right_knee_deg": _angle_3pt(rh, rk, ra),
    }


def _estimate_phase(angle_history):
    if len(angle_history) < 5:
        return "tracking"
    prev = np.mean(list(angle_history)[-5:-2])
    curr = np.mean(list(angle_history)[-2:])
    # Knee angle decreases while bending deeper.
    if curr < prev - 2.0:
        return "descent"
    if curr > prev + 2.0:
        return "ascent"
    return "hold"


def live_feedback_from_angles(exercise, left_knee, right_knee, phase, angle_history):
    """
    Frame-level cue. This is intentionally simple and fast.
    Final correctness is still computed at rep/session level.
    """
    target = TARGET_ANGLES.get(exercise, TARGET_ANGLES.get("default", 90.0))
    active_angle = min(left_knee, right_knee)
    symmetry_gap = abs(left_knee - right_knee)

    flags = []
    feedback = "Good control"

    # For knee-flexion exercises, smaller knee angle generally means deeper bend.
    if exercise in {"squat", "deep_squat", "ctk_squat", "inline_lunge", "side_lunge", "knee_bend"}:
        if phase in {"descent", "hold"} and active_angle > target + 18.0:
            flags.append("insufficient_depth_live")
            feedback = "Bend a little lower"

    if exercise not in UNILATERAL_EXERCISES and symmetry_gap > 25.0:
        flags.append("asymmetric_live")
        feedback = "Move both sides evenly"

    if len(angle_history) >= 4:
        recent = np.asarray(list(angle_history)[-4:], dtype=np.float32)
        speed_proxy = float(np.mean(np.abs(np.diff(recent)))) * FPS
        if speed_proxy > VEL_THRESHOLD:
            flags.append("too_fast_live")
            feedback = "Slow down"

    return feedback, flags


class LiveExerciseSession:
    """
    Dashboard-facing live session.

    Expected usage:
      session = LiveExerciseSession(exercise="squat", patient_id="P001", model=model)

      while camera_running:
          overlay_frame, live = session.process_frame(frame_bgr)
          display overlay_frame and live["live_feedback"]

      summary = session.finalize_session()
    """

    def __init__(
        self,
        exercise,
        patient_id="LIVE_001",
        model=None,
        device=None,
        hardware_mode=0,
        fps=25.0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        expose_model_debug=False,
    ):
        if exercise is None or exercise == "unknown":
            raise ValueError("Exercise must be selected manually by the dashboard.")

        self.exercise = exercise
        self.patient_id = patient_id
        self.model = model
        self.device = device
        self.hardware_mode = hardware_mode
        self.fps = fps
        self.expose_model_debug = expose_model_debug

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.rows = []
        self.frame_idx = 0
        self.detected = 0
        self.start_time = time.time()
        self.angle_history = deque(maxlen=20)
        self.last_live = {
            "tracking": False,
            "live_feedback": "Step into frame",
            "live_flags": ["no_pose"],
            "current_rep_phase": "no_pose",
        }

    def process_frame(self, frame_bgr):
        """
        Process one webcam frame.

        Input:
          frame_bgr: OpenCV BGR frame

        Returns:
          overlay_frame_bgr, live_dict
        """
        self.frame_idx += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        row = {
            "subject_id": self.patient_id,
            "session": "live",
            "label": "unknown",
            "frame_id": self.frame_idx,
            "time_s": round(self.frame_idx / self.fps, 4),
            "detection": 0,
        }
        row.update({col: float("nan") for col in JOINT_COLS})

        overlay = frame_bgr.copy()

        if not res.pose_landmarks:
            cv2.putText(
                overlay,
                "No pose detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            self.rows.append(row)
            self.last_live = {
                "tracking": False,
                "live_angles": {},
                "live_feedback": "Step into frame",
                "live_flags": ["no_pose"],
                "current_rep_phase": "no_pose",
            }
            return overlay, self.last_live

        self.detected += 1
        row["detection"] = 1
        landmarks = res.pose_landmarks.landmark
        row.update(ps1_extract_row(landmarks))
        self.rows.append(row)

        overlay = draw_lower_body_skeleton(overlay, landmarks)
        angles = _frame_joint_angles(landmarks)
        mean_knee = (angles["left_knee_deg"] + angles["right_knee_deg"]) / 2.0
        self.angle_history.append(mean_knee)
        phase = _estimate_phase(self.angle_history)
        feedback, flags = live_feedback_from_angles(
            self.exercise,
            angles["left_knee_deg"],
            angles["right_knee_deg"],
            phase,
            self.angle_history,
        )

        color = (0, 255, 0) if not flags else (0, 180, 255)
        cv2.putText(
            overlay,
            f"{self.exercise} | {phase} | {feedback}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"L knee {angles['left_knee_deg']:.0f}  R knee {angles['right_knee_deg']:.0f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.last_live = {
            "tracking": True,
            "live_angles": {
                "left_knee_deg": round(angles["left_knee_deg"], 2),
                "right_knee_deg": round(angles["right_knee_deg"], 2),
            },
            "live_feedback": feedback,
            "live_flags": flags,
            "current_rep_phase": phase,
        }
        return overlay, self.last_live

    def finalize_session(self):
        """
        Segment buffered live frames into reps and return dashboard-safe summary.
        """
        if not self.rows:
            return {
                "patient": self.patient_id,
                "exercise": self.exercise,
                "n_reps": 0,
                "n_correct": 0,
                "avg_quality": 0.0,
                "reps": [],
            }

        df = pd.DataFrame(self.rows, columns=BASE_COLS + JOINT_COLS)
        df = df[df["detection"] == 1].reset_index(drop=True)
        if len(df) < 20:
            return {
                "patient": self.patient_id,
                "exercise": self.exercise,
                "n_reps": 0,
                "n_correct": 0,
                "avg_quality": 0.0,
                "reps": [],
                "message": "Not enough detected frames.",
            }

        df = ps1_root_center(df)
        df = ps1_bone_normalize(df)
        df = ps1_extract_features(df)
        df = df.ffill().bfill().fillna(0.0)

        samples = ps1_df_to_samples(df, self.patient_id, self.exercise)
        results = []
        for sample in samples:
            sample["exercise"] = self.exercise
            sample["exercise_idx"] = EXERCISE2IDX.get(self.exercise, EXERCISE2IDX["unknown"])
            result = infer_sample_dashboard(
                self.model,
                sample,
                device=self.device,
                hardware_mode=self.hardware_mode,
                expose_model_debug=self.expose_model_debug,
            )
            results.append(result)

        n_correct = sum(r["correct"] for r in results)
        avg_quality = float(np.mean([r["quality_score"] for r in results])) if results else 0.0
        return {
            "patient": self.patient_id,
            "exercise": self.exercise,
            "n_reps": len(results),
            "n_correct": n_correct,
            "pct_correct": round(100.0 * n_correct / max(len(results), 1), 1),
            "avg_quality": round(avg_quality, 3),
            "reps": results,
        }

    def close(self):
        self.pose.close()

