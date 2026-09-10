"""
Clean dashboard runtime for RGDA RehabNet.

Place this file beside:
  - RGDA_DATA_PIPELINE_ONLY.py
  - rgda_rehabnet_architecture.py

Dashboard-facing behavior:
  - Exercise is selected manually by the dashboard.
  - No exercise classifier is loaded.
  - Final correct/incorrect decision is rule-based for sensible live-video output.
  - RGDA model output is kept internally as debug/research signal.
  - Skeleton overlay video is saved if overlay_out is provided.
  - Hardware payload includes torque, target angle, ROM, peak angles, and deficits.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter as _sgf
import mediapipe as mp
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision as mpvision

from RGDA_DATA_PIPELINE_ONLY import *  # noqa: F403,F401
from rgda_rehabnet_architecture import RGDARRehabNet, rule_flags_from_scalars

# Runtime fallbacks for older extracted data-pipeline files.
# These match the scalar storage convention used by extract_scalars:
# velocity is stored as vel/200, jerk is stored as jerk/10.
VEL_NORM = globals().get("VEL_NORM", 200.0)
JERK_NORM = globals().get("JERK_NORM", 10.0)
VEL_THRESHOLD = globals().get("VEL_THRESHOLD", 150.0)
JERK_THRESHOLD = globals().get("JERK_THRESHOLD", 27.66)


LOWER_BODY_MP = {
    "HipLeft": 23,
    "KneeLeft": 25,
    "AnkleLeft": 27,
    "HipRight": 24,
    "KneeRight": 26,
    "AnkleRight": 28,
}

LOWER_BODY_CONNECTIONS = [
    ("HipLeft", "KneeLeft"),
    ("KneeLeft", "AnkleLeft"),
    ("HipRight", "KneeRight"),
    ("KneeRight", "AnkleRight"),
    ("HipLeft", "HipRight"),
    ("KneeLeft", "KneeRight"),
    ("AnkleLeft", "AnkleRight"),
]


def _smooth15(signal):
    a = np.asarray(signal, dtype=np.float64)
    n = len(a)
    if n < 5:
        return a

    w = min(15, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return a

    try:
        return _sgf(a, w, 3)
    except Exception:
        return a


def _safe_angle_arr(hip, knee, ank):
    hip = np.asarray(hip, dtype=np.float64)
    knee = np.asarray(knee, dtype=np.float64)
    ank = np.asarray(ank, dtype=np.float64)

    v1 = hip - knee
    v2 = ank - knee
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def draw_lower_body_skeleton(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = {}

    for name, idx in LOWER_BODY_MP.items():
        lm = landmarks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        pts[name] = (x, y)

    for a, b in LOWER_BODY_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 3)

    for name, p in pts.items():
        cv2.circle(frame, p, 6, (0, 0, 255), -1)
        label = name.replace("Left", "L").replace("Right", "R")
        cv2.putText(
            frame,
            label,
            (p[0] + 5, p[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def ps1_extract_features(df):
    df["angle_knee_L"] = _knee_angle_from_df(df, "L")  # noqa: F405
    df["angle_knee_R"] = _knee_angle_from_df(df, "R")  # noqa: F405
    df["angle_knee_L_smooth"] = _smooth15(df["angle_knee_L"].values)
    df["angle_knee_R_smooth"] = _smooth15(df["angle_knee_R"].values)
    return df


def ps1_process_video_overlay(
    video_path,
    patient_id,
    exercise="unknown",
    session="1",
    label="unknown",
    overlay_out=None,
):
    """
    PS1 video processing with optional lower-body skeleton overlay video.
    Returns the same dataframe format expected by ps1_df_to_samples(...).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or FPS
    rows, frame_idx, detected = [], 0, 0
    writer = None
    t0 = time.time()

    model_path = "/tmp/pose_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model (~30MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            model_path,
        )
        print("Downloaded.")

    opts = mpvision.PoseLandmarkerOptions(  # noqa: F405
        base_options=mptasks.BaseOptions(model_asset_path=model_path),  # noqa: F405
        running_mode=mpvision.RunningMode.VIDEO,  # noqa: F405
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECT_CONF,  # noqa: F405
        min_pose_presence_confidence=MIN_DETECT_CONF,  # noqa: F405
        min_tracking_confidence=MIN_TRACK_CONF,  # noqa: F405
        output_segmentation_masks=False,
    )

    with mpvision.PoseLandmarker.create_from_options(opts) as pose:  # noqa: F405
        frame_step_ms = int(1000.0 / fps_vid)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            frame_ms = frame_idx * frame_step_ms
            prepped = ps1_prepare_frame(frame)  # noqa: F405
            rgb = cv2.cvtColor(prepped, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # noqa: F405
            res = pose.detect_for_video(mp_img, frame_ms)

            row = {
                "subject_id": patient_id,
                "session": session,
                "label": label,
                "frame_id": frame_idx,
                "time_s": round(frame_idx / fps_vid, 4),
                "detection": 0,
            }
            row.update({col: float("nan") for col in JOINT_COLS})  # noqa: F405

            overlay_frame = prepped.copy()

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                detected += 1
                row["detection"] = 1
                landmarks = res.pose_landmarks[0]
                row.update(ps1_extract_row(landmarks))  # noqa: F405
                overlay_frame = draw_lower_body_skeleton(overlay_frame, landmarks)
                cv2.putText(
                    overlay_frame,
                    f"{exercise} | tracking",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    overlay_frame,
                    "No pose detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if overlay_out is not None:
                if writer is None:
                    oh, ow = overlay_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(overlay_out, fourcc, fps_vid, (ow, oh))
                writer.write(overlay_frame)

            rows.append(row)

    cap.release()
    if writer is not None:
        writer.release()
        print("Overlay video saved:", overlay_out)

    if frame_idx == 0:
        raise ValueError("Empty video")

    det_pct = 100 * detected / frame_idx
    print(f"  PS1: {frame_idx} frames  detection={det_pct:.0f}%  {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows, columns=BASE_COLS + JOINT_COLS)  # noqa: F405
    df = df[df["detection"] == 1].reset_index(drop=True)
    df = ps1_root_center(df)  # noqa: F405
    df = ps1_bone_normalize(df)  # noqa: F405
    df = ps1_extract_features(df)
    df = df.ffill().bfill().fillna(0.0)
    return df


def compute_torque(quality, mode=0):
    m = HARDWARE_MODES.get(mode, HARDWARE_MODES[0])  # noqa: F405
    return float(np.clip(m["scale"] * float(quality) + m["bias"], m["min"], m["max"]))


def rule_quality_score(scalars, exercise):
    l_rom = float(scalars[0] * 180.0)
    r_rom = float(scalars[1] * 180.0)
    sym = float(scalars[4] * 100.0)
    vel = float(scalars[5] * VEL_NORM)  # noqa: F405
    jerk = float(scalars[6] * JERK_NORM)  # noqa: F405
    smoothness = float(scalars[9])

    target = ROM_TARGETS.get(exercise, ROM_TARGETS["default"])  # noqa: F405
    active_rom = max(l_rom, r_rom)

    rom_score = np.clip(active_rom / max(target, 1e-6), 0.0, 1.0)
    if exercise in UNILATERAL_EXERCISES:  # noqa: F405
        sym_score = 1.0
    else:
        sym_score = np.clip(1.0 - sym / 35.0, 0.0, 1.0)
    vel_score = np.clip(
        1.0 - max(0.0, vel - 80.0) / max(VEL_THRESHOLD - 80.0, 1e-6),  # noqa: F405
        0.0,
        1.0,
    )
    jerk_score = np.clip(1.0 - jerk / max(JERK_THRESHOLD, 1e-6), 0.0, 1.0)  # noqa: F405
    smooth_score = np.clip(smoothness, 0.0, 1.0)

    quality = float(np.clip(
        0.40 * rom_score +
        0.20 * sym_score +
        0.15 * vel_score +
        0.15 * jerk_score +
        0.10 * smooth_score,
        0.0,
        1.0,
    ))
    return quality


def build_hardware_payload(sample, quality, scalars, torque, mode):
    kps = sample["keypoints"]
    lk = _safe_angle_arr(kps[:, 0, :], kps[:, 1, :], kps[:, 2, :])
    rk = _safe_angle_arr(kps[:, 3, :], kps[:, 4, :], kps[:, 5, :])

    l_peak = float(lk.min())
    r_peak = float(rk.min())
    l_rom = float(lk.max() - l_peak)
    r_rom = float(rk.max() - r_peak)

    exercise = sample.get("exercise", "unknown")
    target = TARGET_ANGLES.get(exercise, TARGET_ANGLES["default"])  # noqa: F405
    l_deficit = max(0.0, target - l_peak)
    r_deficit = max(0.0, target - r_peak)

    return {
        "torque": round(float(torque), 4),
        "mode": HARDWARE_MODES[mode]["name"],  # noqa: F405
        "mode_int": int(mode),
        "joint": {
            "left_knee_peak_deg": round(l_peak, 2),
            "right_knee_peak_deg": round(r_peak, 2),
            "left_knee_rom_deg": round(l_rom, 2),
            "right_knee_rom_deg": round(r_rom, 2),
        },
        "target": {
            "knee_target_deg": target,
            "left_deficit_deg": round(l_deficit, 2),
            "right_deficit_deg": round(r_deficit, 2),
        },
        "torque_per_degree": round(float(torque) / max(l_deficit, r_deficit, 1.0), 5),
    }


def load_rgda_model(model_path, device=None):
    device = device or DEVICE  # noqa: F405
    model = RGDARRehabNet(
        n_exercises=N_EXERCISES,  # noqa: F405
        n_modes=N_MODES,  # noqa: F405
        n_domains=4,
        scalar_dim=10,
        rule_dim=4,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def infer_sample_dashboard(model, sample, device=None, hardware_mode=0, expose_model_debug=False):
    """
    Dashboard-safe inference.

    Patient-facing final output is based on biomechanical rules.
    RGDA prediction is optional debug metadata.
    """
    device = device or DEVICE  # noqa: F405
    exercise = sample.get("exercise", "unknown")

    model_debug = {}
    if model is not None:
        try:
            model.eval()
            kps_t = torch.from_numpy(sample["keypoints"]).permute(2, 0, 1).unsqueeze(0).to(device)
            adj_t = torch.from_numpy(ADJ).to(device)  # noqa: F405
            sc_t = torch.from_numpy(sample["scalars"]).float().unsqueeze(0).to(device)
            flags_t = rule_flags_from_scalars(
                torch.from_numpy(sample["scalars"]).float(),
                exercise=exercise,
            ).unsqueeze(0).to(device)
            mode_t = torch.tensor([hardware_mode], dtype=torch.long, device=device)
            ex_t = torch.tensor([sample["exercise_idx"]], dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(kps_t, adj_t, sc_t, flags_t, mode_t, ex_t, grl_lambda=0.0)
                probs = torch.softmax(out["correctness_logits"], dim=1)[0]
                model_labels = ["incorrect", "correct", "review"]
                pred_idx = int(torch.argmax(probs).item())
                model_debug = {
                    "model_pred": model_labels[pred_idx],
                    "model_quality": round(float(torch.sigmoid(out["quality"]).item()), 4),
                    "p_incorrect": round(float(probs[0]), 4),
                    "p_correct": round(float(probs[1]), 4),
                    "p_review": round(float(probs[2]), 4),
                }
        except Exception as exc:
            model_debug = {"model_debug_error": str(exc)}

    rule_label, reasons = rule_based_label(sample["scalars"], exercise)  # noqa: F405
    final_label = int(rule_label)
    final_name = "correct" if final_label == 1 else "incorrect"

    quality = rule_quality_score(sample["scalars"], exercise)
    if final_label == 0:
        quality = min(quality, 0.49)
    else:
        quality = max(quality, 0.60)

    feedback = generate_feedback(sample["scalars"], exercise=exercise)  # noqa: F405
    feedback_text = feedback[0] if isinstance(feedback, list) else str(feedback)
    torque = compute_torque(quality, hardware_mode)
    hardware = build_hardware_payload(sample, quality, sample["scalars"], torque, hardware_mode)

    scalars = sample["scalars"]
    result = {
        "rep_id": sample.get("rep_id", "?"),
        "exercise": exercise,
        "label": final_label,
        "final": final_name,
        "correct": final_label == 1,
        "quality_score": round(float(quality), 4),
        "feedback": feedback_text,
        "flags": reasons,
        "torque_signal": torque,
        "mode_name": HARDWARE_MODES[hardware_mode]["name"],  # noqa: F405
        "hardware": hardware,
        "scalar_summary": {
            "left_rom_deg": round(float(scalars[0] * 180.0), 2),
            "right_rom_deg": round(float(scalars[1] * 180.0), 2),
            "left_peak_deg": round(float(scalars[2] * 180.0), 2),
            "right_peak_deg": round(float(scalars[3] * 180.0), 2),
            "symmetry_pct": round(float(scalars[4] * 100.0), 2),
            "velocity_deg_s": round(float(scalars[5] * VEL_NORM), 2),  # noqa: F405
            "jerk_deg_s2": round(float(scalars[6] * JERK_NORM), 2),  # noqa: F405
            "lag_frames": round(float(scalars[8] * 150.0), 2),
            "smoothness": round(float(scalars[9]), 4),
            "target_rom_deg": round(float(ROM_TARGETS.get(exercise, ROM_TARGETS["default"])), 2),  # noqa: F405
        },
    }

    if expose_model_debug:
        result["model_debug"] = model_debug

    return result


def run_patient_video_dashboard(
    video_path,
    patient_id,
    model,
    exercise,
    hardware_mode=0,
    device=None,
    fps=25.0,
    overlay_out=None,
    expose_model_debug=False,
):
    if exercise is None or exercise == "unknown":
        raise ValueError("Exercise must be selected manually by the dashboard.")

    device = device or DEVICE  # noqa: F405
    print(f"\nPatient: {patient_id}\nExercise: {exercise}")
    print("[PS1] Processing video...")
    df = ps1_process_video_overlay(video_path, patient_id, exercise=exercise, overlay_out=overlay_out)

    print("[Bridge] Segmenting reps...")
    samples = ps1_df_to_samples(df, patient_id, exercise)  # noqa: F405
    if not samples:
        return {
            "patient": patient_id,
            "exercise": exercise,
            "n_reps": 0,
            "n_correct": 0,
            "avg_quality": 0.0,
            "overlay_video": overlay_out,
            "reps": [],
        }

    results = []
    print(f"[PS2] Dashboard inference on {len(samples)} reps...")
    for sample in samples:
        sample["exercise"] = exercise
        sample["exercise_idx"] = EXERCISE2IDX.get(exercise, EXERCISE2IDX["unknown"])  # noqa: F405
        sample["scalars"] = extract_scalars(sample["keypoints"], exercise=exercise, fps=fps, debug=False)  # noqa: F405
        out = infer_sample_dashboard(
            model,
            sample,
            device=device,
            hardware_mode=hardware_mode,
            expose_model_debug=expose_model_debug,
        )
        print(
            f"  {out['rep_id']:20s} | final={out['final']:9s} "
            f"q={out['quality_score']:.3f} torque={out['torque_signal']:.3f} "
            f"flags={out['flags']} | {out['feedback']}"
        )
        results.append(out)

    n_correct = sum(r["correct"] for r in results)
    avg_quality = float(np.mean([r["quality_score"] for r in results]))
    return {
        "patient": patient_id,
        "exercise": exercise,
        "n_reps": len(results),
        "n_correct": n_correct,
        "pct_correct": round(100.0 * n_correct / max(len(results), 1), 1),
        "avg_quality": round(avg_quality, 3),
        "overlay_video": overlay_out,
        "reps": results,
    }
