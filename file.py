"""
AI-Based Physiotherapy System — MediaPipe Module
=================================================
Responsibilities:
  A. Pose Estimation     — MediaPipe Pose, 33 landmarks, real-time webcam
  B. Joint Angle Calc    — Hip, Knee angles (geometric / dot-product method)
  C. Movement Tracking   — On-screen skeleton overlay + angle display
  D. Background Handling — MOG2 background subtraction (optional, toggle with 'b')
  E. Data Collection     — CSV logging of per-frame joint angles

Controls (while running):
  q  → quit
  b  → toggle background subtraction overlay
  r  → reset CSV / start new session
  s  → take snapshot (saves frame as PNG)

Output files:
  motion_log_<timestamp>.csv   — frame-by-frame angle data
  snapshot_<timestamp>.png     — manual snapshots
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import math
import time
import os
from datetime import datetime


# ──────────────────────────────────────────────
#  GEOMETRY HELPERS
# ──────────────────────────────────────────────

def landmark_to_px(landmark, frame_w, frame_h):
    """Convert normalised MediaPipe landmark → pixel (x, y)."""
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def compute_angle(a, b, c):
    """
    Angle at joint B formed by segments B→A and B→C.
    Uses dot-product: θ = arccos( (BA · BC) / (|BA| |BC|) )
    Returns degrees in [0, 180].
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


# ──────────────────────────────────────────────
#  DRAWING HELPERS
# ──────────────────────────────────────────────

LOWER_LIMB_CONNECTIONS = [
    # Left leg
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,   mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,  mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    # Right leg
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,  mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    # Hip connector
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,   mp.solutions.pose.PoseLandmark.RIGHT_HIP),
]

JOINT_COLOR   = (0, 255, 0)        # green  — joint circles
BONE_COLOR    = (255, 165, 0)      # orange — limb lines
ANGLE_COLOR   = (255, 255, 255)    # white  — angle text
LABEL_COLOR   = (0, 200, 255)      # cyan   — joint name labels
BG_OVERLAY    = (30, 30, 30)       # dark   — semi-transparent BG panel


def draw_lower_limb_skeleton(frame, landmarks, frame_w, frame_h):
    """Draw only the lower-limb bones and joint circles."""
    # Draw bones
    for start_lm, end_lm in LOWER_LIMB_CONNECTIONS:
        start = landmark_to_px(landmarks[start_lm.value], frame_w, frame_h)
        end   = landmark_to_px(landmarks[end_lm.value],   frame_w, frame_h)
        cv2.line(frame, start, end, BONE_COLOR, 3, cv2.LINE_AA)

    # Draw joint circles
    lower_joints = {
        "L.Hip":   mp.solutions.pose.PoseLandmark.LEFT_HIP,
        "R.Hip":   mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        "L.Knee":  mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        "R.Knee":  mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        "L.Ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        "R.Ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    }
    for name, lm_id in lower_joints.items():
        px = landmark_to_px(landmarks[lm_id.value], frame_w, frame_h)
        cv2.circle(frame, px, 8, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, px, 9, (0, 0, 0), 1, cv2.LINE_AA)   # thin black ring


def draw_angle_arc(frame, vertex, angle_deg, label, color=(0, 255, 255)):
    """Draw the computed angle value near the joint."""
    text = f"{label}: {angle_deg:.1f}°"
    cv2.putText(frame, text,
                (vertex[0] + 12, vertex[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_info_panel(frame, angles, fps, bg_mode, frame_idx):
    """Semi-transparent HUD in top-left corner."""
    panel_h, panel_w = 220, 310
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        ("AI Physiotherapy — MediaPipe", (0, 200, 255)),
        (f"Frame: {frame_idx:05d}   FPS: {fps:.1f}", (200, 200, 200)),
        (f"BG Subtract: {'ON' if bg_mode else 'OFF'}", (0, 255, 150) if bg_mode else (180, 180, 180)),
        ("",                                          (255, 255, 255)),
        (f"L Knee  : {angles.get('L_knee',  0):.1f}°", (100, 255, 100)),
        (f"R Knee  : {angles.get('R_knee',  0):.1f}°", (100, 255, 100)),
        (f"L Hip   : {angles.get('L_hip',   0):.1f}°", (255, 200, 100)),
        (f"R Hip   : {angles.get('R_hip',   0):.1f}°", (255, 200, 100)),
    ]
    y = 22
    for text, color in lines:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 24

    # Controls reminder at bottom
    cv2.putText(frame, "[q] Quit  [b] BG  [r] Reset CSV  [s] Snap",
                (5, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
#  CSV LOGGER
# ──────────────────────────────────────────────

class MotionLogger:
    """Writes per-frame joint angles to a CSV file."""

    FIELDS = ["timestamp", "frame", "L_knee", "R_knee", "L_hip", "R_hip"]

    def __init__(self):
        self.filepath = None
        self.file     = None
        self.writer   = None
        self._open_new()

    def _open_new(self):
        if self.file:
            self.file.close()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"motion_log_{ts}.csv"
        self.file   = open(self.filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        print(f"[CSV] Logging to: {self.filepath}")

    def log(self, frame_idx, angles):
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame":     frame_idx,
            "L_knee":    round(angles.get("L_knee", 0), 2),
            "R_knee":    round(angles.get("R_knee", 0), 2),
            "L_hip":     round(angles.get("L_hip",  0), 2),
            "R_hip":     round(angles.get("R_hip",  0), 2),
        }
        self.writer.writerow(row)

    def reset(self):
        self._open_new()

    def close(self):
        if self.file:
            self.file.close()
            print(f"[CSV] Saved → {self.filepath}")


# ──────────────────────────────────────────────
#  BACKGROUND SUBTRACTOR
# ──────────────────────────────────────────────

class BackgroundHandler:
    """
    Wraps OpenCV MOG2 background subtractor.
    Produces a foreground mask that can be blended onto the frame
    to reduce clutter and highlight the subject.
    """

    def __init__(self, history=120, var_threshold=40):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        # Morphological kernels for mask cleanup
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process(self, frame):
        """
        Returns (mask, blended_frame).
        mask          — binary uint8 foreground mask
        blended_frame — original frame dimmed where background is detected
        """
        fg_mask = self.subtractor.apply(frame)

        # Clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        # Dim the background (multiply frame by 0.25 where mask == 0)
        fg_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        background_region = cv2.bitwise_not(fg_3ch)
        dimmed_bg = cv2.addWeighted(frame, 0.25, np.zeros_like(frame), 0.75, 0)
        blended = np.where(background_region > 0, dimmed_bg, frame)

        return fg_mask, blended.astype(np.uint8)


# ──────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────

def run(source=0, show_full_skeleton=False):
    """
    Main loop.

    Parameters
    ----------
    source : int or str
        0 for webcam, or path to a video file.
    show_full_skeleton : bool
        If True, also draws MediaPipe's full-body connections in addition
        to the highlighted lower-limb overlay.
    """

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # ── MediaPipe Pose ──────────────────────────────
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    bg_handler = BackgroundHandler()
    logger     = MotionLogger()

    bg_mode   = False    # toggle with 'b'
    frame_idx = 0
    prev_time = time.time()

    print("\n[INFO] Pipeline started.")
    print("       Press 'q' to quit, 'b' BG toggle, 'r' reset CSV, 's' snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame_idx += 1
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        # ── FPS ────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ── Background Subtraction (optional) ──────────
        if bg_mode:
            _, frame = bg_handler.process(frame)
        else:
            # Still update the model so it's ready when toggled on
            bg_handler.subtractor.apply(frame)

        # ── MediaPipe Inference ─────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        angles = {"L_knee": 0.0, "R_knee": 0.0, "L_hip": 0.0, "R_hip": 0.0}

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Optional: full-body skeleton (faint) for reference
            if show_full_skeleton:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(80, 80, 80), thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(60, 60, 60), thickness=1),
                )

            # ── Lower-limb skeleton overlay ─────────────
            draw_lower_limb_skeleton(frame, lm, frame_w, frame_h)

            # ── Extract key pixel coordinates ────────────
            L_hip   = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_HIP.value],    frame_w, frame_h)
            R_hip   = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HIP.value],   frame_w, frame_h)
            L_knee  = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_KNEE.value],   frame_w, frame_h)
            R_knee  = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value],  frame_w, frame_h)
            L_ankle = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value],  frame_w, frame_h)
            R_ankle = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], frame_w, frame_h)

            # Need shoulder for hip-angle calculation (hip flexion = torso–thigh)
            L_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],  frame_w, frame_h)
            R_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], frame_w, frame_h)

            # ── Angle Computation ────────────────────────
            #   Knee angle: Hip → Knee ← Ankle
            angles["L_knee"] = compute_angle(L_hip,   L_knee, L_ankle)
            angles["R_knee"] = compute_angle(R_hip,   R_knee, R_ankle)

            #   Hip angle:  Shoulder → Hip ← Knee
            angles["L_hip"]  = compute_angle(L_shoulder, L_hip, L_knee)
            angles["R_hip"]  = compute_angle(R_shoulder, R_hip, R_knee)

            # ── Draw angle labels near joints ────────────
            draw_angle_arc(frame, L_knee, angles["L_knee"], "LK", (100, 255, 100))
            draw_angle_arc(frame, R_knee, angles["R_knee"], "RK", (100, 255, 100))
            draw_angle_arc(frame, L_hip,  angles["L_hip"],  "LH", (255, 200, 100))
            draw_angle_arc(frame, R_hip,  angles["R_hip"],  "RH", (255, 200, 100))

        # ── HUD ──────────────────────────────────────────
        draw_info_panel(frame, angles, fps, bg_mode, frame_idx)

        # ── CSV Logging ───────────────────────────────────
        logger.log(frame_idx, angles)

        # ── Display ───────────────────────────────────────
        cv2.imshow("AI Physiotherapy — MediaPipe Lower-Limb", frame)

        # ── Key Handling ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit requested.")
            break
        elif key == ord('b'):
            bg_mode = not bg_mode
            print(f"[BG] Background subtraction {'ON' if bg_mode else 'OFF'}")
        elif key == ord('r'):
            logger.reset()
        elif key == ord('s'):
            snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(snap_name, frame)
            print(f"[SNAP] Saved → {snap_name}")

    # ── Cleanup ───────────────────────────────────────────
    cap.release()
    pose.close()
    logger.close()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Usage:
    #   python mediapipe_physiotherapy.py           → webcam (default)
    #   python mediapipe_physiotherapy.py video.mp4 → video file
    #   python mediapipe_physiotherapy.py 0 full    → webcam + full skeleton

    source = 0
    full_sk = False

    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg
    if len(sys.argv) >= 3 and sys.argv[2] == "full":
        full_sk = True

    run(source=source, show_full_skeleton=full_sk)