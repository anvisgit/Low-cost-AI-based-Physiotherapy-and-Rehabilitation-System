import cv2
import mediapipe as mp
import numpy as np
import csv
import math
import time
import os
from collections import deque
from datetime import datetime
try:
    from scipy.signal import find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("[WARN] scipy not found — rep counter disabled. pip install scipy")

VISIBILITY_THRESHOLD = 0.6          # landmark confidence gate
HIP_ANGLE_WARN       = 40.0         # degrees — posture alert threshold
TRAIL_LENGTH         = 30           # frames of trajectory trail
ROM_GAUGE_RADIUS     = 40           # pixels — ROM arc gauge
REP_PEAK_PROMINENCE  = 15.0         # degrees prominence for find_peaks
REP_PEAK_DISTANCE    = 10           # minimum frames between peaks

JOINT_COLOR  = (0, 255, 0)
BONE_COLOR   = (255, 165, 0)
ANGLE_COLOR  = (255, 255, 255)
WARN_COLOR   = (0, 0, 255)
TRAIL_COLOR  = (0, 200, 255)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def landmark_to_px(landmark, frame_w: int, frame_h: int) -> tuple[int, int]:
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def compute_angle(a, b, c) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def compute_knee_angle_side(hip_px, knee_px, ankle_px) -> float:
    
    return compute_angle(hip_px, knee_px, ankle_px)


def detect_view_mode(lm, mp_pose) -> str:
    l_hip_x = lm[mp_pose.PoseLandmark.LEFT_HIP.value].x
    r_hip_x = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x
    return "side" if abs(l_hip_x - r_hip_x) < 0.06 else "front"
def _vis_ok(lm_item) -> bool:
    return lm_item.visibility >= VISIBILITY_THRESHOLD


LOWER_LIMB_CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,    mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,   mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,   mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,  mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,    mp.solutions.pose.PoseLandmark.RIGHT_HIP),
]

LOWER_JOINTS = {
    "L.Hip":   mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "R.Hip":   mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "L.Knee":  mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "R.Knee":  mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "L.Ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "R.Ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
}


def draw_lower_limb_skeleton(frame, landmarks, frame_w: int, frame_h: int):
    lm = landmarks
    for start_lm, end_lm in LOWER_LIMB_CONNECTIONS:
        s_item = lm[start_lm.value]
        e_item = lm[end_lm.value]
        if not (_vis_ok(s_item) and _vis_ok(e_item)):
            continue
        start = landmark_to_px(s_item, frame_w, frame_h)
        end   = landmark_to_px(e_item, frame_w, frame_h)
        cv2.line(frame, start, end, BONE_COLOR, 3, cv2.LINE_AA)

    for name, lm_id in LOWER_JOINTS.items():
        item = lm[lm_id.value]
        if not _vis_ok(item):
            continue
        px = landmark_to_px(item, frame_w, frame_h)
        cv2.circle(frame, px, 8,  JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, px, 9,  (0, 0, 0),    1, cv2.LINE_AA)

def draw_rom_gauge(frame, center, current_angle: float, max_angle: float,
                   label: str, side: str = "right"):
    r = ROM_GAUGE_RADIUS
    start_angle = -90   
    total_sweep  = 180  

    ox = center[0] + (r + 14) * (1 if side == "right" else -1)
    oy = center[1]
    if max_angle > 0:
        max_sweep = int(min(total_sweep * max_angle / 180.0, total_sweep))
        cv2.ellipse(frame, (ox, oy), (r, r),
                    0, start_angle, start_angle + max_sweep,
                    (80, 80, 80), 6, cv2.LINE_AA)

    if current_angle > 0:
        cur_sweep = int(min(total_sweep * current_angle / 180.0, total_sweep))
        t = current_angle / 180.0
        arc_color = (int(50 + 205 * t), int(255 - 200 * t), 50)
        cv2.ellipse(frame, (ox, oy), (r, r),
                    0, start_angle, start_angle + cur_sweep,
                    arc_color, 4, cv2.LINE_AA)

    cv2.putText(frame, f"{label}", (ox - 12, oy + r + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{current_angle:.0f}°", (ox - 14, oy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, ANGLE_COLOR, 1, cv2.LINE_AA)

class TrajectoryTrail:

    def __init__(self, maxlen: int = TRAIL_LENGTH):
        self._buf: deque[tuple[int, int]] = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def update(self, pt: tuple[int, int]):
        self._buf.append(pt)

    def draw(self, frame):
        pts = list(self._buf)
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (int(TRAIL_COLOR[0] * alpha),
                     int(TRAIL_COLOR[1] * alpha),
                     int(TRAIL_COLOR[2] * alpha))
            cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
class RepCounter:
    def __init__(self, history_size: int = 120):
        self._history: deque[float] = deque(maxlen=history_size)
        self.reps   = 0
        self._last_peak_frame = -99

    def update(self, angle: float, frame_idx: int) -> int:
        self._history.append(angle)
        if not SCIPY_OK or len(self._history) < 20:
            return self.reps

        arr = np.array(self._history)
        valleys, props = find_peaks(-arr,
                                    prominence=REP_PEAK_PROMINENCE,
                                    distance=REP_PEAK_DISTANCE)
        self.reps = len(valleys)
        return self.reps
def draw_posture_alert(frame, hip_angle: float, threshold: float = HIP_ANGLE_WARN):
    if hip_angle > threshold:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "⚠  FORWARD LEAN — STRAIGHTEN BACK  ⚠",
                (w // 2 - 260, h // 2 + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 50), 2, cv2.LINE_AA)
def draw_info_panel(frame, angles: dict, fps: float, bg_mode: bool,
                    frame_idx: int, view_mode: str, reps_l: int, reps_r: int):
    panel_h, panel_w = 245, 330
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    vm_col = (0, 220, 255) if view_mode == "side" else (180, 255, 180)
    lines = [
        ("AI Physiotherapy — MediaPipe v2",  (0, 200, 255)),
        (f"Frame: {frame_idx:05d}   FPS: {fps:.1f}", (200, 200, 200)),
        (f"View: {view_mode.upper():<5}  BG: {'ON' if bg_mode else 'OFF'}",
         vm_col),
        ("", (255, 255, 255)),
        (f"L Knee  : {angles.get('L_knee', 0):.1f}°   Reps: {reps_l}", (100, 255, 100)),
        (f"R Knee  : {angles.get('R_knee', 0):.1f}°   Reps: {reps_r}", (100, 255, 100)),
        (f"L Hip   : {angles.get('L_hip',  0):.1f}°",  (255, 200, 100)),
        (f"R Hip   : {angles.get('R_hip',  0):.1f}°",  (255, 200, 100)),
    ]
    y = 22
    for text, color in lines:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 26

    cv2.putText(frame, "[q]Quit  [b]BG  [r]Reset CSV  [s]Snap",
                (5, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
class MotionLogger:
    FIELDS = [
        "timestamp", "frame", "view_mode",
        "L_knee", "R_knee", "L_hip", "R_hip",
        "Lhip_x",  "Lhip_y",  "Rhip_x",  "Rhip_y",
        "Lknee_x", "Lknee_y", "Rknee_x", "Rknee_y",
        "Lank_x",  "Lank_y",  "Rank_x",  "Rank_y",
        "Lknee_vis", "Rknee_vis",
    ]

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

    def log(self, frame_idx: int, angles: dict, view_mode: str,
            coords: dict | None = None, vis: dict | None = None):
        coords = coords or {}
        vis    = vis    or {}
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame":     frame_idx,
            "view_mode": view_mode,
            "L_knee":    round(angles.get("L_knee", 0), 2),
            "R_knee":    round(angles.get("R_knee", 0), 2),
            "L_hip":     round(angles.get("L_hip",  0), 2),
            "R_hip":     round(angles.get("R_hip",  0), 2),
        }
        for k in ("Lhip_x", "Lhip_y", "Rhip_x", "Rhip_y",
                  "Lknee_x", "Lknee_y", "Rknee_x", "Rknee_y",
                  "Lank_x",  "Lank_y",  "Rank_x",  "Rank_y",
                  "Lknee_vis", "Rknee_vis"):
            row[k] = round(coords.get(k, 0), 4) if k in coords else \
                     round(vis.get(k, 0), 4)
        self.writer.writerow(row)

    def reset(self):
        self._open_new()

    def close(self):
        if self.file:
            self.file.close()
            print(f"[CSV] Saved → {self.filepath}")
class BackgroundHandler:
    def __init__(self, history=120, var_threshold=40):
        self.subtractor   = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process(self, frame):
        fg_mask = self.subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        fg_3ch  = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        bg_mask = cv2.bitwise_not(fg_3ch)
        dimmed  = cv2.addWeighted(frame, 0.25, np.zeros_like(frame), 0.75, 0)
        blended = np.where(bg_mask > 0, dimmed, frame)
        return fg_mask, blended.astype(np.uint8)
def run(source=0, show_full_skeleton: bool = False):
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    bg_handler = BackgroundHandler()
    logger     = MotionLogger()

    # Trajectory trails for knees
    trail_l = TrajectoryTrail()
    trail_r = TrajectoryTrail()

    # Rep counters (one per knee)
    rep_ctr_l = RepCounter()
    rep_ctr_r = RepCounter()

    # Session ROM maxima
    max_rom = {"L_knee": 0.0, "R_knee": 0.0}

    bg_mode    = False
    frame_idx  = 0
    prev_time  = time.time()
    view_mode  = "front"

    print("\n[INFO] Pipeline started.")
    print("       'q'=quit  'b'=BG toggle  'r'=reset CSV  's'=snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame_idx += 1
        frame_w, frame_h = frame.shape[1], frame.shape[0]

        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        if bg_mode:
            _, frame = bg_handler.process(frame)
        else:
            bg_handler.subtractor.apply(frame)          

        proc_frame = preprocess_frame(frame)

        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        angles    = {"L_knee": 0.0, "R_knee": 0.0, "L_hip": 0.0, "R_hip": 0.0}
        coords    = {}
        vis_scores = {}

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            view_mode = detect_view_mode(lm, mp_pose)

            if show_full_skeleton:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(80, 80, 80), thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(60, 60, 60), thickness=1),
                )

            draw_lower_limb_skeleton(frame, lm, frame_w, frame_h)
            def get_px(lm_id):
                item = lm[lm_id.value]
                if _vis_ok(item):
                    return landmark_to_px(item, frame_w, frame_h), item.visibility
                return None, item.visibility

            (L_hip_px,   _)       = get_px(mp_pose.PoseLandmark.LEFT_HIP)
            (R_hip_px,   _)       = get_px(mp_pose.PoseLandmark.RIGHT_HIP)
            (L_knee_px,  lk_vis)  = get_px(mp_pose.PoseLandmark.LEFT_KNEE)
            (R_knee_px,  rk_vis)  = get_px(mp_pose.PoseLandmark.RIGHT_KNEE)
            (L_ankle_px, _)       = get_px(mp_pose.PoseLandmark.LEFT_ANKLE)
            (R_ankle_px, _)       = get_px(mp_pose.PoseLandmark.RIGHT_ANKLE)
            (L_shoulder_px, _)    = get_px(mp_pose.PoseLandmark.LEFT_SHOULDER)
            (R_shoulder_px, _)    = get_px(mp_pose.PoseLandmark.RIGHT_SHOULDER)

            vis_scores["Lknee_vis"] = lk_vis
            vis_scores["Rknee_vis"] = rk_vis

            def norm(px_or_none, axis):
                if px_or_none is None:
                    return 0.0
                return px_or_none[0] / frame_w if axis == "x" else px_or_none[1] / frame_h

            coords.update({
                "Lhip_x":  norm(L_hip_px,    "x"), "Lhip_y":  norm(L_hip_px,    "y"),
                "Rhip_x":  norm(R_hip_px,    "x"), "Rhip_y":  norm(R_hip_px,    "y"),
                "Lknee_x": norm(L_knee_px,   "x"), "Lknee_y": norm(L_knee_px,   "y"),
                "Rknee_x": norm(R_knee_px,   "x"), "Rknee_y": norm(R_knee_px,   "y"),
                "Lank_x":  norm(L_ankle_px,  "x"), "Lank_y":  norm(L_ankle_px,  "y"),
                "Rank_x":  norm(R_ankle_px,  "x"), "Rank_y":  norm(R_ankle_px,  "y"),
            })
            if L_knee_px and L_hip_px and L_ankle_px:
                if view_mode == "side":
                    angles["L_knee"] = compute_knee_angle_side(L_hip_px, L_knee_px, L_ankle_px)
                else:
                    angles["L_knee"] = compute_angle(L_hip_px, L_knee_px, L_ankle_px)

            if R_knee_px and R_hip_px and R_ankle_px:
                if view_mode == "side":
                    angles["R_knee"] = compute_knee_angle_side(R_hip_px, R_knee_px, R_ankle_px)
                else:
                    angles["R_knee"] = compute_angle(R_hip_px, R_knee_px, R_ankle_px)

            if L_hip_px and L_shoulder_px and L_knee_px:
                angles["L_hip"] = compute_angle(L_shoulder_px, L_hip_px, L_knee_px)

            if R_hip_px and R_shoulder_px and R_knee_px:
                angles["R_hip"] = compute_angle(R_shoulder_px, R_hip_px, R_knee_px)
            max_rom["L_knee"] = max(max_rom["L_knee"], angles["L_knee"])
            max_rom["R_knee"] = max(max_rom["R_knee"], angles["R_knee"])
            if L_knee_px:
                cv2.putText(frame, f"LK:{angles['L_knee']:.0f}°",
                            (L_knee_px[0]+12, L_knee_px[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
            if R_knee_px:
                cv2.putText(frame, f"RK:{angles['R_knee']:.0f}°",
                            (R_knee_px[0]+12, R_knee_px[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
            if L_hip_px:
                cv2.putText(frame, f"LH:{angles['L_hip']:.0f}°",
                            (L_hip_px[0]+12, L_hip_px[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)
            if R_hip_px:
                cv2.putText(frame, f"RH:{angles['R_hip']:.0f}°",
                            (R_hip_px[0]+12, R_hip_px[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)

            if L_knee_px:
                trail_l.update(L_knee_px)
            if R_knee_px:
                trail_r.update(R_knee_px)
            trail_l.draw(frame)
            trail_r.draw(frame)
            if L_knee_px:
                draw_rom_gauge(frame, L_knee_px, angles["L_knee"],
                               max_rom["L_knee"], "L-ROM", side="left")
            if R_knee_px:
                draw_rom_gauge(frame, R_knee_px, angles["R_knee"],
                               max_rom["R_knee"], "R-ROM", side="right")
            min_hip = min(angles["L_hip"], angles["R_hip"])
            draw_posture_alert(frame, min_hip)

        reps_l = rep_ctr_l.update(angles["L_knee"], frame_idx)
        reps_r = rep_ctr_r.update(angles["R_knee"], frame_idx)

        draw_info_panel(frame, angles, fps, bg_mode, frame_idx,
                        view_mode, reps_l, reps_r)

        logger.log(frame_idx, angles, view_mode, coords, vis_scores)

        cv2.imshow("MediaPipe Lower Limb v2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit requested.")
            break
        elif key == ord('b'):
            bg_mode = not bg_mode
            print(f"[BG] Background subtraction {'ON' if bg_mode else 'OFF'}")
        elif key == ord('r'):
            logger.reset()
            rep_ctr_l = RepCounter()
            rep_ctr_r = RepCounter()
            max_rom   = {"L_knee": 0.0, "R_knee": 0.0}
            print("[INFO] CSV + counters + ROM reset.")
        elif key == ord('s'):
            snap = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(snap, frame)
            print(f"[SNAP] Saved → {snap}")

    cap.release()
    pose.close()
    logger.close()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    import sys
    source  = 0
    full_sk = False
    if len(sys.argv) >= 2:
        arg    = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg
    if len(sys.argv) >= 3 and sys.argv[2] == "full":
        full_sk = True
    run(source=source, show_full_skeleton=full_sk)
