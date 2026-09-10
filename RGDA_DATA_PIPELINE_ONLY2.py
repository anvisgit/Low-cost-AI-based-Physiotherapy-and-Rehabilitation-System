# Data/preprocessing subset extracted from REHAB_PIPELINE_CLEAN_FIXED_V4.py.
# Use this before rgda_rehabnet_architecture.py for RGDA training.
# It intentionally excludes the old RehabNet/ST-GCN/Transformer architecture.

# ===== Source code block 3 =====

import os, json, time, copy, pickle, urllib.request, logging
from datetime import datetime
from typing import List, Optional
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import pearsonr

# â”€â”€ Environment detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IN_COLAB = False
try:
    import google.colab  # noqa
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/content/drive')
    print('Running in Colab â€” Drive mounted')
except ImportError:
    print('Running in VS Code / local Jupyter')

# â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# â”€â”€ Data paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Edit these to match where your CSV files actually are.
if IN_COLAB:
    DATA_DIR   = '/content/drive/MyDrive/datasets'
    MODEL_SAVE = '/content/drive/MyDrive/models/rehabnet_best.pth'
else:
    DATA_DIR   = './datasets'          # local folder next to this file
    MODEL_SAVE = './models/rehabnet_best.pth'

UIPRMD_CSV = os.path.join(DATA_DIR, 'uiprmd.csv')
KIMORE_CSV = os.path.join(DATA_DIR, 'kimore.csv')
KERAAL_CSV = os.path.join(DATA_DIR, 'keraal.csv')

os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

# â”€â”€ Sequence constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TARGET_LEN  = 150
N_JOINTS    = 6
IN_CHANNELS = 3
MIN_FRAMES  = 20
FPS         = 30
N_EPOCHS    = 60
EPS         = 1e-6

# â”€â”€ Joint order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
JOINT_ORDER = ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']

# â”€â”€ Exercise list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SINGLE definition â€” RehabNet._N_EXERCISES must equal len(EXERCISE_LIST)
EXERCISE_LIST = [
    'deep_squat', 'hurdle_step', 'inline_lunge', 'side_lunge',
    'sit_to_stand', 'straight_leg_raise',
    'squat', 'ctk_squat', 'knee_bend', 'unknown'
]
EXERCISE2IDX = {e: i for i, e in enumerate(EXERCISE_LIST)}
N_EXERCISES  = len(EXERCISE_LIST)   # 10  â† matches RehabNet._N_EXERCISES below

OLD_EXERCISE_LIST = [
    'deep_squat', 'hurdle_step', 'inline_lunge', 'side_lunge',
    'sit_to_stand', 'straight_leg_raise',
    'squat', 'ctk_squat', 'unknown'
]


def adapt_exercise_weights(state, old_exercises=OLD_EXERCISE_LIST, new_exercises=EXERCISE_LIST):
    """
    Adapt older checkpoints trained with 9 exercise classes to the current
    10-class exercise list. Matching exercise rows are copied by name; any new
    row, currently knee_bend, is initialized from unknown/mean weights.
    """
    if not isinstance(state, dict):
        return state

    state = dict(state)
    old_idx = {name: i for i, name in enumerate(old_exercises)}
    new_idx = {name: i for i, name in enumerate(new_exercises)}

    def _fallback_row(tensor):
        if 'unknown' in old_idx and old_idx['unknown'] < tensor.shape[0]:
            return tensor[old_idx['unknown']].clone()
        return tensor.mean(dim=0)

    def _adapt_rows(key, target_rows):
        if key not in state:
            return
        tensor = state[key]
        if not hasattr(tensor, 'shape') or len(tensor.shape) == 0:
            return
        if tensor.shape[0] == target_rows:
            return

        new_shape = (target_rows,) + tuple(tensor.shape[1:])
        new_tensor = tensor.new_empty(new_shape)
        fallback = _fallback_row(tensor)

        for ex_name, ni in new_idx.items():
            if ex_name in old_idx and old_idx[ex_name] < tensor.shape[0]:
                new_tensor[ni] = tensor[old_idx[ex_name]]
            else:
                new_tensor[ni] = fallback

        state[key] = new_tensor
        print(f"Adapted checkpoint tensor {key}: {tuple(tensor.shape)} -> {tuple(new_tensor.shape)}")

    _adapt_rows('ex_emb.weight', len(new_exercises))
    _adapt_rows('ex_head.weight', len(new_exercises))
    _adapt_rows('ex_head.bias', len(new_exercises))
    return state

# â”€â”€ Hardware modes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HARDWARE_MODES = {
    0: {'name': 'assist', 'scale': -0.3, 'bias': 0.5, 'min': 0.0, 'max': 0.8},
    1: {'name': 'resist', 'scale':  0.6, 'bias': 0.1, 'min': 0.0, 'max': 0.8},
}
N_MODES = len(HARDWARE_MODES)   # 2

# â”€â”€ Rep segmentation: minimum inter-peak distance per exercise â”€â”€â”€â”€â”€â”€â”€â”€â”€
EX_MIN_DIST = {
    'inline_lunge': 100,
    'side_lunge':   100,
    'deep_squat':    60,
    'sit_to_stand':  80,
    'straight_leg_raise': 40,
    'knee_bend':     60,
    'squat':         60,
    'ctk_squat':     60,
    'hurdle_step':   45,
    'unknown':       60,
    'default':       60,
}

# â”€â”€ Deficit targets per exercise (degrees) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TARGET_ANGLES = {
    'deep_squat': 90., 'inline_lunge': 90., 'side_lunge': 90.,
    'sit_to_stand': 100., 'straight_leg_raise': 170.,
    'knee_bend': 90., 'squat': 90., 'default': 90.,
}

# â”€â”€ Feedback rules (priority-ordered, highest = most important) â”€â”€â”€â”€â”€â”€â”€â”€
# scalars index map:
#   0=l_rom/180  1=r_rom/180  2=l_peak/180  3=r_peak/180
#   4=sym/100    5=vel/200    6=jerk/10     7=trunk
#   8=lag/150    9=smooth
FEEDBACK_RULES = [
    {'name': 'insufficient_ROM',   'priority': 10,
     'check':   lambda s: min(s[0], s[1]) * 180 < 50,
     'message': 'Bend your knee a little further â€” try to reach 90Â°.',
     'short':   'Bend knee further'},
    {'name': 'too_fast',           'priority': 9,
     'check':   lambda s: s[5] * 200 > 80,
     'message': 'Slow down â€” take about 3 seconds for each bend.',
     'short':   'Move slower'},
    {'name': 'high_jerk',          'priority': 8,
     'check':   lambda s: s[6] * 10 > 8,
     'message': "Don't jerk your knee â€” keep the movement smooth.",
     'short':   'Avoid jerking'},
    {'name': 'trunk_compensation', 'priority': 7,
     'check':   lambda s: s[7] > 0.35,
     'message': 'Keep your back straight â€” try not to lean forward.',
     'short':   'Keep back straight'},
    {'name': 'asymmetric',         'priority': 6,
     'check':   lambda s: s[4] * 100 > 25,
     'message': 'Try to put equal weight on both legs.',
     'short':   'Equal weight both legs'},
    {'name': 'temporal_lag',       'priority': 5,
     'check':   lambda s: s[8] * 150 > 15,
     'message': 'Keep both legs moving at the same time.',
     'short':   'Sync both legs'},
]

# â”€â”€ MediaPipe model path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MP_MODEL_PATH = '/tmp/pose_landmarker.task'
MP_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                 'pose_landmarker/pose_landmarker_heavy/float16/'
                 'latest/pose_landmarker_heavy.task')

print(f'Constants loaded. N_EXERCISES={N_EXERCISES}  N_MODES={N_MODES}')



# ===== Source code block 5 =====

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# A. SHARED HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def safe_np(x, fill=0.0, clip=None, dtype=np.float32):
    x = np.asarray(x, dtype=dtype)
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    if clip is not None:
        x = np.clip(x, -clip, clip)
    return x.astype(dtype)

def valid_np(x):
    x = np.asarray(x)
    return x.size > 0 and np.isfinite(x).all()

def _smooth(arr, win=7, poly=2):
    arr = safe_np(arr)
    if len(arr) < win + 2:
        return arr
    w = win if win % 2 == 1 else win + 1
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    try:
        return safe_np(savgol_filter(arr, w, poly))
    except Exception:
        return arr

def _resample(arr, n):
    arr = safe_np(arr)
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.zeros(n, dtype=np.float32)
    return safe_np(np.interp(np.linspace(0, 1, n),
                              np.linspace(0, 1, len(arr)), arr))

def _fill_missing(kps):
    kps = safe_np(kps)
    T, J, C = kps.shape
    for j in range(J):
        for c in range(C):
            col = kps[:, j, c]
            z   = (~np.isfinite(col)) | (col == 0.0)
            if z.all():
                opp = j + 3 if j < 3 else j - 3
                kps[:, j, c] = kps[:, opp, c]
            elif z.any() and (~z).sum() >= 2:
                f = interp1d(np.where(~z)[0], col[~z],
                             bounds_error=False, fill_value='extrapolate')
                kps[:, j, c] = f(np.arange(T))
    return safe_np(kps)

def _normalise_kps(kps):
    kps  = safe_np(kps)
    hip  = ((kps[:, 0, :] + kps[:, 3, :]) / 2.0).mean(axis=0)
    kps  = kps - hip
    l    = np.mean(np.linalg.norm(kps[:, 0, :] - kps[:, 2, :], axis=1))
    r    = np.mean(np.linalg.norm(kps[:, 3, :] - kps[:, 5, :], axis=1))
    scale = (l + r) / 2.0
    if not np.isfinite(scale) or scale < 1e-4:
        scale = 1.0
    return safe_np(np.clip(kps / scale, -5.0, 5.0), clip=5.0)

def build_adjacency():
    A = np.zeros((N_JOINTS, N_JOINTS), dtype=np.float32)
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]:
        A[i, j] = A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    d = np.diag(1.0 / np.sqrt(A.sum(axis=1) + EPS))
    return (d @ A @ d).astype(np.float32)

ADJ = build_adjacency()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# B. PS1 â€” VIDEO â†’ DATAFRAME
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MIN_DETECT_CONF = 0.5
MIN_TRACK_CONF  = 0.5

JOINT_COLS = []
for _jn in ['HipLeft', 'KneeLeft', 'AnkleLeft', 'HipRight', 'KneeRight', 'AnkleRight']:
    for _ax in ['x', 'y', 'z']:
        JOINT_COLS.append(f'{_jn}_{_ax}')
BASE_COLS = ['subject_id', 'session', 'label', 'frame_id', 'time_s', 'detection']

def ps1_prepare_frame(frame, target_w=640):
    h, w = frame.shape[:2]
    if w > target_w:
        frame = cv2.resize(frame, (target_w, int(h * target_w / w)))
    return frame

def ps1_extract_row(landmarks):
    mp_map = {
        'HipLeft': 23, 'KneeLeft': 25, 'AnkleLeft': 27,
        'HipRight': 24, 'KneeRight': 26, 'AnkleRight': 28,
    }
    row = {}
    for jname, idx in mp_map.items():
        lm = landmarks[idx]
        row[f'{jname}_x'] = float(lm.x)
        row[f'{jname}_y'] = float(lm.y)
        row[f'{jname}_z'] = float(lm.z)
    return row

def ps1_root_center(df):
    for ax in ['x', 'y', 'z']:
        mid = (df[f'HipLeft_{ax}'] + df[f'HipRight_{ax}']) / 2.0
        for jn in ['HipLeft', 'KneeLeft', 'AnkleLeft',
                   'HipRight', 'KneeRight', 'AnkleRight']:
            df[f'{jn}_{ax}'] = df[f'{jn}_{ax}'] - mid
    return df

def ps1_bone_normalize(df):
    for side, (h, k, a) in [
        ('L', ('HipLeft',  'KneeLeft',  'AnkleLeft')),
        ('R', ('HipRight', 'KneeRight', 'AnkleRight')),
    ]:
        for ax in ['x', 'y', 'z']:
            thigh = np.sqrt(((df[f'{h}_{ax}'] - df[f'{k}_{ax}']) ** 2).mean()) + EPS
            for jname in [h, k, a]:
                df[f'{jname}_{ax}'] = df[f'{jname}_{ax}'] / thigh
    return df

def _knee_angle_from_df(df, side='L'):
    prefix = {
        'L': ('HipLeft',  'KneeLeft',  'AnkleLeft'),
        'R': ('HipRight', 'KneeRight', 'AnkleRight'),
    }[side]
    h  = df[[f'{prefix[0]}_{ax}' for ax in 'xyz']].values
    k  = df[[f'{prefix[1]}_{ax}' for ax in 'xyz']].values
    a  = df[[f'{prefix[2]}_{ax}' for ax in 'xyz']].values
    v1 = h - k
    v2 = a - k
    cos = (np.sum(v1 * v2, axis=1) /
           (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + EPS))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def ps1_extract_features(df):
    df['angle_knee_L']        = _knee_angle_from_df(df, 'L')
    df['angle_knee_R']        = _knee_angle_from_df(df, 'R')
    df['angle_knee_L_smooth'] = _smooth(df['angle_knee_L'].values)
    df['angle_knee_R_smooth'] = _smooth(df['angle_knee_R'].values)
    return df

def ps1_clean_nulls(df):
    return df.ffill().bfill().fillna(0.0)

def _ensure_mp_model():
    if not os.path.exists(MP_MODEL_PATH):
        print('  Downloading MediaPipe pose model (~30 MB)...')
        urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
        print('  Downloaded.')

def ps1_process_video(video_path, patient_id,
                      exercise='unknown', session='1', label='unknown'):
    """
    VIDEO FILE â†’ processed DataFrame  (one row per detected frame).
    Works on any mp4/avi/webm file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    fps_vid    = cap.get(cv2.CAP_PROP_FPS) or FPS
    rows, frame_idx, detected = [], 0, 0
    t0 = time.time()

    _ensure_mp_model()

    import mediapipe as mp
    from mediapipe.tasks import python as _mptasks
    from mediapipe.tasks.python import vision as _mpvision

    opts = _mpvision.PoseLandmarkerOptions(
        base_options=_mptasks.BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=_mpvision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECT_CONF,
        min_pose_presence_confidence=MIN_DETECT_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
        output_segmentation_masks=False,
    )

    frame_step_ms = int(1000.0 / fps_vid)

    with _mpvision.PoseLandmarker.create_from_options(opts) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame_ms   = frame_idx * frame_step_ms
            prepped    = ps1_prepare_frame(frame)
            rgb        = cv2.cvtColor(prepped, cv2.COLOR_BGR2RGB)
            mp_img     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res        = pose.detect_for_video(mp_img, frame_ms)

            row = {'subject_id': patient_id, 'session': session,
                   'label': label, 'frame_id': frame_idx,
                   'time_s': round(frame_idx / fps_vid, 4), 'detection': 0}
            row.update({col: float('nan') for col in JOINT_COLS})

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                detected += 1
                row['detection'] = 1
                row.update(ps1_extract_row(res.pose_landmarks[0]))
            rows.append(row)

    cap.release()

    if frame_idx == 0:
        raise ValueError('Video is empty')

    det_pct = 100 * detected / frame_idx
    print(f'  PS1: {frame_idx} frames  detection={det_pct:.0f}%  '
          f'{time.time()-t0:.1f}s')
    if det_pct < 20:
        print(f'  WARNING: low detection ({det_pct:.0f}%) â€” '
              f'check lighting and camera angle')

    df = pd.DataFrame(rows, columns=BASE_COLS + JOINT_COLS)
    df = df[df['detection'] == 1].reset_index(drop=True)
    df = ps1_root_center(df)
    df = ps1_bone_normalize(df)
    df = ps1_extract_features(df)
    df = ps1_clean_nulls(df)
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# C. KEYPOINT PROCESSING  (shared by PS1 pipeline and dataset loaders)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_keypoints(raw):
    """
    raw: np.ndarray shape (T, N_JOINTS, 3)
    Returns: np.ndarray shape (TARGET_LEN, N_JOINTS, 3), normalised
    """
    raw = safe_np(raw)
    if raw.ndim != 3 or raw.shape[1] != N_JOINTS or raw.shape[2] != 3:
        raise ValueError(f'Expected (T,{N_JOINTS},3), got {raw.shape}')
    if len(raw) < MIN_FRAMES:
        raise ValueError(f'Too short: {len(raw)} frames (min={MIN_FRAMES})')
    for j in range(N_JOINTS):
        for c in range(3):
            raw[:, j, c] = _smooth(raw[:, j, c])
    raw = _fill_missing(raw)
    out = np.zeros((TARGET_LEN, N_JOINTS, 3), dtype=np.float32)
    for j in range(N_JOINTS):
        for c in range(3):
            out[:, j, c] = _resample(raw[:, j, c], TARGET_LEN)
    return safe_np(_normalise_kps(out), clip=5.0)

def extract_scalars(kps):
    """
    kps: np.ndarray (TARGET_LEN, N_JOINTS, 3)
    Returns: np.ndarray (10,) normalised scalar features
    """
    kps = safe_np(kps, clip=5.0)

    def _ang(a, b, c):
        ba, bc = a - b, c - b
        d = np.maximum(np.linalg.norm(ba, 1) * np.linalg.norm(bc, 1), EPS)
        return np.degrees(np.arccos(np.clip(np.sum(ba * bc, 1) / d, -1, 1)))

    lk  = _ang(kps[:, 0, :], kps[:, 1, :], kps[:, 2, :])
    rk  = _ang(kps[:, 3, :], kps[:, 4, :], kps[:, 5, :])
    lr  = float(np.nanmax(lk) - np.nanmin(lk))
    rr  = float(np.nanmax(rk) - np.nanmin(rk))
    sym = abs(lr - rr) / ((lr + rr) / 2 + EPS) * 100 if (lr + rr) > 1 else 0.0
    vel = float(np.max(np.abs(np.diff(lk))) * FPS) if len(lk) > 1 else 0.0

    hip  = (kps[:, 0, :] + kps[:, 3, :]) / 2.0
    jk   = np.diff(np.diff(hip, axis=0), axis=0)
    jerk = float(np.nanmean(np.linalg.norm(jk, axis=1))) if len(jk) else 0.0

    pk    = int(np.argmin(lk))
    trunk = float(abs(hip[pk, 0] - hip[0, 0]))

    lkc, rkc = lk - np.nanmean(lk), rk - np.nanmean(rk)
    lag = float(abs(
        np.argmax(np.correlate(lkc, rkc, 'full')) - (len(lk) - 1)
    )) if np.nanstd(lkc) > EPS and np.nanstd(rkc) > EPS else 0.0

    smooth = float(np.clip(1.0 / (np.nanstd(np.diff(lk)) + EPS) / 100, 0, 1))

    return safe_np(np.array([
        np.clip(lr / 180,          0, 1),
        np.clip(rr / 180,          0, 1),
        np.clip(np.min(lk) / 180,  0, 1),
        np.clip(np.min(rk) / 180,  0, 1),
        np.clip(sym / 100,         0, 5),
        np.clip(vel / 200,         0, 5),
        np.clip(jerk / 10,         0, 5),
        np.clip(trunk,             0, 5),
        np.clip(lag / TARGET_LEN,  0, 1),
        smooth,
    ], dtype=np.float32), fill=0.0, clip=10.0)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# D. BRIDGE â€” DATAFRAME â†’ PER-REP SAMPLE DICTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _df_to_kps(df):
    cols = []
    for jn in ['HipLeft', 'KneeLeft', 'AnkleLeft',
               'HipRight', 'KneeRight', 'AnkleRight']:
        cols += [f'{jn}_x', f'{jn}_y', f'{jn}_z']
    return df[cols].values.astype(np.float32).reshape(len(df), N_JOINTS, 3)

def _segment_reps(df, angle_col='angle_knee_L_smooth',
                  min_frames=MIN_FRAMES, min_rom=20.0, exercise='unknown'):
    """Split a continuous DataFrame into per-rep DataFrames."""
    if angle_col in df.columns:
        angles = df[angle_col].ffill().bfill().values
    else:
        angles = _knee_angle_from_df(df, 'L')

    n = len(angles)
    ex_key = exercise if exercise in EX_MIN_DIST else 'default'
    min_dist = max(EX_MIN_DIST.get(ex_key, EX_MIN_DIST['default']), n // 20)
    width = max(3, min_dist // 4)

    peaks, _ = find_peaks(
        -angles,
        distance=min_dist,
        prominence=min_rom * 0.5,
        width=width,
    )

    print(f'  Segmenter: exercise={exercise}  n_frames={n}  '
          f'min_dist={min_dist}  peaks={len(peaks)}')

    if len(peaks) == 0:
        return [df]

    mids   = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    bounds = [0] + mids + [n]
    reps   = [df.iloc[bounds[i]:bounds[i + 1]].copy()
              for i in range(len(bounds) - 1)
              if len(df.iloc[bounds[i]:bounds[i + 1]]) >= min_frames]
    return reps if reps else [df]

def ps1_df_to_samples(df, patient_id, exercise='unknown',
                       label=-1, source='patient_video'):
    """PS1 DataFrame â†’ list of sample dicts ready for RehabNet."""
    reps   = _segment_reps(df, exercise=exercise)
    ex_idx = EXERCISE2IDX.get(exercise, EXERCISE2IDX['unknown'])
    print(f'  Bridge: {len(reps)} rep segments â†’ ', end='')
    samples = []
    for i, rep_df in enumerate(reps):
        try:
            kps = _df_to_kps(rep_df)
            kps = process_keypoints(kps)
            sc  = extract_scalars(kps)
            samples.append({
                'keypoints':    kps,
                'adj':          ADJ,
                'scalars':      sc,
                'label':        max(label, 0),
                'quality':      float(max(label, 0)),
                'exercise':     exercise,
                'exercise_idx': ex_idx,
                'subject':      patient_id,
                'source':       source,
                'rep_id':       f'{patient_id}_r{i + 1}',
            })
        except Exception as e:
            print(f'\n    Rep {i + 1} skipped: {e}')
    print(f'{len(samples)} valid')
    return samples


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# E. DATASET LOADERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _kps_from_df_rows(df, xyz_cols):
    return (df[xyz_cols].values
            .astype(np.float32)
            .reshape(len(df), N_JOINTS, 3))

def load_uiprmd():
    if not os.path.exists(UIPRMD_CSV):
        print(f'UI-PRMD not found: {UIPRMD_CSV}')
        return []
    df       = pd.read_csv(UIPRMD_CSV)
    xyz_cols = [f'{j}_{ax}' for j in JOINT_ORDER for ax in ['x', 'y', 'z']]
    samples  = []
    for (subj, mov, ex_id, lv), grp in df.groupby(
            ['subject', 'movement', 'exercise_id', 'label']):
        grp = grp.sort_values('frame')
        raw = _kps_from_df_rows(grp, xyz_cols) / 1000.0
        try:
            kps = process_keypoints(raw)
            ex  = grp['exercise'].iloc[0]
            samples.append({
                'keypoints':    kps,
                'adj':          ADJ,
                'label':        int(lv),
                'quality':      float(lv),
                'exercise':     ex,
                'exercise_idx': EXERCISE2IDX.get(ex, EXERCISE2IDX['unknown']),
                'subject':      f'uiprmd_{subj}',
                'source':       'uiprmd',
                'scalars':      extract_scalars(kps),
            })
        except Exception as e:
            print(f'  skip uiprmd {subj}/{mov}/{ex_id}: {e}')
    print(f'UI-PRMD: {len(samples)} samples')
    return samples

def load_kimore():
    if not os.path.exists(KIMORE_CSV):
        print(f'KIMORE not found: {KIMORE_CSV}')
        return []
    df = pd.read_csv(KIMORE_CSV)
    xyz_cols = []
    for jn in ['HipLeft', 'KneeLeft', 'AnkleLeft',
               'HipRight', 'KneeRight', 'AnkleRight']:
        xyz_cols += [f'{jn}_x', f'{jn}_y', f'{jn}_z']
    df['quality_norm'] = df[['cTS', 'cPO', 'cCF']].mean(axis=1) / 50.0
    samples = []
    for subj, grp in df.groupby('Subject'):
        grp = grp.sort_values('FrameID')
        raw = _kps_from_df_rows(grp, xyz_cols)
        try:
            kps = process_keypoints(raw)
            samples.append({
                'keypoints':    kps,
                'adj':          ADJ,
                'label':        int(grp['label'].iloc[0]),
                'quality':      float(grp['quality_norm'].mean()),
                'exercise':     'squat',
                'exercise_idx': EXERCISE2IDX['squat'],
                'subject':      f'kimore_{subj}',
                'source':       'kimore',
                'scalars':      extract_scalars(kps),
            })
        except Exception as e:
            print(f'  skip kimore {subj}: {e}')
    print(f'KIMORE: {len(samples)} samples')
    return samples

def load_keraal():
    if not os.path.exists(KERAAL_CSV):
        print(f'Keraal not found: {KERAAL_CSV}')
        return []
    df = pd.read_csv(KERAAL_CSV)
    xyz_cols = []
    for jn in ['HipLeft', 'KneeLeft', 'AnkleLeft',
               'HipRight', 'KneeRight', 'AnkleRight']:
        xyz_cols += [f'{jn}_x', f'{jn}_y', f'{jn}_z']
    samples = []
    for ann_key, grp in df.groupby('ann_key'):
        grp = grp.sort_values('FrameID') if 'FrameID' in grp.columns else grp
        raw = _kps_from_df_rows(grp, xyz_cols)
        try:
            kps  = process_keypoints(raw)
            subj = '-'.join(str(ann_key).split('-')[:3])
            samples.append({
                'keypoints':    kps,
                'adj':          ADJ,
                'label':        int(grp['label'].iloc[0]),
                'quality':      float(grp['label'].iloc[0]),
                'exercise':     'ctk_squat',
                'exercise_idx': EXERCISE2IDX['ctk_squat'],
                'subject':      f'keraal_{subj}',
                'source':       'keraal',
                'scalars':      extract_scalars(kps),
            })
        except Exception as e:
            print(f'  skip keraal {ann_key}: {e}')
    print(f'Keraal: {len(samples)} samples')
    return samples

def build_master_dataset():
    raw   = load_uiprmd() + load_kimore() + load_keraal()
    clean = []
    for s in raw:
        try:
            kps = safe_np(s['keypoints'], clip=5.0)
            sc  = safe_np(s.get('scalars', extract_scalars(kps)), clip=10.0)
            q   = float(np.nan_to_num(s.get('quality', s.get('label', 1)),
                                      nan=float(s.get('label', 1))))
            ex  = int(s.get('exercise_idx', EXERCISE2IDX['unknown']))
            if kps.shape != (TARGET_LEN, N_JOINTS, 3):
                raise ValueError('bad kps shape')
            if sc.shape != (10,):
                raise ValueError('bad scalar shape')
            if not valid_np(kps) or not valid_np(sc):
                raise ValueError('NaN/Inf detected')
            if not (0 <= ex < N_EXERCISES):
                ex = EXERCISE2IDX['unknown']
            s.update({'keypoints': kps, 'scalars': sc,
                      'label':        int(s.get('label', 1)),
                      'quality':      float(np.clip(q, 0, 1)),
                      'exercise_idx': ex})
            clean.append(s)
        except Exception as e:
            print(f"  DROP {s.get('subject', '?')}: {e}")
    print(f'\nMaster dataset: {len(clean)} samples')
    print(f"  correct={sum(s['label']==1 for s in clean)}  "
          f"incorrect={sum(s['label']==0 for s in clean)}")
    print(f"  sources:   {set(s['source']   for s in clean)}")
    print(f"  exercises: {set(s['exercise'] for s in clean)}")
    return clean

def loso_split(samples, held_out):
    return ([s for s in samples if s['subject'] != held_out],
            [s for s in samples if s['subject'] == held_out])

def get_uiprmd_subjects(samples):
    return sorted({s['subject'] for s in samples if s['source'] == 'uiprmd'})


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# F. MODEL â€” STGCNBlock â†’ STGCN â†’ ClinicalTransformer â†’ RehabNet
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# ===== Final active runtime fixes =====

"""
Final runtime patch for the rehab notebook/script.

Run this AFTER loading all model/helper definitions and BEFORE testing live/uploaded
videos. It fixes:
  1. no-movement false reps,
  2. training-compatible jerk scalar,
  3. df_full/path API mismatch,
  4. rule-based jerk threshold/multiplier.
"""

JERK_THRESHOLD = 27.66

ROM_TARGETS = {
    'inline_lunge': 90.0,
    'side_lunge': 90.0,
    'deep_squat': 95.0,
    'sit_to_stand': 80.0,
    'straight_leg_raise': 55.0,
    'knee_bend': 80.0,
    'squat': 90.0,
    'ctk_squat': 90.0,
    'hurdle_step': 65.0,
    'unknown': 90.0,
    'default': 90.0,
}

UNILATERAL_EXERCISES = {
    'inline_lunge',
    'side_lunge',
    'straight_leg_raise',
    'hurdle_step',
}

EX_MIN_DIST.update({
    'inline_lunge': 100,
    'side_lunge': 100,
    'deep_squat': 60,
    'sit_to_stand': 80,
    'straight_leg_raise': 40,
    'knee_bend': 60,
    'squat': 60,
    'ctk_squat': 60,
    'hurdle_step': 45,
    'unknown': 60,
    'default': 60,
})

EX_MIN_ROM = {
    'inline_lunge': 35.0,
    'side_lunge': 35.0,
    'deep_squat': 30.0,
    'sit_to_stand': 35.0,
    'straight_leg_raise': 25.0,
    'knee_bend': 30.0,
    'squat': 30.0,
    'ctk_squat': 30.0,
    'hurdle_step': 20.0,
    'unknown': 30.0,
    'default': 30.0,
}


def _angle_from_kps(kps, hip_idx, knee_idx, ankle_idx):
    kps = safe_np(kps, clip=5.0)
    a = kps[:, hip_idx, :]
    b = kps[:, knee_idx, :]
    c = kps[:, ankle_idx, :]
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + EPS
    cos = np.sum(ba * bc, axis=1) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _training_style_angle_features(angle, fps=FPS):
    angle = safe_np(angle)
    if len(angle) < 5:
        angle = np.pad(angle, (0, max(0, 5 - len(angle))), mode='edge')

    try:
        s = _smooth(angle)
    except Exception:
        s = angle

    vel = np.gradient(s, 1.0 / fps)
    jerk_arr = np.gradient(np.abs(vel), 1.0 / fps)
    return {
        'smooth_angle': s,
        'rom': float(np.nanmax(s) - np.nanmin(s)),
        'peak': float(np.nanmin(s)),
        'max_velocity': float(np.nanmax(np.abs(vel))),
        'mean_velocity': float(np.nanmean(np.abs(vel))),
        'mean_jerk': float(np.nanmean(jerk_arr)),
        'peak_jerk': float(np.nanmax(jerk_arr)),
    }


def extract_scalars(kps, exercise='unknown', fps=FPS, debug=False):
    kps = safe_np(kps, clip=5.0)
    lk = _angle_from_kps(kps, 0, 1, 2)
    rk = _angle_from_kps(kps, 3, 4, 5)
    lf = _training_style_angle_features(lk, fps=fps)
    rf = _training_style_angle_features(rk, fps=fps)

    lr = lf['rom']
    rr = rf['rom']
    sym = abs(lr - rr) / ((lr + rr) / 2.0 + EPS) * 100.0 if (lr + rr) > 1.0 else 0.0
    vel = max(lf['mean_velocity'], rf['mean_velocity'])
    jerk = max(lf['mean_jerk'], rf['mean_jerk'])

    hip = (kps[:, 0, :] + kps[:, 3, :]) / 2.0
    peak_idx = int(np.nanargmin(lf['smooth_angle'])) if len(lf['smooth_angle']) else 0
    peak_idx = int(np.clip(peak_idx, 0, len(hip) - 1))
    trunk = float(abs(hip[peak_idx, 0] - hip[0, 0]))

    lk_c = lf['smooth_angle'] - np.nanmean(lf['smooth_angle'])
    rk_c = rf['smooth_angle'] - np.nanmean(rf['smooth_angle'])
    if np.nanstd(lk_c) > EPS and np.nanstd(rk_c) > EPS:
        lag = float(abs(np.argmax(np.correlate(lk_c, rk_c, 'full')) - (len(lk_c) - 1)))
    else:
        lag = 0.0

    try:
        smooth = _spectral_arc_length(lf['smooth_angle'])
    except Exception:
        diff_std = float(np.nanstd(np.diff(lf['smooth_angle'])))
        smooth = float(np.clip(1.0 / (diff_std + EPS) / 100.0, 0.0, 1.0))

    scalars = np.array([
        np.clip(lr / 180.0, 0, 1),
        np.clip(rr / 180.0, 0, 1),
        np.clip(lf['peak'] / 180.0, 0, 1),
        np.clip(rf['peak'] / 180.0, 0, 1),
        np.clip(sym / 100.0, 0, 5),
        np.clip(vel / 200.0, 0, 5),
        np.clip(jerk / 10.0, 0, 20),
        np.clip(trunk, 0, 5),
        np.clip(lag / TARGET_LEN, 0, 1),
        np.clip(smooth, 0, 1),
    ], dtype=np.float32)

    if debug:
        print(
            f"[Scalar Debug] L_ROM={lr:.1f}deg  R_ROM={rr:.1f}deg  "
            f"L_peak={lf['peak']:.1f}deg  R_peak={rf['peak']:.1f}deg  "
            f"Symmetry={sym:.1f}  Vel={vel:.1f}deg/s  "
            f"Jerk={jerk:.2f}  Lag={lag:.0f}fr  Smooth={smooth:.3f}"
        )

    return safe_np(scalars, fill=0.0, clip=20.0)


def _df_motion_proxy(df):
    cols = [
        'HipLeft_y', 'HipRight_y', 'KneeLeft_y', 'KneeRight_y',
        'AnkleLeft_y', 'AnkleRight_y',
        'HipLeft_x', 'HipRight_x', 'KneeLeft_x', 'KneeRight_x',
    ]
    present = [c for c in cols if c in df.columns]
    if not present:
        return 1.0
    vals = df[present].ffill().bfill().values.astype(np.float32)
    ranges = np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0)
    return float(np.nanmax(ranges))


def _segment_reps(df, angle_col='angle_knee_L_smooth',
                  min_frames=MIN_FRAMES, min_rom=None, exercise='unknown'):
    if angle_col in df.columns:
        angles_l = df[angle_col].ffill().bfill().values
    else:
        angles_l = _knee_angle_from_df(df, 'L')

    try:
        angles_r = _knee_angle_from_df(df, 'R')
    except Exception:
        angles_r = angles_l

    angles_l = safe_np(angles_l)
    angles_r = safe_np(angles_r)
    n = len(angles_l)
    if n < min_frames:
        print(f'  Segmenter: too short n_frames={n}')
        return []

    ex_key = exercise if exercise in EX_MIN_DIST else 'default'
    min_dist = max(EX_MIN_DIST[ex_key], n // 20)
    min_rom = EX_MIN_ROM.get(ex_key, EX_MIN_ROM['default']) if min_rom is None else min_rom

    rom_l = float(np.nanmax(angles_l) - np.nanmin(angles_l))
    rom_r = float(np.nanmax(angles_r) - np.nanmin(angles_r))
    rom = max(rom_l, rom_r)
    motion = _df_motion_proxy(df)

    # Camera/pose jitter can create fake angle ROM. Require actual landmark displacement too.
    if rom < min_rom or motion < 0.035:
        print(f'  Segmenter: exercise={exercise}  n_frames={n}  '
              f'min_dist={min_dist}  ROM_L={rom_l:.1f}  ROM_R={rom_r:.1f}  '
              f'motion={motion:.3f}  NO VALID MOVEMENT')
        return []

    angles = angles_l if rom_l >= rom_r else angles_r
    peaks, _ = find_peaks(
        -angles,
        distance=min_dist,
        prominence=max(12.0, min_rom * 0.6),
        width=max(5, min_dist // 5),
    )

    print(f'  Segmenter: exercise={exercise}  n_frames={n}  '
          f'min_dist={min_dist}  ROM_L={rom_l:.1f}  ROM_R={rom_r:.1f}  '
          f'motion={motion:.3f}  peaks={len(peaks)}')

    if len(peaks) == 0:
        print('  Segmenter: movement exists but no clean rep peaks')
        return []

    mids = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    bounds = [0] + mids + [n]
    reps = []
    for i in range(len(bounds) - 1):
        rep = df.iloc[bounds[i]:bounds[i + 1]].copy()
        if len(rep) < min_frames:
            continue
        try:
            rep_l = _knee_angle_from_df(rep, 'L')
            rep_r = _knee_angle_from_df(rep, 'R')
            rep_rom = max(
                float(np.nanmax(rep_l) - np.nanmin(rep_l)),
                float(np.nanmax(rep_r) - np.nanmin(rep_r)),
            )
            rep_motion = _df_motion_proxy(rep)
        except Exception:
            rep_rom = rom
            rep_motion = motion
        if rep_rom >= min_rom and rep_motion >= 0.035:
            reps.append(rep)
        else:
            print(f'    reject rep {i + 1}: ROM={rep_rom:.1f}, motion={rep_motion:.3f}')
    return reps


def rule_based_label(scalars, exercise='inline_lunge'):
    l_rom = scalars[0] * 180.0
    r_rom = scalars[1] * 180.0
    sym = scalars[4] * 100.0
    vel = scalars[5] * 200.0
    jerk = scalars[6] * 10.0
    target = ROM_TARGETS.get(exercise, ROM_TARGETS['default'])
    active_rom = max(l_rom, r_rom)
    reasons = []
    if active_rom < target * 0.7:
        reasons.append('insufficient_rom')
    if sym > 35.0 and exercise not in UNILATERAL_EXERCISES:
        reasons.append('asymmetric')
    if jerk > JERK_THRESHOLD:
        reasons.append('jerky')
    if vel > 150.0:
        reasons.append('too_fast')
    return (1 if not reasons else 0), reasons


def generate_feedback(*args, exercise='squat'):
    """
    Backward-compatible feedback helper.

    New style:
        generate_feedback(scalars, exercise='squat') -> list[str]

    Old notebook style:
        generate_feedback(label, quality, scalars) -> dict
    """
    old_style = False
    if len(args) == 1:
        scalars = args[0]
    elif len(args) >= 3:
        old_style = True
        scalars = args[2]
    else:
        raise TypeError("generate_feedback expects scalars or label, quality, scalars")

    label, reasons = rule_based_label(scalars, exercise)
    if label == 1:
        messages = ["Good rep - maintain that form."]
        if old_style:
            return {
                'message': 'Good rep - maintain that form.',
                'short': 'Good rep',
                'flags': [],
                'severity': 'good',
            }
        return messages

    messages = []
    if 'insufficient_rom' in reasons:
        messages.append("Bend your knee further - aim for a deeper range of motion.")
    if 'asymmetric' in reasons:
        messages.append("Try to move both sides equally.")
    if 'jerky' in reasons:
        messages.append("Try to move more smoothly - avoid jerky or sudden changes.")
    if 'too_fast' in reasons:
        messages.append("Slow down - controlled movement is safer and more effective.")
    messages = messages or ["Work on a smooth, controlled movement pattern."]

    if old_style:
        return {
            'message': messages[0],
            'short': messages[0].split(' - ')[0][:40],
            'flags': reasons,
            'severity': 'error' if reasons else 'warning',
        }
    return messages


