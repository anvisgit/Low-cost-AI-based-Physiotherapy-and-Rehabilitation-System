from .data_pipeline import *
from .classic_model import RehabNet

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# I. INFERENCE + FEEDBACK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_torque(quality, mode):
    """Single definition â€” FIX: was defined twice with different logic."""
    m = HARDWARE_MODES.get(mode, HARDWARE_MODES[0])
    return float(np.clip(m['scale'] * quality + m['bias'], m['min'], m['max']))

def generate_feedback(label, quality, scalars):
    """Priority-ordered rule engine â†’ feedback dict."""
    s = np.asarray(scalars)
    triggered = sorted(
        [r for r in FEEDBACK_RULES if r['check'](s)],
        key=lambda r: r['priority'], reverse=True,
    )
    flags = [r['name'] for r in triggered]

    if label == 1 and quality >= 0.75:
        return {'message': 'Great rep â€” keep it up!',
                'short': 'Great rep', 'flags': [], 'severity': 'good'}
    if label == 1 and quality >= 0.5:
        return {'message': 'Good rep â€” maintain that form.',
                'short': 'Good rep', 'flags': flags, 'severity': 'good'}
    if triggered:
        top = triggered[0]
        return {'message':  top['message'],
                'short':    top['short'],
                'flags':    flags,
                'severity': 'error' if top['priority'] >= 8 else 'warning'}
    return {'message': 'Focus on smooth, controlled movement.',
            'short': 'Keep it smooth', 'flags': [], 'severity': 'warning'}

def build_hardware_payload(sample, quality, scalars, torque, mode):
    kps = sample['keypoints']

    def _ang(a, b, c):
        ba, bc = a - b, c - b
        d = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + EPS
        return np.degrees(np.arccos(np.clip(np.sum(ba * bc, axis=1) / d, -1, 1)))

    lk     = _ang(kps[:, 0, :], kps[:, 1, :], kps[:, 2, :])
    rk     = _ang(kps[:, 3, :], kps[:, 4, :], kps[:, 5, :])
    l_peak = float(np.min(lk));  r_peak = float(np.min(rk))
    l_rom  = float(np.max(lk) - l_peak)
    r_rom  = float(np.max(rk) - r_peak)
    ex     = sample.get('exercise', 'unknown')
    target = TARGET_ANGLES.get(ex, TARGET_ANGLES['default'])
    l_def  = max(0., target - l_peak)
    r_def  = max(0., target - r_peak)

    return {
        'torque':   round(torque, 4),
        'mode':     HARDWARE_MODES[mode]['name'],
        'mode_int': mode,
        'joint': {
            'left_knee_peak_deg':  round(l_peak, 2),
            'right_knee_peak_deg': round(r_peak, 2),
            'left_knee_rom_deg':   round(l_rom, 2),
            'right_knee_rom_deg':  round(r_rom, 2),
        },
        'target': {
            'knee_target_deg':   target,
            'left_deficit_deg':  round(l_def, 2),
            'right_deficit_deg': round(r_def, 2),
        },
        'torque_per_degree': round(torque / max(l_def, 1.0), 5),
    }

@torch.no_grad()
def infer_sample(model, sample, device=DEVICE, hardware_mode=0):
    """
    Run RehabNet on one rep.  Returns full result dict.
    FIX: model returns 3 values; always unpack with *_ or explicitly.
    """
    model.eval()
    kps_t  = (torch.from_numpy(sample['keypoints'])
              .permute(2, 0, 1).unsqueeze(0).to(device))
    adj_t  = torch.from_numpy(ADJ).to(device)
    sc_t   = torch.from_numpy(sample['scalars']).unsqueeze(0).to(device)
    mode_t = torch.tensor([hardware_mode], dtype=torch.long, device=device)
    ex_t   = torch.tensor([sample['exercise_idx']], dtype=torch.long,
                           device=device)

    logits, quality, *_ = model(kps_t, adj_t, sc_t, mode_t, ex_t)  # 3 outputs
    label   = int(logits.argmax(1).item())
    quality = float(torch.sigmoid(quality).item())
    torque  = compute_torque(quality, hardware_mode)
    fb      = generate_feedback(label, quality, sample['scalars'])
    hw      = build_hardware_payload(sample, quality, sample['scalars'],
                                     torque, hardware_mode)

    return {
        'rep_id':         sample.get('rep_id', '?'),
        'exercise':       sample.get('exercise', 'unknown'),
        'label':          label,
        'correct':        label == 1,
        'quality_score':  round(quality, 4),
        'feedback':       fb['message'],
        'feedback_short': fb['short'],
        'severity':       fb['severity'],
        'flags':          fb['flags'],
        'hardware':       hw,
        'torque_signal':  torque,
        'mode_name':      HARDWARE_MODES[hardware_mode]['name'],
        'scalars':        sample['scalars'].tolist(),
    }



def _predict_exercise_from_samples(model, samples, device=DEVICE):
    """
    Predict exercise from all rough rep segments and return majority vote.
    This is used only when exercise=None.
    """
    if not samples:
        return 'unknown', EXERCISE2IDX['unknown']

    votes = []
    adj_t = torch.from_numpy(ADJ).to(device)
    model.eval()

    with torch.no_grad():
        for s in samples:
            kps_t = (torch.from_numpy(s['keypoints'])
                     .permute(2, 0, 1).unsqueeze(0).to(device))
            pred_ex, pred_idx = model.predict_exercise(kps_t, adj_t)
            votes.append((pred_ex, pred_idx))

    if not votes:
        return 'unknown', EXERCISE2IDX['unknown']

    counts = Counter(v[0] for v in votes)
    pred_ex = counts.most_common(1)[0][0]
    pred_idx = EXERCISE2IDX.get(pred_ex, EXERCISE2IDX['unknown'])
    print(f'[Exercise Classification] Votes: {dict(counts)}')
    return pred_ex, pred_idx


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# J. PATIENT SESSION â€” VIDEO â†’ PER-REP FEEDBACK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_patient_video(video_path, patient_id,
                      model=None, exercise=None,
                      hardware_mode=0, device=DEVICE, debug=True):
    """
    Full pipeline: video file -> per-rep feedback dict + session summary.

    Important:
      - If exercise is provided, segmentation uses that exercise directly.
      - If exercise=None and model is provided, the pipeline does:
          rough segmentation -> exercise majority vote -> re-segmentation
        so the final rep count uses exercise-specific EX_MIN_DIST.
    """
    print(f"\n{'='*50}\nPatient: {patient_id}\n{'='*50}")

    active_exercise = exercise or 'unknown'
    print('\n[PS1] Processing video...')
    df = ps1_process_video(video_path, patient_id, active_exercise)

    print('\n[Bridge] Segmenting reps...')

    if exercise is None and model is not None:
        rough_samples = ps1_df_to_samples(df, patient_id, 'unknown')
        if not rough_samples:
            print('No valid rough reps found - check video quality')
            return {}

        pred_ex, pred_idx = _predict_exercise_from_samples(
            model, rough_samples, device
        )
        active_exercise = pred_ex
        print(f'\n[Exercise Classification] Predicted majority: {active_exercise}')

        samples = ps1_df_to_samples(df, patient_id, active_exercise)
        for s in samples:
            s['exercise'] = active_exercise
            s['exercise_idx'] = pred_idx
    else:
        print(f'\n[Exercise] Using: {active_exercise}')
        samples = ps1_df_to_samples(df, patient_id, active_exercise)

    if not samples:
        print('No valid reps found - check video quality')
        return {}

    if model is None:
        print('\n[PS2] No model provided - returning PS1 features only')
        return {
            'patient': patient_id,
            'exercise': active_exercise,
            'n_reps': len(samples),
            'ps1_only': True,
            'samples': samples,
        }

    print('\n[PS2] Running RehabNet inference...')
    results = []

    for s in samples:
        out = infer_sample(model, s, device, hardware_mode)
        status = 'CORRECT' if out['label'] == 1 else 'INCORRECT'
        print(f"  {out['rep_id']:20s} | {status} "
              f"| quality={out['quality_score']:.2f} "
              f"| torque={out['torque_signal']:.2f} "
              f"| {out['feedback_short']}")
        results.append(out)

    n_correct = sum(r['label'] == 1 for r in results)
    avg_q = float(np.mean([r['quality_score'] for r in results]))
    avg_t = float(np.mean([r['torque_signal'] for r in results]))

    summary = {
        'patient': patient_id,
        'exercise': active_exercise,
        'n_reps': len(results),
        'n_correct': n_correct,
        'pct_correct': round(n_correct / max(len(results), 1) * 100, 1),
        'avg_quality': round(avg_q, 3),
        'avg_torque': round(avg_t, 3),
        'hw_mode': HARDWARE_MODES[hardware_mode]['name'],
        'reps': results,
    }

    print(f'\nSession complete: {len(results)} reps  '
          f'correct={n_correct}/{len(results)}  '
          f'avg_quality={avg_q:.3f}')

    return summary

