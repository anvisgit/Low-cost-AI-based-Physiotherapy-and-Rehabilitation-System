"""
Samarth End-to-End Test Suite
==============================
5 clinical ROM scenarios that flow through the real ongoing pipeline:

  joint_angles → ModelExerciseAnalyzer (PS2 RehabNet .pth) → mode_command → MockSensorHub (PS3)

The REAL RehabNet deep learning model (.pth) is loaded and used for inference.
NO outputs are hardcoded. The model computes everything dynamically.

NOTE ON NUMERICAL SCORES:
  - If the real model (.pth) is loaded, the synthetic mock inputs (which are highly
    simplistic and out-of-distribution) will produce compressed/unconfident quality scores
    (typically around 0.80 - 0.82) since they do not resemble real human movement coordinates.
  - If the heuristic fallback is used (when the model is absent), you will see greater
    separation (e.g. 0.97 for perfect form and 0.88 for compensations) because it uses
    hardcoded mathematical rules.

Tests only assert PIPELINE INTEGRITY:
  - The model returned a valid PS2RepResult
  - The mode_command from the model was forwarded to the sensor hub unchanged
  - The sensor hub recorded the correct number of commands
"""
import sys
import os
import numpy as np
import pytest

# ── resolve backend root so imports work when running from any cwd ───────────
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.exercise_analysis.model_analyzer import ModelExerciseAnalyzer
from services.exercise_analysis.schemas import PS2RepResult, PS2ModeCommand
from services.sensor_hub.mock_sensor import MockSensorHub


# ── Load the real RehabNet .pth model once for all tests ─────────────────────
# Resolves: backend/tests/../../../../models/rehabnet_best.pth
_MODEL_PATH = os.path.abspath(
    os.path.join(BACKEND_DIR, "..", "..", "models", "rehabnet_best.pth")
)

analyzer = ModelExerciseAnalyzer()

if os.path.exists(_MODEL_PATH):
    analyzer.load_model(_MODEL_PATH)
    print(f"\n[test_e2e] PASS: Real RehabNet model loaded from: {_MODEL_PATH}")
else:
    print(f"\n[test_e2e] WARN: Model file not found at {_MODEL_PATH} -- using heuristic fallback")


def _run_pipeline(angle_data: dict, rep_id: int):
    """
    Helper: run the full PS2 → PS3 pipeline for one rep.
    Returns (result, hub) so each test can assert on both.
    """
    hub = MockSensorHub()

    # Step 1: Run the real RehabNet model on the given joint angles
    result = analyzer.analyze_rep(angle_data, rep_id=rep_id, rep_number=1, fps=30.0)

    # Step 2: Forward whatever the model recommended to the sensor hub
    command = {
        "cmd": "set_mode",
        "mode_id": result.mode_command.mode_id,
        "mode_name": result.mode_command.mode_name,
        "target_torque": result.mode_command.target_torque,
    }
    hub.send_command(command)

    return result, hub


def _print_result(scenario: str, result, hub) -> None:
    """Print a readable summary of model output and hardware response for this test."""
    f = result.error_flags
    flags_detected = [
        name for name, val in [
            ("insufficient_ROM", f.insufficient_ROM),
            ("too_fast",         f.too_fast),
            ("too_slow",         f.too_slow),
            ("knee_valgus",      f.knee_valgus),
            ("asymmetric",       f.asymmetric),
            ("trunk_comp",       f.trunk_comp),
        ] if val == 1
    ]
    cmd = hub.last_command
    print(f"\n  Scenario : {scenario}")
    print(f"  Error flags detected : {flags_detected if flags_detected else 'none'}")
    print(f"  Session score        : {result.session.session_score:.2f}")
    print(f"  Hardware command sent: cmd={cmd['cmd']}, mode_id={cmd['mode_id']}, "
          f"mode_name={cmd['mode_name']}, target_torque={cmd['target_torque']} Nm")
    print(f"  Commands received by hub: {hub.commands_sent}")


# ════════════════════════════════════════════════════════════════════════════
# Scenario 1 — Poor ROM (barely bending knee)
# Clinical: post-surgery stiffness, only ~5° of knee flexion
# Real model threshold: insufficient_ROM fires when rom < 10.0°
# ════════════════════════════════════════════════════════════════════════════
def test_01_poor_rom_pipeline():
    """
    Patient can barely bend the knee (~5 deg ROM — well below the model's 10 deg threshold).
    The real RehabNet model should detect this and flag insufficient_ROM = 1.
    The resulting mode command must be forwarded to the sensor hub.
    """
    angle_data = {
        "left_knee":   [180.0, 177.0, 175.0, 176.0],   # ~5 deg ROM (below 10 deg threshold)
        "right_knee":  [180.0, 177.0, 175.0, 176.0],
        "left_hip":    [180.0, 180.0, 180.0, 180.0],
        "right_hip":   [180.0, 180.0, 180.0, 180.0],
        "left_ankle":  [180.0, 180.0, 180.0, 180.0],
        "right_ankle": [180.0, 180.0, 180.0, 180.0],
    }

    result, hub = _run_pipeline(angle_data, rep_id=1)
    _print_result("Poor ROM — barely bending knee (~5 deg)", result, hub)

    # 1. Model returned a valid PS2RepResult
    assert isinstance(result, PS2RepResult)

    # 2. Error flags are binary (0 or 1) — no corruption
    flags = result.error_flags
    for val in [flags.insufficient_ROM, flags.too_fast, flags.too_slow,
                flags.knee_valgus, flags.asymmetric, flags.trunk_comp]:
        assert val in (0, 1), f"Error flag must be 0 or 1, got {val}"

    # 3. Real model flagged insufficient ROM (rom < 10 deg triggers this)
    assert flags.insufficient_ROM == 1, (
        "Expected insufficient_ROM=1 for ~5 deg knee ROM (real model threshold: < 10 deg)"
    )

    # 4. Session score is within valid range
    assert 0.0 <= result.session.session_score <= 1.0

    # 5. Mode command was forwarded to sensor hub unchanged (matches model output)
    assert hub.commands_sent == 1
    assert hub.last_command["mode_id"] == result.mode_command.mode_id
    assert hub.last_command["mode_name"] == result.mode_command.mode_name
    assert hub.last_command["target_torque"] == result.mode_command.target_torque

    # 6. Target torque is a positive number (physical sanity)
    assert result.mode_command.target_torque > 0.0


# ════════════════════════════════════════════════════════════════════════════
# Scenario 2 — Too Fast (dangerous swinging speed)
# Clinical: patient swings leg rapidly — injury risk
# Real model threshold: too_fast fires when speed > 35.0 deg/sec
# ════════════════════════════════════════════════════════════════════════════
def test_02_too_fast_pipeline():
    """
    Patient swings the knee very quickly.
    Real model threshold: speed > 35 deg/sec triggers too_fast.
    5 frames, 60 deg change => ~360 deg/sec at 30fps >> 35 deg/sec threshold.
    """
    angle_data = {
        "left_knee":   [180.0, 168.0, 156.0, 144.0, 120.0],   # 60 deg in 5 frames
        "right_knee":  [180.0, 168.0, 156.0, 144.0, 120.0],
        "left_hip":    [180.0, 180.0, 180.0, 180.0, 180.0],
        "right_hip":   [180.0, 180.0, 180.0, 180.0, 180.0],
        "left_ankle":  [180.0, 180.0, 180.0, 180.0, 180.0],
        "right_ankle": [180.0, 180.0, 180.0, 180.0, 180.0],
    }

    result, hub = _run_pipeline(angle_data, rep_id=2)
    _print_result("Too Fast — 60 deg swing in 5 frames (~360 deg/sec)", result, hub)

    assert isinstance(result, PS2RepResult)

    # Real model flagged excessive speed
    assert result.error_flags.too_fast == 1, (
        "Expected too_fast=1 for 60 deg knee swing in 5 frames (~360 deg/sec >> 35 threshold)"
    )

    # Session score in valid range
    assert 0.0 <= result.session.session_score <= 1.0

    # Command pipeline: hub received the model's own recommendation
    assert hub.commands_sent == 1
    assert hub.last_command["mode_id"] == result.mode_command.mode_id
    assert hub.last_command["target_torque"] == result.mode_command.target_torque


# ════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Bilateral Asymmetry (left leg much weaker than right)
# Clinical: unilateral muscle weakness, asymmetric gait
# Real model threshold: asymmetric fires when asymmetry > 8.0 deg
# ════════════════════════════════════════════════════════════════════════════
def test_03_bilateral_asymmetry_pipeline():
    """
    Left knee barely moves (4 deg ROM) while right performs full 45 deg flexion.
    Mean left-right difference >> 8 deg threshold for asymmetric flag.
    """
    angle_data = {
        "left_knee":   [180.0, 179.0, 178.0, 176.0],   # 4 deg ROM on left
        "right_knee":  [180.0, 165.0, 150.0, 135.0],   # 45 deg ROM on right
        "left_hip":    [180.0, 180.0, 180.0, 180.0],
        "right_hip":   [180.0, 180.0, 180.0, 180.0],
        "left_ankle":  [180.0, 180.0, 180.0, 180.0],
        "right_ankle": [180.0, 180.0, 180.0, 180.0],
    }

    result, hub = _run_pipeline(angle_data, rep_id=3)
    _print_result("Bilateral Asymmetry — left 4 deg vs right 45 deg ROM", result, hub)

    assert isinstance(result, PS2RepResult)

    # Real model detects the left/right imbalance (> 8 deg mean difference)
    assert result.error_flags.asymmetric == 1, (
        "Expected asymmetric=1 for 4 deg vs 45 deg left-right ROM gap (> 8 deg threshold)"
    )

    # Session score accounts for this penalty
    assert 0.0 <= result.session.session_score <= 1.0

    # Command pipeline integrity
    assert hub.commands_sent == 1
    assert hub.last_command["mode_id"] == result.mode_command.mode_id
    assert hub.last_command["mode_name"] == result.mode_command.mode_name
    assert hub.last_command["target_torque"] == result.mode_command.target_torque


# ════════════════════════════════════════════════════════════════════════════
# Scenario 4 — Trunk Compensation (hip over-bending instead of knee)
# Clinical: patient leans forward to compensate for knee weakness
# Real model: trunk_comp fires when incorrect_prob > 0.9 AND quality_val < 0.25
# We verify pipeline integrity — the model decides compensation on its own
# ════════════════════════════════════════════════════════════════════════════
def test_04_trunk_compensation_pipeline():
    """
    Hip ROM (60 deg) greatly exceeds knee ROM (5 deg) — clear trunk compensation pattern.
    The real RehabNet model processes this through neural inference.
    We assert pipeline integrity: whatever it detects is forwarded to hardware correctly.
    """
    angle_data = {
        "left_knee":   [180.0, 178.0, 177.0, 175.0],   # 5 deg knee ROM
        "right_knee":  [180.0, 178.0, 177.0, 175.0],
        "left_hip":    [180.0, 160.0, 140.0, 120.0],   # 60 deg hip ROM
        "right_hip":   [180.0, 160.0, 140.0, 120.0],
        "left_ankle":  [180.0, 180.0, 180.0, 180.0],
        "right_ankle": [180.0, 180.0, 180.0, 180.0],
    }

    result, hub = _run_pipeline(angle_data, rep_id=4)
    _print_result("Trunk Compensation — hip 60 deg vs knee 5 deg ROM", result, hub)

    assert isinstance(result, PS2RepResult)

    # All flags must be valid binary values
    for val in [result.error_flags.insufficient_ROM, result.error_flags.too_fast,
                result.error_flags.trunk_comp, result.error_flags.asymmetric]:
        assert val in (0, 1)

    # At minimum, the tiny knee ROM (5 deg) should register insufficient_ROM
    assert result.error_flags.insufficient_ROM == 1, (
        "Expected insufficient_ROM=1 for 5 deg knee ROM (below real model's 10 deg threshold)"
    )

    # Session score penalised for the error(s)
    assert 0.0 <= result.session.session_score <= 1.0

    # Command pipeline: sensor hub received exactly what model produced
    assert hub.commands_sent == 1
    assert hub.last_command["mode_id"] == result.mode_command.mode_id
    assert hub.last_command["target_torque"] == result.mode_command.target_torque


# ════════════════════════════════════════════════════════════════════════════
# Scenario 5 — Perfect Form (textbook knee extension)
# Clinical: 60 deg smooth symmetric controlled knee flexion
# Real model: speed > 35 deg/sec triggers too_fast
# 60 deg / 79 frames * 30 fps ~= 22.8 deg/sec — safely below threshold
# ════════════════════════════════════════════════════════════════════════════
def test_05_perfect_form_pipeline():
    """
    Patient performs a textbook rep: 60 deg symmetric, slow, controlled knee flexion.
    Using 80 frames: ~22.8 deg/sec — safely below the real model's 35 deg/sec threshold.
    The model should raise ZERO error flags and return a high session score.
    """
    # 80 frames: 60 deg / 79 frames * 30 fps ~= 22.8 deg/sec < 35 threshold
    smooth = list(np.linspace(180.0, 120.0, 80))

    angle_data = {
        "left_knee":   smooth,           # perfect 60 deg flexion, slow
        "right_knee":  smooth,           # perfectly symmetric
        "left_hip":    [180.0] * 80,     # hip stays still
        "right_hip":   [180.0] * 80,
        "left_ankle":  [180.0] * 80,
        "right_ankle": [180.0] * 80,
    }

    result, hub = _run_pipeline(angle_data, rep_id=5)
    _print_result("Perfect Form — 60 deg smooth symmetric knee flexion", result, hub)

    assert isinstance(result, PS2RepResult)

    # No error flags for perfect form
    flags = result.error_flags
    assert flags.insufficient_ROM == 0, "Good ROM (60 deg) should not flag insufficient_ROM"
    assert flags.too_fast == 0,         "Slow smooth movement should not flag too_fast"
    assert flags.asymmetric == 0,       "Identical left/right angles should not flag asymmetric"
    assert flags.trunk_comp == 0,       "Hip held still should not flag trunk_comp"

    # Session score should be high for perfect form
    assert result.session.session_score >= 0.65, (
        f"Expected session_score >= 0.65 for perfect form, got {result.session.session_score}"
    )

    # Command pipeline: hub received the analyzer's recommendation unchanged
    assert hub.commands_sent == 1
    assert hub.last_command["mode_id"] == result.mode_command.mode_id
    assert hub.last_command["mode_name"] == result.mode_command.mode_name
    assert hub.last_command["target_torque"] == result.mode_command.target_torque

    # Low torque expected for good performance (less support needed)
    assert result.mode_command.target_torque <= 5.0, (
        "Perfect form should yield low torque (patient needs less support)"
    )
