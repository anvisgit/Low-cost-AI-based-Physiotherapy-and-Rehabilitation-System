import argparse
import json
import os

import torch

from rehabnet_classic import data_pipeline as dp
from rehabnet_classic.classic_inference import run_patient_video
from rehabnet_classic.classic_model import RehabNet


OLD_EXERCISE_LIST = [
    "deep_squat", "hurdle_step", "inline_lunge", "side_lunge",
    "sit_to_stand", "straight_leg_raise", "squat", "ctk_squat", "unknown",
]


def adapt_exercise_weights(state, old_exercises=OLD_EXERCISE_LIST, new_exercises=None):
    new_exercises = new_exercises or dp.EXERCISE_LIST
    if not isinstance(state, dict):
        return state
    state = dict(state)
    old_idx = {name: i for i, name in enumerate(old_exercises)}
    new_idx = {name: i for i, name in enumerate(new_exercises)}

    def fallback(tensor):
        if "unknown" in old_idx and old_idx["unknown"] < tensor.shape[0]:
            return tensor[old_idx["unknown"]].clone()
        return tensor.mean(dim=0)

    def adapt_rows(key, rows):
        if key not in state:
            return
        tensor = state[key]
        if not hasattr(tensor, "shape") or len(tensor.shape) == 0 or tensor.shape[0] == rows:
            return
        new_tensor = tensor.new_empty((rows,) + tuple(tensor.shape[1:]))
        fb = fallback(tensor)
        for name, ni in new_idx.items():
            new_tensor[ni] = tensor[old_idx[name]] if name in old_idx and old_idx[name] < tensor.shape[0] else fb
        state[key] = new_tensor

    adapt_rows("ex_emb.weight", len(new_exercises))
    adapt_rows("ex_head.weight", len(new_exercises))
    adapt_rows("ex_head.bias", len(new_exercises))
    return state


def load_model(path, device):
    model = RehabNet().to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = adapt_exercise_weights(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run classic RehabNet on a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exercise", default="squat", choices=dp.EXERCISE_LIST)
    parser.add_argument("--patient-id", default="P001")
    parser.add_argument("--hardware-mode", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    summary = run_patient_video(
        args.video,
        args.patient_id,
        model=model,
        exercise=args.exercise,
        hardware_mode=args.hardware_mode,
        device=device,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()

