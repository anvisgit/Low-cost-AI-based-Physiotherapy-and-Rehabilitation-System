import argparse
import json
import os

import torch

from rehabnet import data_pipeline as dp
from rehabnet.rgda_model import RGDARRehabNet, hybrid_decision_from_outputs, rule_flags_from_scalars


LABEL_NAMES = ["incorrect", "correct", "review"]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = RGDARRehabNet(
        n_exercises=dp.N_EXERCISES,
        n_modes=dp.N_MODES,
        n_domains=4,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_video(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    adj = torch.from_numpy(dp.ADJ).to(device)

    df = dp.ps1_process_video(args.video, args.patient_id, args.exercise)
    samples = dp.ps1_df_to_samples(df, args.patient_id, args.exercise)
    print(f"Detected reps: {len(samples)}")

    results = []
    with torch.no_grad():
        for sample in samples:
            scalars = torch.tensor(sample["scalars"], dtype=torch.float32)
            rule_flags = rule_flags_from_scalars(scalars, exercise=args.exercise)
            keypoints = torch.from_numpy(sample["keypoints"]).permute(2, 0, 1).unsqueeze(0).to(device)

            outputs = model(
                keypoints=keypoints,
                adj=adj,
                scalars=scalars.unsqueeze(0).to(device),
                rule_flags=rule_flags.unsqueeze(0).to(device),
                mode=torch.tensor([args.hardware_mode], dtype=torch.long, device=device),
                exercise_idx=torch.tensor([dp.EXERCISE2IDX[args.exercise]], dtype=torch.long, device=device),
                grl_lambda=0.0,
            )

            probs = torch.softmax(outputs["correctness_logits"], dim=1)[0]
            model_pred = int(torch.argmax(probs).item())
            final = hybrid_decision_from_outputs(outputs, rule_flags.unsqueeze(0).to(device))
            final_label = int(final["final_label"].item())

            row = {
                "rep_id": sample.get("rep_id", "?"),
                "model_pred": LABEL_NAMES[model_pred],
                "final": LABEL_NAMES[final_label],
                "p_incorrect": round(float(probs[0]), 4),
                "p_correct": round(float(probs[1]), 4),
                "p_review": round(float(probs[2]), 4),
                "quality": round(float(torch.sigmoid(outputs["quality"]).item()), 4),
                "torque": round(float(outputs["torque"].item()), 4),
                "rule_flags": [float(x) for x in rule_flags.cpu().numpy().tolist()],
            }
            results.append(row)
            print(
                f"{row['rep_id']:15s} | model={row['model_pred']:9s} "
                f"final={row['final']:9s} p_correct={row['p_correct']:.3f} "
                f"p_incorrect={row['p_incorrect']:.3f} q={row['quality']:.3f} "
                f"torque={row['torque']:.3f} flags={row['rule_flags']}"
            )

    summary = {
        "patient_id": args.patient_id,
        "exercise": args.exercise,
        "video": args.video,
        "n_reps": len(results),
        "n_correct": sum(1 for row in results if row["final"] == "correct"),
        "n_review": sum(1 for row in results if row["final"] == "review"),
        "n_incorrect": sum(1 for row in results if row["final"] == "incorrect"),
        "reps": results,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results: {args.output}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run RGDA-RehabNet on an uploaded/local video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exercise", default="squat", choices=dp.EXERCISE_LIST)
    parser.add_argument("--patient-id", default="P001")
    parser.add_argument("--hardware-mode", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    infer_video(parse_args())

