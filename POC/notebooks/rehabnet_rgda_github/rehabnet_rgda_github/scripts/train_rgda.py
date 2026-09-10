import argparse
import copy
import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from rehabnet import data_pipeline as dp
from rehabnet.rgda_model import RGDARRehabNet, rgda_loss, rule_flags_from_scalars


DOMAIN_LIST = ["uiprmd", "kimore", "keraal", "mediapipe_video"]
DOMAIN2IDX = {name: idx for idx, name in enumerate(DOMAIN_LIST)}


def prepare_samples(samples):
    for sample in samples:
        src = sample.get("source", "mediapipe_video")
        if src not in DOMAIN2IDX:
            src = "mediapipe_video"

        sample["domain_idx"] = DOMAIN2IDX[src]
        sample["label3"] = int(sample["label"])

        scalars = torch.tensor(sample["scalars"], dtype=torch.float32)
        flags = rule_flags_from_scalars(scalars, exercise=sample["exercise"])
        sample["fault_flags"] = flags.cpu().numpy().astype("float32")
    return samples


class RGDADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        keypoints = torch.from_numpy(np.asarray(sample["keypoints"], dtype=np.float32))
        keypoints = keypoints.permute(2, 0, 1)

        return {
            "keypoints": keypoints,
            "scalars": torch.from_numpy(np.asarray(sample["scalars"], dtype=np.float32)),
            "rule_flags": torch.from_numpy(np.asarray(sample["fault_flags"], dtype=np.float32)),
            "label": torch.tensor(int(sample["label3"]), dtype=torch.long),
            "quality": torch.tensor(float(sample.get("quality", sample["label"])), dtype=torch.float32),
            "fault_targets": torch.from_numpy(np.asarray(sample["fault_flags"], dtype=np.float32)),
            "exercise_idx": torch.tensor(int(sample["exercise_idx"]), dtype=torch.long),
            "mode": torch.tensor(0, dtype=torch.long),
            "hardware_target": torch.tensor(0, dtype=torch.long),
            "torque_target": torch.tensor(0.3, dtype=torch.float32),
            "domain_idx": torch.tensor(int(sample["domain_idx"]), dtype=torch.long),
        }


def split_samples(samples, val_ratio=0.2, seed=42):
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    return samples[n_val:], samples[:n_val]


def grl_schedule(epoch, total_epochs):
    p = epoch / max(total_epochs - 1, 1)
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def evaluate(model, loader, adj, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                keypoints=batch["keypoints"].to(device),
                adj=adj,
                scalars=batch["scalars"].to(device),
                rule_flags=batch["rule_flags"].to(device),
                mode=batch["mode"].to(device),
                exercise_idx=batch["exercise_idx"].to(device),
                grl_lambda=0.0,
            )
            pred = outputs["correctness_logits"].argmax(1)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())

    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.data_dir:
        dp.DATA_DIR = args.data_dir
        dp.UIPRMD_CSV = os.path.join(dp.DATA_DIR, "uiprmd.csv")
        dp.KIMORE_CSV = os.path.join(dp.DATA_DIR, "squats_tabular_timeseries_binary.csv")
        dp.KERAAL_CSV = os.path.join(dp.DATA_DIR, "keraal_final_binary.csv")

    samples = prepare_samples(dp.build_master_dataset())
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    print("Train:", len(train_samples), Counter(s["label3"] for s in train_samples))
    print("Val:", len(val_samples), Counter(s["label3"] for s in val_samples))

    train_loader = DataLoader(RGDADataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RGDADataset(val_samples), batch_size=args.batch_size * 2, shuffle=False)

    model = RGDARRehabNet(
        n_exercises=dp.N_EXERCISES,
        n_modes=dp.N_MODES,
        n_domains=len(DOMAIN_LIST),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    adj = torch.from_numpy(dp.ADJ).to(device)

    best_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        grl_lambda = grl_schedule(epoch, args.epochs)

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                keypoints=batch["keypoints"].to(device),
                adj=adj,
                scalars=batch["scalars"].to(device),
                rule_flags=batch["rule_flags"].to(device),
                mode=batch["mode"].to(device),
                exercise_idx=batch["exercise_idx"].to(device),
                grl_lambda=grl_lambda,
            )
            loss, _ = rgda_loss(
                outputs=outputs,
                labels=batch["label"].to(device),
                quality_targets=batch["quality"].to(device),
                fault_targets=batch["fault_targets"].to(device),
                exercise_targets=batch["exercise_idx"].to(device),
                hardware_targets=batch["hardware_target"].to(device),
                torque_targets=batch["torque_target"].to(device),
                domain_targets=batch["domain_idx"].to(device),
                weights={
                    "correct": 1.0,
                    "quality": 0.3,
                    "fault": 0.6,
                    "exercise": 0.2,
                    "hardware": 0.1,
                    "torque": 0.1,
                    "domain": 0.2,
                },
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        metrics = evaluate(model, val_loader, adj, device)
        print(
            f"epoch {epoch + 1:02d} | loss={total_loss / max(n_batches, 1):.4f} "
            f"| grl={grl_lambda:.2f} | val_acc={metrics['acc']:.3f} | val_f1={metrics['f1']:.3f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "exercise_list": dp.EXERCISE_LIST,
            "domain_list": DOMAIN_LIST,
            "best_val_f1": best_f1,
        },
        args.output,
    )
    print(f"Saved: {args.output}")
    print(f"Best val_f1: {best_f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RGDA-RehabNet.")
    parser.add_argument("--data-dir", default=None, help="Folder containing uiprmd.csv, KIMORE CSV, and Keraal CSV.")
    parser.add_argument("--output", default="models/rgda_rehabnet_best.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

