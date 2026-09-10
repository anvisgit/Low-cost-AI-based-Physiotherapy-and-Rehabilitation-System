import argparse
import os
from rehabnet_classic import data_pipeline as dp
from rehabnet_classic.classic_train import run_loso
def main():
    parser = argparse.ArgumentParser(description="Train classic RehabNet with LOSO validation.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="models/rehabnet_best.pth")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    if args.data_dir:
        dp.DATA_DIR = args.data_dir
        dp.UIPRMD_CSV = os.path.join(dp.DATA_DIR, "uiprmd.csv")
        dp.KIMORE_CSV = os.path.join(dp.DATA_DIR, "squats_tabular_timeseries_binary.csv")
        dp.KERAAL_CSV = os.path.join(dp.DATA_DIR, "keraal_final_binary.csv")

    dp.MODEL_SAVE = args.output
    samples = dp.build_master_dataset()
    run_loso(samples=samples, n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()

