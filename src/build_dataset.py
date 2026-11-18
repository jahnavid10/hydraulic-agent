"""Build data/processed/features.csv and labels.csv (with a fixed train/val/test split).

Usage:  python -m src.build_dataset [--raw-dir data/raw]

Split (fixed seed, stratified by healthy vs faulty so every split has both):
    train 60%  -> the autoencoder is fitted on the *healthy* cycles of this split
    val   20%  -> healthy cycles of this split set the anomaly threshold
    test  20%  -> never used for fitting or thresholding; used for all reported metrics
                  and (later) for the agent evaluation.

Note: cycles are recorded in time order and neighbouring cycles are similar, so a random
split is slightly optimistic. Mention this under "Limitations" in the README.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROC_DIR, RAW_DIR, SEED, TRAIN_FRAC, VAL_FRAC
from src.features import build_feature_table, load_labels


def assign_splits(is_healthy: np.ndarray, seed: int = SEED) -> np.ndarray:
    n = len(is_healthy)
    idx = np.arange(n)
    train_idx, rest_idx = train_test_split(
        idx, train_size=TRAIN_FRAC, stratify=is_healthy, random_state=seed
    )
    val_share = VAL_FRAC / (1 - TRAIN_FRAC)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_share, stratify=is_healthy[rest_idx], random_state=seed
    )
    split = np.empty(n, dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROC_DIR)
    args = ap.parse_args()

    print("Extracting features ...")
    feats = build_feature_table(args.raw_dir)
    labels = load_labels(args.raw_dir)
    if len(feats) != len(labels):
        raise ValueError(f"{len(feats)} feature rows vs {len(labels)} label rows")

    labels["split"] = assign_splits(labels["is_healthy"].to_numpy())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.out_dir / "features.csv")
    labels.to_csv(args.out_dir / "labels.csv")

    print(f"\nFeatures: {feats.shape}   Labels: {labels.shape}")
    print("\nHealthy vs faulty per split (WRITE THESE NUMBERS DOWN for your README):")
    print(pd.crosstab(labels["split"], labels["is_healthy"].map({True: "healthy", False: "faulty"})))
    n_h = int(labels["is_healthy"].sum())
    print(f"\nTotal healthy cycles: {n_h} of {len(labels)}")
    if n_h < 60:
        print("WARNING: very few healthy cycles; thresholds will be noisy.")


if __name__ == "__main__":
    main()