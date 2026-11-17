"""Train the autoencoder on HEALTHY training cycles, set the anomaly threshold, score everything.

Usage:  python -m src.train_ae [--epochs 300] [--percentile 95] [--stable-only]

Outputs (artifacts/):
    ae.pt             model weights + architecture
    scaler.joblib     StandardScaler fitted on healthy training cycles only
    threshold.json    anomaly threshold + how it was chosen + feature names
    scores.csv        per-cycle anomaly score merged with labels/splits
reports/:
    training_curve.png
"""
import argparse
import json
import random

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ART_DIR, PROC_DIR, REPORT_DIR, SEED
from src.model import Autoencoder, per_feature_errors
from src.tracking import log_run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=25, help="early-stopping patience (epochs)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="threshold = this percentile of healthy VALIDATION errors")
    ap.add_argument("--stable-only", action="store_true",
                    help="train only on healthy cycles whose stable flag is 0")
    args = ap.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feats = pd.read_csv(PROC_DIR / "features.csv", index_col="cycle_id")
    labels = pd.read_csv(PROC_DIR / "labels.csv", index_col="cycle_id")
    assert (feats.index == labels.index).all()

    healthy = labels["is_healthy"].astype(bool)
    train_mask = (labels["split"] == "train") & healthy
    val_mask = (labels["split"] == "val") & healthy
    if args.stable_only:
        train_mask &= labels["stable"] == 0
    print(f"healthy train cycles: {int(train_mask.sum())} | healthy val cycles: {int(val_mask.sum())}")
    if train_mask.sum() < 30 or val_mask.sum() < 10:
        raise SystemExit("Too few healthy cycles to train/validate. Send me the counts printed above.")

    # Fit the scaler on healthy TRAIN cycles only: no information leaks from val/test/faulty.
    scaler = StandardScaler().fit(feats[train_mask].to_numpy())
    X_all = scaler.transform(feats.to_numpy()).astype(np.float32)
    X_train = torch.tensor(X_all[train_mask.to_numpy()], device=device)
    X_val = torch.tensor(X_all[val_mask.to_numpy()], device=device)

    model = Autoencoder(X_all.shape[1], tuple(args.hidden), args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train), batch_size=args.batch_size, shuffle=True)

    history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), X_val).item()
        history.append((train_loss, val_loss))

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if bad_epochs >= args.patience:
            print(f"early stop at epoch {epoch} (best val {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Score every cycle; threshold from healthy VALIDATION cycles only.
    errors = per_feature_errors(model, X_all, device)       # (n_cycles, n_features)
    score = errors.mean(axis=1)
    threshold = float(np.percentile(score[val_mask.to_numpy()], args.percentile))

    ART_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config()}, ART_DIR / "ae.pt")
    joblib.dump(scaler, ART_DIR / "scaler.joblib")
    (ART_DIR / "threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "percentile": args.percentile,
        "n_val_healthy": int(val_mask.sum()),
        "n_train_healthy": int(train_mask.sum()),
        "stable_only": args.stable_only,
        "feature_names": list(feats.columns),
    }, indent=2))
    out = labels.copy()
    out["score"] = score
    out.to_csv(ART_DIR / "scores.csv")

    h = np.array(history)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h[:, 0], label="train (healthy)")
    ax.plot(h[:, 1], label="validation (healthy)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "training_curve.png", dpi=150)
    plt.close(fig)

    log_run("train_ae", params=vars(args),
        metrics={"threshold": threshold, "best_val_loss": best_val,
                "n_train_healthy": int(train_mask.sum())},
        artifacts=[REPORT_DIR / "training_curve.png"], history=history)
    
    print(f"\nthreshold = {threshold:.4f}  ({args.percentile}th percentile of healthy val errors)")
    print(f"saved model, scaler, threshold, scores to {ART_DIR}")
    print("next: python -m src.evaluate")


if __name__ == "__main__":
    main()