"""Evaluate anomaly scores on the held-out TEST split and make the README plots.

Usage:  python -m src.evaluate        (reads artifacts/scores.csv + artifacts/threshold.json)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import ART_DIR, COMPONENTS, REPORT_DIR
from src.tracking import log_run


def compute_metrics(scores: pd.DataFrame, threshold: float) -> dict:
    """`scores` needs columns: split, score, is_healthy, faulty_<component>."""
    test = scores[scores["split"] == "test"]
    anomalous = ~test["is_healthy"].to_numpy(dtype=bool)  # positive class = faulty cycle
    s = test["score"].to_numpy()
    flagged = s > threshold

    out = {
        "threshold": float(threshold),
        "n_test": int(len(test)),
        "n_test_healthy": int((~anomalous).sum()),
        "n_test_faulty": int(anomalous.sum()),
    }
    if anomalous.any() and (~anomalous).any():
        out["roc_auc"] = float(roc_auc_score(anomalous, s))
        out["recall"] = float(flagged[anomalous].mean())
        out["false_positive_rate"] = float(flagged[~anomalous].mean())
        tp = int((flagged & anomalous).sum())
        out["precision"] = float(tp / flagged.sum()) if flagged.sum() else float("nan")

    # Which fault types does the autoencoder actually catch?
    per_component = {}
    for c in COMPONENTS:
        mask = test[f"faulty_{c}"].to_numpy(dtype=bool)
        if mask.any():
            per_component[c] = {"n": int(mask.sum()), "recall": float(flagged[mask].mean())}
    out["recall_by_faulty_component"] = per_component
    return out


def plot_score_histogram(scores: pd.DataFrame, threshold: float, path) -> None:
    test = scores[scores["split"] == "test"]
    s = np.maximum(test["score"].to_numpy(), 1e-12)
    healthy = test["is_healthy"].to_numpy(dtype=bool)
    bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 40)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s[healthy], bins=bins, alpha=0.7, label="healthy")
    ax.hist(s[~healthy], bins=bins, alpha=0.7, label="faulty")
    ax.axvline(threshold, color="k", ls="--", label=f"threshold = {threshold:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("anomaly score (mean squared reconstruction error)")
    ax.set_ylabel("test cycles")
    ax.set_title("Autoencoder anomaly score on held-out test cycles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(scores: pd.DataFrame, path) -> None:
    test = scores[scores["split"] == "test"]
    y = ~test["is_healthy"].to_numpy(dtype=bool)
    fpr, tpr, _ = roc_curve(y, test["score"].to_numpy())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, test['score']):.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC (test split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _flat(m: dict) -> dict:
    out = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    for c, d in m["recall_by_faulty_component"].items():
        out[f"recall_{c}"] = d["recall"]
    return out


def run(scores_path=ART_DIR / "scores.csv", threshold_path=ART_DIR / "threshold.json") -> dict:
    scores = pd.read_csv(scores_path, index_col="cycle_id")
    threshold = json.loads(open(threshold_path).read())["threshold"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(scores, threshold)
    (REPORT_DIR / "day1_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_score_histogram(scores, threshold, REPORT_DIR / "score_histogram.png")
    if "roc_auc" in metrics:
        plot_roc(scores, REPORT_DIR / "roc_curve.png")
    log_run("evaluate", params={"threshold": threshold}, metrics=_flat(metrics),
            artifacts=[REPORT_DIR / "day1_metrics.json", REPORT_DIR / "score_histogram.png",
                       REPORT_DIR / "roc_curve.png"])
    return metrics


def main() -> None:
    m = run()
    print("\n=== Held-out TEST results ===")
    print(f"test cycles: {m['n_test']}  (healthy {m['n_test_healthy']}, faulty {m['n_test_faulty']})")
    for k in ("roc_auc", "recall", "false_positive_rate", "precision"):
        if k in m:
            print(f"{k:22s}: {m[k]:.3f}")
    print("\nRecall by faulty component (how often the AE flags a cycle where this part is faulty):")
    for c, d in m["recall_by_faulty_component"].items():
        print(f"  {c:12s} n={d['n']:4d}  recall={d['recall']:.3f}")
    print(f"\nSaved: {REPORT_DIR}/day1_metrics.json, score_histogram.png, roc_curve.png")


if __name__ == "__main__":
    main()