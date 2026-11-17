"""Train the component-state classifier (which part is degraded, and how badly).

Usage:  python -m src.train_clf

One RandomForest with four outputs (cooler, valve, pump_leak, accumulator). Fitted on the
TRAIN split only; the TEST split is used for the reported numbers.
Outputs: artifacts/clf.joblib, reports/classifier_metrics.json
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.config import ART_DIR, COMPONENTS, PROC_DIR, REPORT_DIR, SEED
from src.tracking import log_run


def fit_and_eval(feats: pd.DataFrame, labels: pd.DataFrame, n_estimators: int = 300):
    train = (labels["split"] == "train").to_numpy()
    test = (labels["split"] == "test").to_numpy()
    X, Y = feats.to_numpy(), labels[COMPONENTS].to_numpy()

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)
    clf.fit(X[train], Y[train])
    pred = clf.predict(X[test])

    metrics = {"n_train": int(train.sum()), "n_test": int(test.sum())}
    for j, c in enumerate(COMPONENTS):
        metrics[f"accuracy_{c}"] = float(accuracy_score(Y[test][:, j], pred[:, j]))
        metrics[f"macro_f1_{c}"] = float(f1_score(Y[test][:, j], pred[:, j], average="macro"))
    return clf, metrics


def main() -> None:
    feats = pd.read_csv(PROC_DIR / "features.csv", index_col="cycle_id")
    labels = pd.read_csv(PROC_DIR / "labels.csv", index_col="cycle_id")
    clf, metrics = fit_and_eval(feats, labels)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "components": COMPONENTS, "feature_names": list(feats.columns)},
                ART_DIR / "clf.joblib")
    (REPORT_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Component classifier, held-out TEST ===")
    for c in COMPONENTS:
        print(f"  {c:12s} accuracy={metrics[f'accuracy_{c}']:.3f}  macro-F1={metrics[f'macro_f1_{c}']:.3f}")
    log_run("train_clf", params={"model": "RandomForest", "n_estimators": clf.n_estimators},
            metrics=metrics, artifacts=[REPORT_DIR / "classifier_metrics.json"])


if __name__ == "__main__":
    main()