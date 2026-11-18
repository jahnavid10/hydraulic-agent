"""Batch scoring and feature-drift checks for the monitoring dashboard.

Drift here means: how different is a window of cycles from the training-split reference,
measured per feature with the Population Stability Index (PSI) and summarised per sensor.
Rule of thumb: PSI < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 significant shift.

The dataset has no real timestamps, so "cycle order" is only a stand-in for time.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src import tools
from src.config import PROC_DIR

MIN_WINDOW = 30  # PSI on fewer cycles is too noisy to mean anything


def _batch_errors(x_scaled: np.ndarray) -> np.ndarray:
    from src.model import per_feature_errors

    return per_feature_errors(tools._bundle().ae, x_scaled)


def score_range(start: int, end: int) -> dict:
    b = tools._bundle()
    n = len(b.feats)
    start, end = max(0, int(start)), min(n, int(end))
    if start >= end:
        raise ValueError("start must be smaller than end, and inside 0..{}".format(n))
    window = b.feats.iloc[start:end]
    x = b.scaler.transform(window.to_numpy()).astype(np.float32)
    scores = _batch_errors(x).mean(axis=1)
    points = [{"cycle_id": int(cid), "score": round(float(s), 4),
               "ratio": round(float(s) / b.threshold, 2)}
              for cid, s in zip(window.index, scores)]
    return {"threshold": b.threshold, "points": points}


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of one feature; bins come from reference quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:           # (almost) constant feature: nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


@lru_cache(maxsize=1)
def _reference_mask() -> np.ndarray:
    """True for the cycles that form the reference: the training split (60% fallback)."""
    feats = tools._bundle().feats
    path = PROC_DIR / "labels.csv"
    if path.exists():
        split = pd.read_csv(path, usecols=["cycle_id", "split"], index_col="cycle_id")["split"]
        return (split.reindex(feats.index) == "train").to_numpy()
    mask = np.zeros(len(feats), dtype=bool)
    mask[: int(0.6 * len(feats))] = True
    return mask


def _status(value: float) -> str:
    return "stable" if value < 0.1 else ("moderate" if value < 0.25 else "significant")


def drift_report(start: int, end: int) -> dict:
    feats = tools._bundle().feats
    n = len(feats)
    start, end = max(0, int(start)), min(n, int(end))
    if end - start < MIN_WINDOW:
        raise ValueError(f"window must contain at least {MIN_WINDOW} cycles")

    in_window = np.zeros(n, dtype=bool)
    in_window[start:end] = True
    ref = feats[_reference_mask() & ~in_window]
    cur = feats.iloc[start:end]
    if len(ref) < MIN_WINDOW:
        raise ValueError("too few reference cycles outside the chosen window")

    per_feature = {c: psi(ref[c].to_numpy(), cur[c].to_numpy()) for c in feats.columns}
    per_sensor: dict[str, float] = {}
    for name, value in per_feature.items():
        sensor = name.split("_")[0]
        per_sensor[sensor] = max(per_sensor.get(sensor, 0.0), value)

    sensors = [{"sensor": s, "meaning": tools.SENSOR_INFO.get(s, s), "psi": round(v, 3),
                "status": _status(v)} for s, v in sorted(per_sensor.items(), key=lambda kv: -kv[1])]
    return {"reference": f"training split ({len(ref)} cycles, window excluded)",
            "window": {"start": start, "end": end, "n": end - start},
            "sensors": sensors}