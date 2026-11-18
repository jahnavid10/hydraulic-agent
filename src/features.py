"""Turn raw cycle matrices into one compact feature row per cycle.

Raw layout: one tab/space-delimited file per sensor, rows = cycles, columns = samples
inside the 60 s cycle (60, 600 or 6000 columns depending on the sampling rate).
Because the sampling rates differ, we summarise each sensor per cycle with five
statistics: mean, std, min, max and linear slope  ->  17 sensors x 5 = 85 features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COMPONENTS, NORMAL_LEVELS, OPTIMAL, PROFILE_COLS, SENSORS, STAT_NAMES


def extract_stats(matrix: np.ndarray) -> np.ndarray:
    """(n_cycles, n_samples) -> (n_cycles, 5) with columns mean, std, min, max, slope.

    slope = least-squares slope per sample step, computed in closed form (vectorised).
    """
    m = np.asarray(matrix, dtype=np.float64)
    n_samples = m.shape[1]
    t = np.arange(n_samples, dtype=np.float64)
    t_centered = t - t.mean()
    slope = (m - m.mean(axis=1, keepdims=True)) @ t_centered / np.sum(t_centered**2)
    return np.column_stack([m.mean(axis=1), m.std(axis=1), m.min(axis=1), m.max(axis=1), slope])


def feature_names() -> list[str]:
    return [f"{s}_{stat}" for s in SENSORS for stat in STAT_NAMES]


def load_matrix(raw_dir: Path, name: str) -> np.ndarray:
    path = Path(raw_dir) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.download_data` first.")
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32).to_numpy()


def build_feature_table(raw_dir: Path) -> pd.DataFrame:
    blocks, n_rows = [], None
    for name, expected_cols in SENSORS.items():
        m = load_matrix(raw_dir, name)
        if m.shape[1] != expected_cols:
            raise ValueError(f"{name}: expected {expected_cols} columns, got {m.shape[1]}")
        if n_rows is None:
            n_rows = m.shape[0]
        elif m.shape[0] != n_rows:
            raise ValueError(f"{name}: {m.shape[0]} rows, other sensors have {n_rows}")
        blocks.append(extract_stats(m))
        print(f"  {name:5s} {m.shape} -> 5 features")
    df = pd.DataFrame(np.hstack(blocks), columns=feature_names())
    df.index.name = "cycle_id"
    return df


def load_labels(raw_dir: Path) -> pd.DataFrame:
    """profile.txt -> labels with derived columns.

    faulty_<component>: component is in a SEVERE state (outside NORMAL_LEVELS in config.py)
    n_faulty:           how many components are severely degraded in this cycle
    is_healthy:         "normal" cycle: no component severely degraded (mild wear allowed)
    strictly_optimal:   all four components at their optimal value (only a handful of cycles)
    """
    path = Path(raw_dir) / "profile.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.download_data` first.")
    lab = pd.read_csv(path, sep=r"\s+", header=None, names=PROFILE_COLS)
    lab.index.name = "cycle_id"
    for c in COMPONENTS:
        lab[f"faulty_{c}"] = ~lab[c].isin(NORMAL_LEVELS[c])
    lab["n_faulty"] = lab[[f"faulty_{c}" for c in COMPONENTS]].sum(axis=1)
    lab["is_healthy"] = lab["n_faulty"] == 0
    lab["strictly_optimal"] = (lab[COMPONENTS] == pd.Series(OPTIMAL)).all(axis=1)
    return lab