"""Generate a small FAKE dataset in the same file format as the UCI one.

Only for testing the pipeline plumbing without downloading 73 MB. The numbers mean
nothing physically; never report results from this data.
"""
from pathlib import Path

import numpy as np

from src.config import COMPONENTS, OPTIMAL, SENSORS

LEVELS = {
    "cooler": [100, 20, 3],
    "valve": [100, 90, 80, 73],
    "pump_leak": [0, 1, 2],
    "accumulator": [130, 115, 100, 90],
}


def make_synthetic_raw(out_dir: Path, n_cycles: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    prof = np.zeros((n_cycles, 5), dtype=int)
    for j, c in enumerate(COMPONENTS):
        optimal = rng.random(n_cycles) < 0.8
        faulty_levels = rng.choice([v for v in LEVELS[c] if v != OPTIMAL[c]], n_cycles)
        prof[:, j] = np.where(optimal, OPTIMAL[c], faulty_levels)
    prof[:, 4] = rng.integers(0, 2, n_cycles)
    np.savetxt(out_dir / "profile.txt", prof, fmt="%d", delimiter="\t")

    sev = np.column_stack([(prof[:, j] != OPTIMAL[c]).astype(float) for j, c in enumerate(COMPONENTS)])
    for k, (name, n_samples) in enumerate(SENSORS.items()):
        base = 10 + k
        t = np.linspace(0, 1, n_samples)
        shape = np.sin(2 * np.pi * (1 + k % 3) * t)
        mat = base + shape[None, :] + 0.1 * rng.standard_normal((n_cycles, n_samples))
        # each fault shifts a few sensors a little, so detection is possible but not trivial
        mat += (sev[:, k % 4] * 0.6)[:, None] + (sev[:, (k + 1) % 4] * 0.3)[:, None] * t[None, :]
        np.savetxt(out_dir / f"{name}.txt", mat, fmt="%.4f", delimiter="\t")