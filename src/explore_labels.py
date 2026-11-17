"""Print how many cycles fall into each condition level, and how many count as normal.

Usage:  python -m src.explore_labels
"""
import pandas as pd

from src.config import COMPONENTS, NORMAL_LEVELS, OPTIMAL, PROFILE_COLS, RAW_DIR

lab = pd.read_csv(RAW_DIR / "profile.txt", sep=r"\s+", header=None, names=PROFILE_COLS)
print(f"cycles: {len(lab)}\n")
for c in COMPONENTS:
    print(f"{c}: {lab[c].value_counts().sort_index().to_dict()}")

strict = (lab[COMPONENTS] == pd.Series(OPTIMAL)).all(axis=1)
normal = pd.concat([lab[c].isin(NORMAL_LEVELS[c]) for c in COMPONENTS], axis=1).all(axis=1)
print(f"\nall four optimal (strict):            {int(strict.sum())}")
print(f"normal = no component severe (rule):  {int(normal.sum())}   <- needs to be a few hundred")
print(f"severe cycles (anomalies):            {int((~normal).sum())}")