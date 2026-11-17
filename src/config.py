"""Central configuration: paths and dataset constants.

Dataset: UCI "Condition monitoring of hydraulic systems" (id 447), CC BY 4.0.
Helwig, Pignanelli, Schütze (2015). https://doi.org/10.24432/C5CW21
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
ART_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition+monitoring+of+hydraulic+systems.zip"
)

# sensor name -> samples per 60 s cycle (100 Hz = 6000, 10 Hz = 600, 1 Hz = 60)
SENSORS = {
    "PS1": 6000, "PS2": 6000, "PS3": 6000, "PS4": 6000, "PS5": 6000, "PS6": 6000,
    "EPS1": 6000,
    "FS1": 600, "FS2": 600,
    "TS1": 60, "TS2": 60, "TS3": 60, "TS4": 60,
    "VS1": 60, "CE": 60, "CP": 60, "SE": 60,
}

# columns of profile.txt (one row per cycle)
PROFILE_COLS = ["cooler", "valve", "pump_leak", "accumulator", "stable"]
COMPONENTS = ["cooler", "valve", "pump_leak", "accumulator"]

# value of each component when it is in perfect condition
OPTIMAL = {"cooler": 100, "valve": 100, "pump_leak": 0, "accumulator": 130}

NORMAL_LEVELS = {
    "cooler": {100, 20},
    "valve": {100, 90},
    "pump_leak": {0, 1},
    "accumulator": {130, 115},
}

STAT_NAMES = ["mean", "std", "min", "max", "slope"]

SEED = 42
TRAIN_FRAC = 0.60   # of all cycles
VAL_FRAC = 0.20     # of all cycles; the remaining 20% is the held-out test set