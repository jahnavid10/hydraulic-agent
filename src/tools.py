"""The five tools the agent can call.

Plain Python functions: the agent framework reads each signature and docstring to decide
when and how to call them, so keep docstrings precise. Every tool returns a dict and reports
problems as {"error": "..."} instead of raising, so the agent can recover.

Tools read sensor features from data/processed/features.csv (a stand-in for a sensor
database). They never read the ground-truth labels.
"""
import json
import re
from functools import lru_cache
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from src import workorders
from src.config import ART_DIR, COMPONENTS, NORMAL_LEVELS, PROC_DIR, ROOT

NOTES_PATH = ROOT / "notes" / "fault_notes.md"

SENSOR_INFO = {
    "PS1": "pressure sensor 1", "PS2": "pressure sensor 2", "PS3": "pressure sensor 3",
    "PS4": "pressure sensor 4", "PS5": "pressure sensor 5", "PS6": "pressure sensor 6",
    "EPS1": "motor power", "FS1": "volume flow 1", "FS2": "volume flow 2",
    "TS1": "temperature 1", "TS2": "temperature 2", "TS3": "temperature 3",
    "TS4": "temperature 4", "VS1": "vibration", "CE": "cooling efficiency (virtual)",
    "CP": "cooling power (virtual)", "SE": "efficiency factor (virtual)",
}

STATE_MEANING = {
    "cooler": {100: "full efficiency", 20: "reduced efficiency", 3: "close to total failure"},
    "valve": {100: "optimal switching", 90: "small lag", 80: "severe lag",
              73: "close to total failure"},
    "pump_leak": {0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
    "accumulator": {130: "optimal pressure", 115: "slightly reduced pressure",
                    100: "severely reduced pressure", 90: "close to total failure"},
}


@lru_cache(maxsize=1)
def _bundle() -> SimpleNamespace:
    """Load models and data once. torch is imported here so the rest stays importable without it."""
    from src.model import load_model

    th = json.loads((ART_DIR / "threshold.json").read_text())
    clf = joblib.load(ART_DIR / "clf.joblib")
    return SimpleNamespace(
        ae=load_model(ART_DIR / "ae.pt"),
        scaler=joblib.load(ART_DIR / "scaler.joblib"),
        clf=clf["model"],
        threshold=float(th["threshold"]),
        feature_names=list(th["feature_names"]),
        feats=pd.read_csv(PROC_DIR / "features.csv", index_col="cycle_id"),
    )


def _clean_id(cycle_id):
    """Return (int id, None) or (None, error dict)."""
    try:
        cid = int(cycle_id)
        if cid != float(cycle_id):
            raise ValueError
    except (TypeError, ValueError):
        return None, {"error": f"cycle_id must be an integer, got {cycle_id!r}"}
    n = len(_bundle().feats)
    if not 0 <= cid < n:
        return None, {"error": f"cycle_id must be between 0 and {n - 1}, got {cid}"}
    return cid, None


def _errors(cid: int) -> np.ndarray:
    """Per-feature squared reconstruction error for one cycle, shape (n_features,)."""
    from src.model import per_feature_errors

    b = _bundle()
    x = b.scaler.transform(b.feats.iloc[[cid]].to_numpy()).astype(np.float32)
    return per_feature_errors(b.ae, x)[0]


def score_cycle(cycle_id: int) -> dict:
    """Score one sensor cycle with the anomaly detector.

    Args:
        cycle_id: integer id of the cycle, from 0 to 2204.
    Returns:
        anomaly score, the alert threshold, and is_anomaly (True means the cycle looks like
        a severe fault). ratio = score / threshold; larger means more abnormal.
    """
    cid, err = _clean_id(cycle_id)
    if err:
        return err
    score = float(_errors(cid).mean())
    thr = _bundle().threshold
    return {"cycle_id": cid, "score": round(score, 4), "threshold": round(thr, 4),
            "ratio": round(score / thr, 2), "is_anomaly": bool(score > thr)}


def explain_anomaly(cycle_id: int, top_k: int = 5) -> dict:
    """Say which sensors contribute most to a cycle's anomaly score.

    Args:
        cycle_id: integer id of the cycle, from 0 to 2204.
        top_k: how many sensors to return (1 to 10).
    Returns:
        the top sensors with their share of the total reconstruction error and the most
        deviating statistic of each (for example PS2_mean).
    """
    cid, err = _clean_id(cycle_id)
    if err:
        return err
    top_k = max(1, min(int(top_k), 10))
    errs = _errors(cid)
    names = _bundle().feature_names
    total = float(errs.sum())
    groups: dict[str, list] = {}
    for name, e in zip(names, errs):
        groups.setdefault(name.split("_")[0], []).append((float(e), name))
    rows = []
    for sensor, items in groups.items():
        rows.append({
            "sensor": sensor,
            "meaning": SENSOR_INFO.get(sensor, sensor),
            "share_of_error": round(sum(e for e, _ in items) / total, 3) if total > 0 else 0.0,
            "most_deviating_statistic": max(items)[1],
        })
    rows.sort(key=lambda r: -r["share_of_error"])
    return {"cycle_id": cid, "top_sensors": rows[:top_k]}


def diagnose_component(cycle_id: int) -> dict:
    """Estimate the condition of all four components (cooler, valve, pump_leak, accumulator).

    Args:
        cycle_id: integer id of the cycle, from 0 to 2204.
    Returns:
        for each component its estimated state, a plain-language meaning, a confidence
        between 0 and 1, and whether the state counts as severe. Also lists severe components.
    """
    cid, err = _clean_id(cycle_id)
    if err:
        return err
    b = _bundle()
    proba = b.clf.predict_proba(b.feats.iloc[[cid]].to_numpy())  # list of (1, n_classes) arrays
    result, severe = {}, []
    for j, comp in enumerate(COMPONENTS):
        k = int(np.argmax(proba[j][0]))
        state = int(b.clf.classes_[j][k])
        is_severe = state not in NORMAL_LEVELS[comp]
        result[comp] = {
            "state": state,
            "meaning": STATE_MEANING[comp].get(state, "unknown"),
            "confidence": round(float(proba[j][0][k]), 2),
            "severe": is_severe,
        }
        if is_severe:
            severe.append(comp)
    return {"cycle_id": cid, "components": result, "severe_components": severe}


@lru_cache(maxsize=1)
def _notes_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = NOTES_PATH.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    sections = []
    for p in parts:
        title, _, body = p.partition("\n")
        sections.append({"section": title.strip(), "text": body.strip()})
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform([s["section"] + " " + s["text"] for s in sections])
    return sections, vec, matrix


def lookup_fault_notes(query: str) -> dict:
    """Search the maintenance fault notes for guidance on a component or symptom.

    Args:
        query: short search text, for example "pump leakage temperature rise" or "cooler".
    Returns:
        up to two matching note sections with their text and a similarity score.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "query must be a non-empty string"}
    sections, vec, matrix = _notes_index()
    sims = (matrix @ vec.transform([query]).T).toarray().ravel()
    order = np.argsort(-sims)[:2]
    matches = [{**sections[i], "similarity": round(float(sims[i]), 3)} for i in order if sims[i] > 0]
    if not matches:
        return {"matches": [], "note": "no relevant notes found"}
    return {"matches": matches}


def draft_work_order(cycle_id: int, component: str, severity: str, evidence: str) -> dict:
    """Create a DRAFT maintenance work order. It is saved as a draft only and is NOT filed;
    a human must approve it separately.

    Args:
        cycle_id: integer id of the cycle that triggered the order.
        component: one of cooler, valve, pump_leak, accumulator.
        severity: "mild" or "severe".
        evidence: one or two sentences citing the tool results (score, sensors, state).
    Returns:
        the saved draft, including its id and status DRAFT.
    """
    cid, err = _clean_id(cycle_id)
    if err:
        return err
    if component not in COMPONENTS:
        return {"error": f"component must be one of {COMPONENTS}"}
    if severity not in {"mild", "severe"}:
        return {"error": "severity must be 'mild' or 'severe'"}
    if not isinstance(evidence, str) or not evidence.strip():
        return {"error": "evidence must be a non-empty string"}
    order = workorders.create_draft(cid, component, severity, evidence.strip())
    return {**order, "message": "Draft saved. It is NOT filed until a human approves it."}


AGENT_TOOLS = [score_cycle, explain_anomaly, diagnose_component, lookup_fault_notes,
               draft_work_order]