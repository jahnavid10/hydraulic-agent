"""Tests for batch scoring and PSI drift detection (no torch or trained files needed)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src import monitoring, tools


@pytest.fixture()
def fake(monkeypatch):
    rng = np.random.default_rng(0)
    cols = [f"{s}_{st}" for s in ("PS1", "TS1") for st in ("mean", "std")]
    feats = pd.DataFrame(rng.normal(size=(300, 4)), columns=cols)
    feats.index.name = "cycle_id"
    feats.iloc[200:, 2] += 5.0           # TS1_mean shifts in the last 100 cycles
    scaler = StandardScaler().fit(feats.iloc[:150])
    bundle = SimpleNamespace(feats=feats, scaler=scaler, threshold=0.5, ae=None)
    monkeypatch.setattr(tools, "_bundle", lambda: bundle)
    monkeypatch.setattr(monitoring, "_reference_mask",
                        lambda: np.arange(300) < 150)        # first 150 cycles = reference
    monkeypatch.setattr(monitoring, "_batch_errors", lambda x: x ** 2)
    return bundle


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    assert monitoring.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert monitoring.psi(a, rng.normal(size=2000)) < 0.1
    assert monitoring.psi(a, rng.normal(loc=3.0, size=2000)) > 0.25


def test_psi_handles_constant_feature():
    assert monitoring.psi(np.ones(100), np.ones(50)) == 0.0


def test_score_range_returns_one_point_per_cycle(fake):
    out = monitoring.score_range(10, 20)
    assert out["threshold"] == 0.5 and len(out["points"]) == 10
    assert [p["cycle_id"] for p in out["points"]] == list(range(10, 20))
    assert all(p["ratio"] == pytest.approx(p["score"] / 0.5, abs=0.02) for p in out["points"])


def test_score_range_rejects_empty_window(fake):
    with pytest.raises(ValueError):
        monitoring.score_range(50, 50)


def test_drift_report_finds_the_shifted_sensor(fake):
    rep = monitoring.drift_report(200, 300)
    assert rep["window"]["n"] == 100
    assert rep["sensors"][0]["sensor"] == "TS1" and rep["sensors"][0]["status"] == "significant"
    ps1 = next(s for s in rep["sensors"] if s["sensor"] == "PS1")
    assert ps1["status"] != "significant"            # small windows are noisy, but not THIS far off
    assert ps1["psi"] < rep["sensors"][0]["psi"] / 5  # the shifted sensor stands far above it


def test_drift_report_rejects_small_window(fake):
    with pytest.raises(ValueError):
        monitoring.drift_report(0, 10)