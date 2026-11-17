"""Tests for work orders, notes lookup, tool input validation and the classifier.

These run without trained model files or torch: the model-dependent parts are replaced by
small fakes. Run:  pytest -q
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src import tools, workorders
from src.build_dataset import assign_splits
from src.config import COMPONENTS
from src.features import build_feature_table, load_labels
from src.train_clf import fit_and_eval
from tests.synthetic import make_synthetic_raw


@pytest.fixture()
def outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(workorders, "OUTBOX", tmp_path / "outbox")
    return tmp_path / "outbox"


@pytest.fixture()
def fake_bundle(monkeypatch):
    names = [f"{s}_{st}" for s in ("PS1", "TS1", "CE") for st in ("mean", "std")]
    bundle = SimpleNamespace(feature_names=names, feats=list(range(10)), threshold=1.0)
    monkeypatch.setattr(tools, "_bundle", lambda: bundle)
    monkeypatch.setattr(tools, "_errors", lambda cid: np.array([0.1, 0.1, 3.0, 1.0, 0.0, 0.0]))
    return bundle


def test_notes_lookup_finds_the_right_section():
    top = tools.lookup_fault_notes("cooler temperature efficiency")["matches"][0]
    assert top["section"].startswith("Cooler")
    top = tools.lookup_fault_notes("accumulator nitrogen pre-charge")["matches"][0]
    assert top["section"].startswith("Accumulator")


def test_notes_lookup_rejects_empty_query():
    assert "error" in tools.lookup_fault_notes("   ")


def test_work_order_lifecycle(outbox):
    d = workorders.create_draft(412, "pump_leak", "severe", "score 3x threshold")
    assert d["status"] == "DRAFT" and d["priority"] == "high" and d["id"] == "WO-0412-pump_leak"
    assert len(workorders.list_orders("DRAFT")) == 1
    approved = workorders.set_status(d["id"], "APPROVED")
    assert approved["status"] == "APPROVED" and "decided_at" in approved
    with pytest.raises(ValueError):          # cannot decide twice
        workorders.set_status(d["id"], "REJECTED")
    with pytest.raises(ValueError):          # only APPROVED / REJECTED are allowed
        workorders.set_status(d["id"], "DRAFT")
    with pytest.raises(KeyError):
        workorders.set_status("WO-9999-cooler", "APPROVED")


def test_draft_tool_validates_input(outbox, fake_bundle):
    ok = tools.draft_work_order(3, "cooler", "mild", "temperatures slightly high")
    assert ok["status"] == "DRAFT" and "NOT filed" in ok["message"]
    assert "error" in tools.draft_work_order(3, "engine", "mild", "x")
    assert "error" in tools.draft_work_order(3, "cooler", "critical", "x")
    assert "error" in tools.draft_work_order(3, "cooler", "mild", "  ")
    assert "error" in tools.draft_work_order(99, "cooler", "mild", "x")   # id out of range
    assert len(workorders.list_orders()) == 1                              # only the valid one


def test_agent_cannot_approve():
    names = {f.__name__ for f in tools.AGENT_TOOLS}
    assert "set_status" not in names and not any("approve" in n for n in names)


def test_explain_anomaly_aggregates_by_sensor(fake_bundle):
    out = tools.explain_anomaly(5, top_k=2)
    assert [r["sensor"] for r in out["top_sensors"]] == ["TS1", "PS1"]
    assert out["top_sensors"][0]["most_deviating_statistic"] == "TS1_mean"
    total = sum(r["share_of_error"] for r in tools.explain_anomaly(5, top_k=10)["top_sensors"])
    assert total == pytest.approx(1.0, abs=0.01)


def test_cycle_id_validation(fake_bundle):
    assert "error" in tools.explain_anomaly(-1)
    assert "error" in tools.explain_anomaly("abc")
    assert "error" in tools.explain_anomaly(2.5)
    assert tools.explain_anomaly(7.0)["cycle_id"] == 7


def test_classifier_trains_and_reports_all_components(tmp_path):
    make_synthetic_raw(tmp_path, n_cycles=200)
    feats = build_feature_table(tmp_path)
    labels = load_labels(tmp_path)
    labels["split"] = assign_splits(labels["is_healthy"].to_numpy())
    clf, metrics = fit_and_eval(feats, labels, n_estimators=50)
    for c in COMPONENTS:
        assert 0.0 <= metrics[f"accuracy_{c}"] <= 1.0
    assert len(clf.classes_) == 4