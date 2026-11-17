"""API tests. The model-backed tools are replaced by fakes, so no trained files are needed.

Run:  pytest -q      (needs fastapi and httpx, both in requirements.txt)
"""
import pytest
from fastapi.testclient import TestClient

from src import api, tools, workorders

client = TestClient(api.app)  # no `with`: skips the model-loading startup hook

SCORE_OK = {"cycle_id": 7, "score": 0.5, "threshold": 0.1459, "ratio": 3.43, "is_anomaly": True}
EXPLAIN_OK = {"cycle_id": 7, "top_sensors": [
    {"sensor": "TS1", "meaning": "temperature 1", "share_of_error": 0.4,
     "most_deviating_statistic": "TS1_mean"}]}
DIAG_OK = {"cycle_id": 7, "severe_components": ["cooler"], "components": {
    "cooler": {"state": 3, "meaning": "close to total failure", "confidence": 0.9, "severe": True},
    "valve": {"state": 100, "meaning": "optimal switching", "confidence": 0.8, "severe": False}}}


@pytest.fixture()
def outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(workorders, "OUTBOX", tmp_path / "outbox")


def test_score_endpoint(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: SCORE_OK)
    r = client.get("/cycles/7/score")
    assert r.status_code == 200 and r.json()["is_anomaly"] is True


def test_tool_error_becomes_422(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: {"error": "cycle_id must be between 0 and 2204"})
    r = client.get("/cycles/99999/score")
    assert r.status_code == 422 and "between 0 and 2204" in r.json()["detail"]


def test_non_integer_cycle_id_is_rejected():
    assert client.get("/cycles/abc/score").status_code == 422


def test_explain_validates_top_k():
    assert client.get("/cycles/7/explain?top_k=0").status_code == 422
    assert client.get("/cycles/7/explain?top_k=11").status_code == 422


def test_triage_normal_cycle_skips_further_tools(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: {**SCORE_OK, "is_anomaly": False})

    def boom(*a, **k):
        raise AssertionError("explain must not be called for a normal cycle")

    monkeypatch.setattr(tools, "explain_anomaly", boom)
    r = client.get("/cycles/7/triage")
    assert r.status_code == 200 and "No action" in r.json()["recommendation"]


def test_triage_anomalous_cycle_names_severe_component(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: SCORE_OK)
    monkeypatch.setattr(tools, "explain_anomaly", lambda cid, k=5: EXPLAIN_OK)
    monkeypatch.setattr(tools, "diagnose_component", lambda cid: DIAG_OK)
    body = client.get("/cycles/7/triage").json()
    assert "cooler" in body["recommendation"]
    assert body["diagnosis"]["components"]["cooler"]["severe"] is True


def test_workorder_approval_flow(outbox):
    workorders.create_draft(7, "cooler", "severe", "TS1 high")
    drafts = client.get("/workorders?status=DRAFT").json()
    assert len(drafts) == 1 and drafts[0]["status"] == "DRAFT"
    oid = drafts[0]["id"]
    r = client.post(f"/workorders/{oid}/approve")
    assert r.status_code == 200 and r.json()["status"] == "APPROVED"
    assert client.post(f"/workorders/{oid}/reject").status_code == 409     # already decided
    assert client.post("/workorders/WO-9999-cooler/approve").status_code == 404
    assert client.get("/workorders?status=BOGUS").status_code == 422


def test_metrics_endpoint_counts_requests(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: SCORE_OK)
    before = client.get("/metrics").json()["requests_total"]
    client.get("/cycles/7/score")
    after = client.get("/metrics").json()
    assert after["requests_total"] > before and after["latency_ms"]["n"] > 0


def test_health_reports_missing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ART_DIR", tmp_path)
    monkeypatch.setattr(api, "PROC_DIR", tmp_path)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"