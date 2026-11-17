"""API tests. The model-backed tools are replaced by fakes, so no trained files are needed.

Run:  pytest -q      (needs fastapi and httpx, both in requirements.txt)
"""
import pytest
from fastapi.testclient import TestClient

from types import SimpleNamespace

from src import api, monitoring, tools, workorders

client = TestClient(api.app)  # no `with`: skips the model-loading startup hook

SCORE_OK = {"cycle_id": 7, "score": 0.5, "threshold": 0.1459, "ratio": 3.43, "is_anomaly": True}
EXPLAIN_OK = {"cycle_id": 7, "top_sensors": [
    {"sensor": "TS1", "meaning": "temperature 1", "share_of_error": 0.4,
     "most_deviating_statistic": "TS1_mean"}]}
DIAG_OK = {"cycle_id": 7, "severe_components": ["cooler"], "components": {
    "cooler": {"state": 3, "meaning": "close to total failure", "confidence": 0.9, "severe": True},
    "valve": {"state": 100, "meaning": "optimal switching", "confidence": 0.8, "severe": False}}}
DIAG_NONE = {"cycle_id": 7, "severe_components": [], "components": {
    "cooler": {"state": 100, "meaning": "full efficiency", "confidence": 0.9, "severe": False}}}


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


def test_triage_normal_cycle_has_no_alert_and_skips_explain(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: {**SCORE_OK, "is_anomaly": False})
    monkeypatch.setattr(tools, "diagnose_component", lambda cid: DIAG_NONE)

    def boom(*a, **k):
        raise AssertionError("explain must not be called when there is no alert")

    monkeypatch.setattr(tools, "explain_anomaly", boom)
    body = client.get("/cycles/7/triage").json()
    assert body["alert"] is False and body["alert_source"] == "none"
    assert "No action" in body["recommendation"]


def test_triage_anomalous_cycle_names_severe_component(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: SCORE_OK)
    monkeypatch.setattr(tools, "explain_anomaly", lambda cid, k=5: EXPLAIN_OK)
    monkeypatch.setattr(tools, "diagnose_component", lambda cid: DIAG_OK)
    body = client.get("/cycles/7/triage").json()
    assert body["alert"] is True and body["alert_source"] == "both"
    assert "cooler" in body["recommendation"]
    assert body["diagnosis"]["components"]["cooler"]["severe"] is True


def test_triage_classifier_catches_what_the_detector_misses(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: {**SCORE_OK, "is_anomaly": False, "ratio": 0.7})
    monkeypatch.setattr(tools, "explain_anomaly", lambda cid, k=5: EXPLAIN_OK)
    monkeypatch.setattr(tools, "diagnose_component", lambda cid: DIAG_OK)
    body = client.get("/cycles/7/triage").json()
    assert body["alert"] is True and body["alert_source"] == "classifier"
    assert "classifier only" in body["recommendation"]


def test_triage_detector_only_asks_for_manual_inspection(monkeypatch):
    monkeypatch.setattr(tools, "score_cycle", lambda cid: SCORE_OK)
    monkeypatch.setattr(tools, "explain_anomaly", lambda cid, k=5: EXPLAIN_OK)
    monkeypatch.setattr(tools, "diagnose_component", lambda cid: DIAG_NONE)
    body = client.get("/cycles/7/triage").json()
    assert body["alert"] is True and body["alert_source"] == "detector"
    assert "Inspect manually" in body["recommendation"]


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


def test_monitoring_scores_endpoint(monkeypatch):
    fake = {"threshold": 0.1459, "points": [{"cycle_id": 0, "score": 0.1, "ratio": 0.69}]}
    monkeypatch.setattr(monitoring, "score_range", lambda a, b: fake)
    r = client.get("/monitoring/scores?start=0&end=10")
    assert r.status_code == 200 and r.json()["points"][0]["cycle_id"] == 0


def test_monitoring_errors_map_to_http_codes(monkeypatch):
    def bad(a, b):
        raise ValueError("window must contain at least 30 cycles")

    monkeypatch.setattr(monitoring, "drift_report", bad)
    r = client.get("/monitoring/drift?start=0&end=5")
    assert r.status_code == 422 and "at least 30" in r.json()["detail"]

    def missing(a, b):
        raise FileNotFoundError("ae.pt")

    monkeypatch.setattr(monitoring, "score_range", missing)
    assert client.get("/monitoring/scores").status_code == 503
    assert client.get("/monitoring/scores?start=-1").status_code == 422


def test_create_workorder_endpoint(outbox, monkeypatch):
    monkeypatch.setattr(tools, "_bundle", lambda: SimpleNamespace(feats=list(range(10))))
    ok = client.post("/workorders", json={"cycle_id": 3, "component": "cooler",
                                          "severity": "severe", "evidence": "TS1 high"})
    assert ok.status_code == 201 and ok.json()["status"] == "DRAFT"
    bad = client.post("/workorders", json={"cycle_id": 3, "component": "engine",
                                           "severity": "severe", "evidence": "x"})
    assert bad.status_code == 422
    assert len(client.get("/workorders?status=DRAFT").json()) == 1


def test_agent_endpoint_returns_result_and_maps_errors(monkeypatch):
    ok = {"final_text": 'Drafted.\nRESULT: {"alert": true, "severe_components": ["cooler"]}',
          "tool_calls": [], "tokens": 10, "latency_s": 1.0, "error": None, "cycle_id": 7}
    monkeypatch.setattr(api, "_run_agent", lambda cid: ok)
    r = client.post("/agent/diagnose", json={"cycle_id": 7})
    assert r.status_code == 200 and r.json()["result"]["severe_components"] == ["cooler"]
    monkeypatch.setattr(api, "_run_agent", lambda cid: {**ok, "error": "RuntimeError: 429"})
    assert client.post("/agent/diagnose", json={"cycle_id": 7}).status_code == 502
    assert client.post("/agent/diagnose", json={"cycle_id": 99999}).status_code == 422