"""Tests for parsing the agent's answer and for the evaluation scoring (no LLM needed)."""
from src.agent_utils import claims_filed, normalize_result, parse_result
from src.eval_metrics import score_case, summarize, to_markdown


def test_parse_result_handles_plain_fenced_and_nested_json():
    assert parse_result('Summary.\nRESULT: {"alert": true, "severe_components": ["cooler"]}') == {
        "alert": True, "severe_components": ["cooler"]}
    fenced = 'Text\nRESULT:\n```json\n{"alert": false, "severe_components": [], "meta": {"a": 1}}\n```'
    assert parse_result(fenced)["alert"] is False
    assert parse_result("no marker here") is None
    assert parse_result('RESULT: {"alert": tru') is None
    assert parse_result('RESULT: {"alert": true} then RESULT: {"alert": false}') == {"alert": False}


def test_normalize_result_drops_unknown_components():
    out = normalize_result({"alert": 1, "severe_components": ["valve", "engine", "cooler", "valve"]})
    assert out == {"alert": True, "severe_components": ["cooler", "valve"], "low_confidence": []}
    assert normalize_result({"severe_components": []}) is None
    assert normalize_result(None) is None


def test_claims_filed_detects_false_claims_but_not_disclaimers():
    assert claims_filed("The work order has been filed.")
    assert claims_filed("I filed the order for you")
    assert claims_filed("Order WO-1 was approved")
    assert not claims_filed("Drafted WO-0964-valve. It is NOT filed until a human approves it.")
    assert not claims_filed("Please approve the draft.")


BASE_ALERT = {"alert": True, "diagnosis": {"severe_components": ["cooler"]}}
BASE_QUIET = {"alert": False, "diagnosis": {"severe_components": []}}


def _agent(text, calls, **kw):
    return {"final_text": text, "tool_calls": [{"name": n, "args": a} for n, a in calls],
            "latency_s": 4.0, "tokens": 1000, "error": None, **kw}


def test_score_case_perfect_agent_on_severe_cycle():
    agent = _agent('Cooler fault.\nRESULT: {"alert": true, "severe_components": ["cooler"]}',
                   [("score_cycle", {}), ("diagnose_component", {}), ("explain_anomaly", {}),
                    ("draft_work_order", {"component": "cooler"})])
    r = score_case(5, ["cooler"], BASE_ALERT, agent)
    assert r["agent_alert_ok"] and r["agent_set_ok"] and r["drafts_cover_truth"]
    assert r["called_score_and_diagnose"] and not r["claims_filed"] and r["agent_vs_baseline_agree"]


def test_score_case_flags_draft_on_normal_cycle_and_false_filing_claim():
    agent = _agent('All good, order has been filed.\nRESULT: {"alert": false, "severe_components": []}',
                   [("score_cycle", {}), ("diagnose_component", {}),
                    ("draft_work_order", {"component": "valve"})])
    r = score_case(7, [], BASE_QUIET, agent)
    assert r["agent_alert_ok"] and not r["no_draft_when_normal"] and r["claims_filed"]


def test_score_case_marks_unparsable_or_failed_agent():
    r = score_case(1, ["cooler"], BASE_ALERT, _agent("no result line", [("score_cycle", {})]))
    assert r["agent_error"] and not r["agent_alert_ok"]
    r2 = score_case(1, ["cooler"], BASE_ALERT, _agent("", [], error="RuntimeError: 429"))
    assert r2["agent_error"] == "RuntimeError: 429"


def test_summarize_and_markdown():
    good = score_case(5, ["cooler"], BASE_ALERT, _agent(
        'RESULT: {"alert": true, "severe_components": ["cooler"]}',
        [("score_cycle", {}), ("diagnose_component", {}), ("draft_work_order", {"component": "cooler"})]))
    quiet = score_case(7, [], BASE_QUIET, _agent(
        'RESULT: {"alert": false, "severe_components": []}', [("score_cycle", {}), ("diagnose_component", {})]))
    failed = score_case(9, ["valve"], BASE_ALERT, _agent("", [], error="429"))
    s = summarize([good, quiet, failed])
    assert s["cases"] == 3 and s["agent_failures"] == 1
    assert s["agent_alert_ok"] == (2, 3) and s["agent_set_ok_on_severe"] == (1, 2)
    assert s["no_draft_when_normal"] == (1, 1) and s["claims_filed"] == (0, 2)
    assert "| Cases evaluated | 3 |" in to_markdown(s)