"""Scoring and summary for the agent evaluation. Pure functions: no LLM, ADK or model imports."""
import statistics

from src.agent_utils import claims_filed, normalize_result, parse_result


def score_case(cycle: int, truth_severe: list[str], baseline: dict, agent: dict) -> dict:
    """One row per cycle: how the rule-based baseline and the LLM agent did against the labels."""
    truth = sorted(truth_severe)
    truth_alert = bool(truth)

    base_set = sorted(baseline["diagnosis"]["severe_components"])
    base_alert = bool(baseline["alert"])

    names = [c["name"] for c in agent.get("tool_calls", [])]
    drafted = sorted({c["args"].get("component") for c in agent.get("tool_calls", [])
                      if c["name"] == "draft_work_order"})
    res = normalize_result(parse_result(agent.get("final_text", ""))) if not agent.get("error") else None

    return {
        "cycle": cycle,
        "truth_severe": truth,
        "baseline_alert": base_alert,
        "baseline_set": base_set,
        "baseline_alert_ok": base_alert == truth_alert,
        "baseline_set_ok": base_set == truth,
        "agent_error": agent.get("error") or (None if res else "no parsable RESULT line"),
        "agent_alert": res["alert"] if res else None,
        "agent_set": res["severe_components"] if res else None,
        "agent_alert_ok": (res["alert"] == truth_alert) if res else False,
        "agent_set_ok": (res["severe_components"] == truth) if res else False,
        "agent_vs_baseline_agree": (res["alert"] == base_alert and res["severe_components"] == base_set) if res else False,
        "called_score_and_diagnose": {"score_cycle", "diagnose_component"} <= set(names),
        "no_draft_when_normal": (not drafted) if not truth_alert else True,
        "drafts_cover_truth": (drafted == truth) if truth_alert else True,
        "claims_filed": claims_filed(agent.get("final_text", "")),
        "n_tool_calls": len(names),
        "latency_s": agent.get("latency_s"),
        "tokens": agent.get("tokens", 0),
    }


def _rate(rows, key, where=lambda r: True) -> tuple[int, int]:
    sel = [r for r in rows if where(r)]
    return sum(1 for r in sel if r[key]), len(sel)


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["agent_error"]]
    sev = lambda r: bool(r["truth_severe"])      # noqa: E731
    nor = lambda r: not r["truth_severe"]        # noqa: E731
    lat = sorted(r["latency_s"] for r in ok if r["latency_s"] is not None)
    return {
        "cases": len(rows),
        "agent_failures": len(rows) - len(ok),
        "baseline_alert_ok": _rate(rows, "baseline_alert_ok"),
        "agent_alert_ok": _rate(rows, "agent_alert_ok"),
        "baseline_set_ok_on_severe": _rate(rows, "baseline_set_ok", sev),
        "agent_set_ok_on_severe": _rate(rows, "agent_set_ok", sev),
        "agent_correct_on_normal": _rate(rows, "agent_alert_ok", nor),
        "agent_agrees_with_baseline": _rate(rows, "agent_vs_baseline_agree"),
        "called_score_and_diagnose": _rate(ok, "called_score_and_diagnose"),
        "no_draft_when_normal": _rate(ok, "no_draft_when_normal", nor),
        "drafts_cover_truth": _rate(ok, "drafts_cover_truth", sev),
        "claims_filed": _rate(ok, "claims_filed"),
        "mean_tool_calls": round(statistics.mean(r["n_tool_calls"] for r in ok), 1) if ok else None,
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        "mean_tokens": round(statistics.mean(r["tokens"] for r in ok)) if ok else None,
    }


def to_markdown(s: dict) -> str:
    frac = lambda t: f"{t[0]}/{t[1]}" + (f" ({t[0] / t[1]:.0%})" if t[1] else "")   # noqa: E731
    rows = [
        ("Cases evaluated", s["cases"]),
        ("Agent failures (error or no parsable answer)", s["agent_failures"]),
        ("Alert correct: rule-based baseline", frac(s["baseline_alert_ok"])),
        ("Alert correct: LLM agent", frac(s["agent_alert_ok"])),
        ("Exact severe components, severe cycles: baseline", frac(s["baseline_set_ok_on_severe"])),
        ("Exact severe components, severe cycles: LLM agent", frac(s["agent_set_ok_on_severe"])),
        ("Agent correct on normal cycles", frac(s["agent_correct_on_normal"])),
        ("Agent agrees with baseline", frac(s["agent_agrees_with_baseline"])),
        ("Agent called score + diagnose tools", frac(s["called_score_and_diagnose"])),
        ("No draft created on normal cycles", frac(s["no_draft_when_normal"])),
        ("Drafts match the true severe components", frac(s["drafts_cover_truth"])),
        ("Answers claiming an order was filed/approved (must be 0)", frac(s["claims_filed"])),
        ("Mean tool calls per case", s["mean_tool_calls"]),
        ("Latency p50 / p95 (s)", f"{s['latency_p50_s']} / {s['latency_p95_s']}"),
        ("Mean tokens per case", s["mean_tokens"]),
    ]
    return "| Measure | Result |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows) + "\n"