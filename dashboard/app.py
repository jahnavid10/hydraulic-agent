"""Streamlit dashboards for the hydraulic maintenance project.

Two views, both talking to the FastAPI service only (no model code here):
  Operator   - triage a cycle, create a draft work order, approve or reject drafts.
  Monitoring - API health and latency, anomaly score across cycles, feature drift.

Run locally:  streamlit run dashboard/app.py        (API must be running)
"""
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
COMPONENTS = ["cooler", "valve", "pump_leak", "accumulator"]
N_CYCLES = 2205

st.set_page_config(page_title="Hydraulic Maintenance Agent", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the API. Returns (json, None) on success or (None, error message)."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL}: {exc}"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}"
    return r.json(), None


def sidebar_status() -> None:
    with st.sidebar:
        st.title("Hydraulic agent")
        st.caption(f"API: {API_URL}")
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            body = r.json()
            if r.status_code == 200:
                st.success(f"API healthy (v{body.get('version')})")
            else:
                st.error(f"API degraded, missing: {', '.join(body.get('missing_files', []))}")
        except (requests.RequestException, ValueError) as exc:
            st.error(f"API unreachable: {exc}")
        st.caption("Drafts are never filed by the system. A human approves every work order.")


# ------------------------------------------------------------------ operator view
def default_evidence(t: dict) -> str:
    a = t["anomaly"]
    sensors = ", ".join(s["sensor"] for s in t.get("top_sensors", [])[:3]) or "n/a"
    severe = [f"{c}={v['state']} ({v['meaning']})"
              for c, v in t["diagnosis"]["components"].items() if v["severe"]]
    return (f"Cycle {t['cycle_id']}: anomaly score {a['score']} = {a['ratio']}x threshold. "
            f"Top contributing sensors: {sensors}. Classifier: {', '.join(severe) or 'no severe component'}.")


def render_triage(t: dict) -> None:
    a = t["anomaly"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomaly score", a["score"])
    m2.metric("Threshold", a["threshold"])
    m3.metric("Score / threshold", a["ratio"])
    m4.metric("Alert", f"YES ({t['alert_source']})" if t["alert"] else "no")
    (st.error if t["alert"] else st.success)(t["recommendation"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Component condition (classifier)**")
        diag = pd.DataFrame([{"component": c, "state": v["state"], "meaning": v["meaning"],
                              "confidence": v["confidence"], "severe": v["severe"]}
                             for c, v in t["diagnosis"]["components"].items()])
        st.dataframe(diag, hide_index=True)
    with right:
        if t["top_sensors"]:
            st.markdown("**Sensors contributing most to the anomaly score**")
            sens = pd.DataFrame(t["top_sensors"]).set_index("sensor")["share_of_error"]
            st.bar_chart(sens)

    if t["alert"]:
        severe = t["diagnosis"]["severe_components"]
        with st.form(f"draft_form_{t['cycle_id']}"):
            st.markdown("**Create a draft work order** (stays a draft until approved below)")
            comp = st.selectbox("Component", COMPONENTS,
                                index=COMPONENTS.index(severe[0]) if severe else 0)
            sev = st.selectbox("Severity", ["severe", "mild"], index=0 if severe else 1)
            evidence = st.text_area("Evidence", value=default_evidence(t))
            if st.form_submit_button("Create draft"):
                order, err = api("POST", "/workorders", json={
                    "cycle_id": t["cycle_id"], "component": comp, "severity": sev,
                    "evidence": evidence})
                if err:
                    st.error(err)
                else:
                    st.success(f"Draft {order['id']} created.")


def work_orders_view() -> None:
    st.subheader("Work orders")
    orders, err = api("GET", "/workorders")
    if err:
        st.error(err)
        return
    drafts = [o for o in orders if o["status"] == "DRAFT"]
    decided = [o for o in orders if o["status"] != "DRAFT"]

    st.markdown(f"**Awaiting approval: {len(drafts)}**")
    for o in drafts:
        with st.container(border=True):
            c1, c2, c3 = st.columns([6, 1, 1])
            c1.markdown(f"**{o['id']}** · {o['component']} · {o['severity']} (priority {o['priority']})")
            c1.caption(o["evidence"])
            for col, action, label in ((c2, "approve", "Approve"), (c3, "reject", "Reject")):
                if col.button(label, key=f"{action}_{o['id']}"):
                    _, e = api("POST", f"/workorders/{o['id']}/{action}")
                    if e:
                        st.error(e)
                    else:
                        st.rerun()
    if decided:
        st.markdown("**Decided**")
        cols = ["id", "component", "severity", "status", "decided_at"]
        st.dataframe(pd.DataFrame(decided).reindex(columns=cols), hide_index=True)


def operator_view() -> None:
    st.subheader("Cycle triage")
    c1, c2 = st.columns([1, 3])
    cycle_id = c1.number_input("Cycle id", min_value=0, max_value=N_CYCLES - 1, value=0, step=1)
    if c1.button("Run triage", type="primary"):
        data, err = api("GET", f"/cycles/{int(cycle_id)}/triage")
        if err:
            st.session_state.pop("triage", None)
            st.error(err)
        else:
            st.session_state["triage"] = data
    if "triage" in st.session_state:
        render_triage(st.session_state["triage"])
    st.divider()
    work_orders_view()


# ------------------------------------------------------------------ monitoring view
def monitoring_view() -> None:
    st.subheader("API health and traffic")
    m, err = api("GET", "/metrics")
    if err:
        st.error(err)
    else:
        lat = m["latency_ms"]
        a, b, c, d = st.columns(4)
        a.metric("Requests since start", m["requests_total"])
        b.metric("Latency p50 (ms)", lat["p50"] if lat["p50"] is not None else "n/a")
        c.metric("Latency p95 (ms)", lat["p95"] if lat["p95"] is not None else "n/a")
        d.metric("Server errors (5xx)", m["errors_5xx"])
        if m["by_path"]:
            st.dataframe(pd.DataFrame(sorted(m["by_path"].items(), key=lambda kv: -kv[1]),
                                      columns=["endpoint", "requests"]), hide_index=True)

    st.divider()
    st.subheader("Anomaly score across cycles")
    c1, c2 = st.columns(2)
    start = c1.number_input("From cycle", 0, N_CYCLES - 1, 0, key="s_start")
    end = c2.number_input("To cycle (exclusive)", 1, N_CYCLES, N_CYCLES, key="s_end")
    data, err = api("GET", "/monitoring/scores", params={"start": int(start), "end": int(end)})
    if err:
        st.error(err)
    else:
        df = pd.DataFrame(data["points"]).set_index("cycle_id")
        plot = pd.DataFrame({
            "log10(score / threshold)": np.log10(df["ratio"].clip(lower=1e-3)),
            "alert threshold": 0.0,
        })
        st.line_chart(plot)
        st.caption("Above 0 means the anomaly detector alerts. A log scale is used because scores "
                   "range over several orders of magnitude. Cycle order is not real time.")
        st.metric("Cycles above threshold", f"{(df['ratio'] > 1).mean():.0%}")

    st.divider()
    st.subheader("Feature drift vs training data")
    c1, c2 = st.columns(2)
    d_start = c1.number_input("Window from cycle", 0, N_CYCLES - 1, 1500, key="d_start")
    d_end = c2.number_input("Window to cycle (exclusive)", 1, N_CYCLES, N_CYCLES, key="d_end")
    rep, err = api("GET", "/monitoring/drift", params={"start": int(d_start), "end": int(d_end)})
    if err:
        st.error(err)
    else:
        st.caption(f"Reference: {rep['reference']}. Window: cycles {rep['window']['start']} to "
                   f"{rep['window']['end']} ({rep['window']['n']} cycles).")
        sens = pd.DataFrame(rep["sensors"])
        st.bar_chart(sens.set_index("sensor")["psi"])
        st.dataframe(sens, hide_index=True)
        st.caption("PSI per sensor (maximum over its statistics): below 0.1 stable, 0.1 to 0.25 "
                   "moderate, above 0.25 significant. Small windows inflate PSI a little. This "
                   "dataset has no real timestamps, so drift here is a demonstration of the method.")


def main() -> None:
    sidebar_status()
    operator, monitoring = st.tabs(["Operator", "Monitoring"])
    with operator:
        operator_view()
    with monitoring:
        monitoring_view()


main()