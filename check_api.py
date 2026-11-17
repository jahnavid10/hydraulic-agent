"""Compare the running API's /triage output with the true labels on held-out TEST cycles.

Start the API first (docker compose up, or uvicorn src.api:app), then:
    python check_api.py
    python check_api.py --n-normal 10 --n-severe 30 --url http://localhost:8000
"""
import argparse
import json
import urllib.request

import pandas as pd

COMPONENTS = ["cooler", "valve", "pump_leak", "accumulator"]


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def run(url: str, n_normal: int, n_severe: int, labels_path: str = "data/processed/labels.csv",
        get=fetch) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, index_col="cycle_id")
    test = labels[labels["split"] == "test"]
    normal = test[test["is_healthy"]]
    severe = test[~test["is_healthy"]]
    picked = pd.concat([normal.sample(min(n_normal, len(normal)), random_state=0),
                        severe.sample(min(n_severe, len(severe)), random_state=0)])

    rows = []
    for cid, row in picked.iterrows():
        t = get(f"{url}/cycles/{cid}/triage")
        true_sev = [c for c in COMPONENTS if row[f"faulty_{c}"]]
        diag = t.get("diagnosis")
        pred_sev = diag["severe_components"] if diag else []
        rows.append({
            "cycle": cid,
            "true_severe": ",".join(true_sev) or "-",
            "alert": t["alert"],
            "source": t["alert_source"],
            "ratio": t["anomaly"]["ratio"],
            "pred_severe": ",".join(pred_sev) or "-",
            "correct_flag": t["alert"] == bool(true_sev),
            "exact_components": sorted(true_sev) == sorted(pred_sev),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    sev = df[df["true_severe"] != "-"]
    nor = df[df["true_severe"] == "-"]
    flagged_sev = sev[sev["alert"]]
    detector_sev = sev[sev["source"].isin(["detector", "both"])]
    lines = [
        f"cycles checked: {len(df)} ({len(nor)} normal, {len(sev)} severe)",
        f"normal cycles correctly left unflagged: {int((~nor['alert']).sum())}/{len(nor)}",
        f"severe cycles alerted (detector OR classifier): {int(sev['alert'].sum())}/{len(sev)}",
        f"  of which the anomaly detector alone flagged:   {len(detector_sev)}/{len(sev)}",
        f"alerted severe cycles with the exact severe-component set right: "
        f"{int(flagged_sev['exact_components'].sum())}/{len(flagged_sev)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n-normal", type=int, default=5)
    ap.add_argument("--n-severe", type=int, default=15)
    args = ap.parse_args()
    result = run(args.url, args.n_normal, args.n_severe)
    print(result.to_string(index=False))
    print()
    print(summarize(result))