"""Compare the LLM agent with the rule-based /triage baseline on HELD-OUT test cycles.

    python -m eval.run_eval --n-normal 10 --n-severe 20 --sleep 6
Writes reports/agent_eval.json and reports/agent_eval.md
"""
import argparse
import asyncio
import json
import random

import pandas as pd

from src.config import COMPONENTS, PROC_DIR, ROOT
from src.eval_metrics import score_case, summarize, to_markdown


def pick_cases(n_normal: int, n_severe: int, seed: int = 7, extra: tuple = ()) -> list[tuple[int, list[str]]]:
    labels = pd.read_csv(PROC_DIR / "labels.csv", index_col="cycle_id")
    test = labels[labels["split"] == "test"]
    rng = random.Random(seed)
    groups = ((test[test["is_healthy"]], n_normal), (test[~test["is_healthy"]], n_severe))
    out = []
    for df, n in groups:
        ids = list(df.index)
        rng.shuffle(ids)
        for i in ids[:n]:
            out.append((int(i), [c for c in COMPONENTS if bool(df.loc[i, f"faulty_{c}"])]))
    have = {c for c, _ in out}
    for i in extra:                       # known hard cases, added on top of the random ones
        if i not in have:
            if i not in test.index:
                raise ValueError(f"cycle {i} is not in the test split")
            out.append((int(i), [c for c in COMPONENTS if bool(test.loc[i, f"faulty_{c}"])]))
    return out


async def main(n_normal: int, n_severe: int, sleep: float, extra: tuple = ()):
    from src import api
    from src.agent import run_agent_async   # imported here so --help works without ADK
    cases, rows = pick_cases(n_normal, n_severe, extra=extra), []
    for k, (cid, truth) in enumerate(cases, 1):
        baseline = api.triage(cid)
        baseline = baseline.model_dump() if hasattr(baseline, "model_dump") else baseline
        agent = await run_agent_async(cid)
        row = score_case(cid, truth, baseline, agent)
        rows.append(row)
        print(f"[{k}/{len(cases)}] cycle {cid} truth={truth} agent={row['agent_set']} "
              f"err={row['agent_error']} {row['latency_s']}s")
        await asyncio.sleep(sleep)   # stay under free-tier requests/minute
    summary = summarize(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "agent_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    (ROOT / "reports" / "agent_eval.md").write_text(to_markdown(summary))
    print("\n" + to_markdown(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-normal", type=int, default=10)
    ap.add_argument("--n-severe", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=20.0)
    ap.add_argument("--extra", type=str, default="", help="comma-separated TEST-split cycle ids to always include")
    a = ap.parse_args()
    extra = tuple(int(x) for x in a.extra.split(",") if x.strip())
    asyncio.run(main(a.n_normal, a.n_severe, a.sleep, extra))