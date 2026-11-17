# Hydraulic Maintenance Triage Agent

An end-to-end ML + LLM-agent system on the public UCI hydraulic-systems dataset. It scores
test-rig cycles with an anomaly detector, diagnoses component faults with a classifier,
and uses an LLM agent (Google ADK + Gemini) to explain findings and **draft** work orders.
A human approves or rejects every order. The agent cannot approve anything.

Built from scratch on public data (no employer code), using free tools only.

## Architecture

```
cycle id -> [features] -> autoencoder (anomaly score) + random forest (component states)
                 |
        5 tools (score, explain, diagnose, look up notes, draft work order)
                 |
   FastAPI  <->  LLM agent (Google ADK, Gemini)      Streamlit: operator + monitoring
                 |
   work-order outbox (DRAFT -> APPROVED / REJECTED, humans only)
```

- **Features:** 17 sensors x 5 statistics = 85 features per 60 s cycle.
- **Anomaly detector:** PyTorch autoencoder trained on normal cycles only; threshold = 95th
  percentile of validation reconstruction error.
- **Classifier:** random forest for the four component states.
- **Triage rule:** alert = detector OR classifier flags a severe component.
- **Engineering:** FastAPI (typed responses, health, metrics), Streamlit dashboards,
  Docker + docker-compose, MLflow tracking, PSI drift monitoring, pytest, GitHub Actions CI.

## Results (held-out test split, 441 cycles)

| Component | Result |
|---|---|
| Anomaly detector | AUC 0.982, recall 0.910, precision 0.995, false-positive rate 0.049 |
| Classifier accuracy | cooler 0.998, valve 0.961, pump leak 0.998, accumulator 0.995 |
| Combined triage | 400/400 severe cycles alerted; 39/41 normal cycles correct; exact severe-component set 393/400 |

### Agent evaluation

30 random held-out test cycles (10 normal, 20 severe, fixed seed) against the rule-based triage baseline.
Model: `gemini-3.5-flash-lite` (free tier).

| Measure | Result |
|---|---|
| Alert correct: baseline / agent | 30/30 / 30/30 |
| Exact severe components, severe cycles: baseline / agent | 20/20 / 20/20 |
| Agent called both score and diagnose tools | 30/30 |
| No draft on normal cycles | 10/10 |
| Drafts match true severe components | 19/20 |
| Answers claiming an order was filed or approved | 0/30 |
| Latency p50 / p95 | 3.5 s / 13.0 s |
| Mean tool calls / tokens per case | 4.5 / 4361 |

**Known hard cases** (4 cycles, chosen because the baseline got them wrong, so not representative):
the agent reproduced the baseline on all four. It recovered cycle 1290, where the detector misses
the fault, by always calling the classifier. It over-called a valve fault on cycle 964
(classifier confidence 0.54), flagged it as uncertain, but still drafted it. It raised alerts on two
healthy cycles (1923, 1749) from the detector threshold and created no work orders for them.

**What this shows:** the agent is faithful to the tools (it never invents values, never claims
an order was filed, and respects the approve gate). It adds explanations, a notes lookup and drafted
orders, but **no detection accuracy beyond the underlying models**, because it relays their outputs.

**Agent failure case:** on cycle 627, where all four components are severe, the agent reported all four correctly but drafted only three work orders (7 tool calls). The final answer and the actions disagreed, so for completeness-critical actions the system should verify the drafts against the diagnosis in code instead of trusting the model..

## Limitations (read before trusting any number)

- **Optimistic split:** cycles are split randomly, and neighbouring cycles are near-duplicates.
  A split that holds out whole fault combinations would give lower numbers.
- **"Normal" is a relaxed definition:** only 21 of 2205 cycles are strictly optimal, so normal means
  "no component in a severe state" (208 cycles). Mild wear is treated as normal.
- **Threshold trade-off:** the 95th-percentile threshold trades recall against false alarms.
  It was not tuned on the test set.
- **Small agent evaluation:** 30 cases. It supports "likely above ~88% correct", not "100% accurate".
- **Fault notes are illustrative**, not manufacturer guidance.
- **Free-tier LLM limits:** 15 requests per minute; the client retries with backoff and falls back to a second model.

## Run it

```
pip install -r requirements.txt
python -m src.download_data && python -m src.build_dataset
python -m src.train_ae && python -m src.train_clf
pytest -q
docker compose up --build        # API :8000/docs, dashboard :8501
```

Agent: put `GOOGLE_API_KEY=...` and `AGENT_MODEL=gemini-3.5-flash-lite,gemini-3.1-flash-lite` in `.env`
(never commit it), then `python -m src.agent 18` or
`python -m eval.run_eval --n-normal 10 --n-severe 20 --sleep 20`.

## Data

UCI "Condition monitoring of hydraulic systems" (Helwig, Pignanelli, Schutze, 2015), CC BY 4.0.