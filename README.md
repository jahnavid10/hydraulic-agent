# Hydraulic Maintenance Triage Agent

[![CI](https://github.com/jahnavid10/hydraulic-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jahnavid10/hydraulic-agent/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License: dataset CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)

**Live demo (dashboard):** https://hydraulic-agent.streamlit.app/
(deployed from the [`demo` branch](https://github.com/jahnavid10/hydraulic-agent/tree/demo); free hosting, may take a moment to wake up)

A condition-monitoring system for a hydraulic test rig. Given one 60-second machine cycle it
detects whether the cycle looks abnormal, diagnoses which component is severely degraded, and uses an
LLM agent to explain the findings and **draft** a maintenance work order.
**A human approves or rejects every order. The agent cannot approve anything.**

Built from scratch on public data (no employer code), using free tools only.

## Contents

1. [At a glance](#at-a-glance)
2. [The problem](#the-problem)
3. [Dataset](#dataset)
4. [How it works](#how-it-works)
5. [Example run](#example-run)
6. [Dashboard](#dashboard)
7. [API](#api)
8. [Results](#results)
9. [Agent evaluation](#agent-evaluation)
10. [Design decisions](#design-decisions)
11. [Engineering practices](#engineering-practices)
12. [Limitations](#limitations)
13. [Possible next steps](#possible-next-steps)
14. [Run it locally](#run-it-locally)
15. [Project structure](#project-structure)
16. [Tech stack and credits](#tech-stack-and-credits)

## At a glance

| | |
|---|---|
| **Input** | One test-rig cycle (60 s, 17 sensors) |
| **Output** | Alert yes/no, faulty components with confidence, sensor-level explanation, draft work order |
| **Models** | PyTorch autoencoder (anomaly detection) + random forest (component diagnosis) |
| **Agent** | Google ADK + Gemini (free tier), five tools, draft-only permissions |
| **Serving** | FastAPI with typed responses, Streamlit dashboards, Docker Compose |
| **Quality** | pytest suite, GitHub Actions CI, MLflow tracking, PSI drift monitoring, agent evaluation |
| **Safety** | Human approval gate; the agent can create drafts only |

## The problem

Hydraulic systems fail gradually: a cooler loses efficiency, a valve starts to lag, a pump leaks
internally, an accumulator loses pre-charge. Operators see raw sensor streams, not a diagnosis.
This project turns a cycle of sensor data into a triage decision an operator can act on:

- **Is something wrong?** An anomaly detector compares the cycle with what normal cycles look like.
- **What is wrong?** A classifier estimates the state of four components.
- **What should I do?** An LLM agent explains the result in plain language, looks up fault guidance,
  and drafts a work order for a human to review.

It doubles as a worked example of putting models behind real engineering: an API, dashboards,
containers, tests, CI, drift monitoring, an agent with guard rails, and an honest evaluation.

## Dataset

UCI [Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
(Helwig, Pignanelli, Schutze, 2015), licensed CC BY 4.0.

| Property | Value |
|---|---|
| Cycles | 2205, each 60 seconds long |
| Sensors | 17 (pressures, flow, temperatures, vibration, motor power, cooling efficiency and power, efficiency factor) |
| Labels | Condition of four components per cycle: cooler, valve, internal pump leakage, hydraulic accumulator |

## How it works

```
cycle id -> [features] -> autoencoder (anomaly score) + random forest (component states)
                 |
        5 tools (score, explain, diagnose, look up notes, draft work order)
                 |
   FastAPI  <->  LLM agent (Google ADK, Gemini)      Streamlit: operator + monitoring
                 |
   work-order outbox (DRAFT -> APPROVED / REJECTED, humans only)
```

### 1. Feature engineering
Each cycle becomes 85 numbers: 17 sensors x 5 statistics (mean, standard deviation, minimum, maximum, slope).

### 2. Anomaly detector
A PyTorch autoencoder (85-64-32-8-32-64-85) is trained **only on normal cycles**. A cycle's score is its
mean squared reconstruction error per feature. The alert threshold is the 95th percentile of validation
errors on normal cycles (0.1459 in this build). Faults the model never saw reconstruct badly, so the
score rises.

### 3. Component classifier
A random forest predicts the state of each of the four components and a confidence value.
Diagnoses below 0.7 confidence are marked uncertain.

### 4. Combined triage rule
Alert if the detector fires **or** the classifier finds a severe component. Running both matters: on the
test split the detector alone caught 364 of 400 severe cycles, and adding the classifier caught all 400.

### 5. LLM agent
Google ADK with Gemini and five tools. On every cycle it must call the scoring and diagnosis tools, may
use only numbers the tools return, and can only create draft work orders.

| Tool | What it does |
|---|---|
| `score_cycle` | Anomaly score, threshold and ratio for a cycle |
| `explain_anomaly` | Sensors contributing most to the anomaly score |
| `diagnose_component` | State and confidence for each component, and which are severe |
| `lookup_fault_notes` | Retrieves matching guidance from a notes file (TF-IDF search) |
| `draft_work_order` | Saves a DRAFT work order (cannot approve it) |

### 6. Human in the loop
Approving or rejecting a draft is possible only through the API or dashboard, never through the agent.

## Example run

Real agent output (`python -m src.agent 18`) on a cycle with a failing cooler:

> Cycle 18 is abnormal with a score of 3.996 (ratio 27.4), primarily driven by temperature sensors
> (TS2, TS3, TS1). Component diagnosis identifies the cooler as severe (state 3: close to total failure)
> with 0.92 confidence, while all other components are healthy. A draft work order has been created for
> the cooler and is waiting for human approval.

Tool calls: `score_cycle`, `diagnose_component`, `explain_anomaly`, `lookup_fault_notes`, `draft_work_order`.

On a healthy cycle (932: score 0.1007 against threshold 0.1459) the agent calls the scoring and
diagnosis tools, reports no severe components, and drafts nothing.

## Dashboard

Two views, both talking to the FastAPI service only:

- **Operator:** enter a cycle id, see the triage result, draft a work order, approve or reject drafts.
- **Monitoring:** API health and latency, anomaly score across cycles against the threshold,
  and per-sensor feature drift (PSI).

The live demo runs the rule-based triage and the approval workflow. The LLM agent is **not** exposed
publicly, to protect the free-tier API key; see the agent evaluation below.

## API

FastAPI with typed (Pydantic) responses and interactive docs at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /metrics` | Service health (503 if model files are missing) and request metrics |
| `GET /cycles/{id}/score`, `/explain`, `/diagnose` | The individual model outputs |
| `GET /cycles/{id}/triage` | Combined rule-based triage (also the baseline for the agent evaluation) |
| `POST /agent/diagnose` | LLM agent triage for one cycle (needs a Gemini API key) |
| `GET /monitoring/scores`, `GET /monitoring/drift` | Score series and PSI drift report |
| `GET/POST /workorders`, `POST /workorders/{id}/approve`, `/reject` | Work-order workflow |

Invalid cycle ids (outside 0 to 2204) return a 422 validation error.

## Results

Held-out test split, 441 cycles. Every number below comes from data the models never trained on,
and the threshold was not tuned on it.

| Component | Result |
|---|---|
| Anomaly detector | AUC 0.982, recall 0.910, precision 0.995, false-positive rate 0.049 |
| Classifier accuracy | cooler 0.998, valve 0.961, pump leak 0.998, accumulator 0.995 |
| Combined triage | 400/400 severe cycles alerted; 39/41 normal cycles correct; exact severe-component set 393/400 |

Both false alarms on normal cycles came from the detector threshold.

## Agent evaluation

**Method:** 30 random held-out test cycles (10 normal, 20 severe, fixed seed) were sent to the agent and to
the rule-based triage baseline, and both were scored against the true labels. Model:
`gemini-3.5-flash-lite` (free tier).

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

**Known hard cases** (4 cycles chosen because the baseline got them wrong, so not representative):
the agent reproduced the baseline on all four.

| Cycle | Truth | Agent and baseline | What happened |
|---|---|---|---|
| 964 | pump leak | pump leak + valve | Valve over-called at 0.54 confidence; the agent flagged it as uncertain but still drafted it |
| 1290 | accumulator | accumulator | Detector misses it; the classifier catches it because the agent always calls diagnosis |
| 1923 | none | alert, no components | False alarm from the detector threshold; no work order created |
| 1749 | none | alert, no components | False alarm from the detector threshold; no work order created |

**What this shows:** the agent is faithful to the tools. It never invented values, never claimed an order
was filed, and respected the approval gate. It adds explanations, a notes lookup and drafted orders, but
**no detection accuracy beyond the underlying models**, because it relays their outputs.

**Agent failure case:** on cycle 627, where all four components are severe, the agent reported all four
correctly but drafted only three work orders (7 tool calls). The final answer and the actions disagreed,
so for completeness-critical actions the system should verify the drafts against the diagnosis in code
instead of trusting the model.

## Design decisions

| Decision | Why |
|---|---|
| Redefine "normal" as "no severe component" | Only 21 of 2205 cycles have every component at its optimal value, too few to train on. This gives 208 normal cycles (125 train, 42 validation, 41 test). |
| Train the autoencoder on normal cycles only | It can then flag faults it was never shown. |
| Run the detector and the classifier on every cycle | Each catches faults the other misses. |
| Keep approval outside the agent | Guard rails live in the tool permissions and the API, not only in the prompt, and the evaluation measures whether the agent respects them. |
| Derive the alert flag in code | The model once wrote "alert: false" while listing severe components, so the alert is forced on whenever any are listed. |
| Retry and fall back across models | Free-tier models return overload (503) and rate-limit (429) errors. |
| Do not tune on the test set | Test numbers stay honest, and the threshold trade-off is documented instead. |

## Engineering practices

- **Tests:** 36 pytest tests covering tools, API, monitoring, answer parsing and evaluation scoring.
  They use a synthetic dataset, so they need no trained models or LLM key.
- **CI:** GitHub Actions runs the tests and builds both Docker images.
- **Containers:** `docker-compose` runs the API and the dashboard, with models and data mounted read-only
  and secrets kept out of the image.
- **Tracking:** MLflow (SQLite backend) logs training and evaluation runs.
- **Monitoring:** PSI-based feature drift and anomaly-score tracking.
- **Reproducibility:** fixed seed (42), stratified 60/20/20 split, saved evaluation reports.

## Limitations

Read these before trusting any number.

- **Optimistic split:** cycles are split randomly, and neighbouring cycles are near-duplicates.
  A split that holds out whole fault combinations would give lower numbers.
- **Relaxed "normal" definition:** mild wear is treated as normal.
- **Threshold trade-off:** the 95th-percentile threshold trades recall against false alarms.
- **Small agent evaluation:** 30 cases supports "likely above about 88% correct", not "100% accurate".
- **Fault notes are illustrative**, not manufacturer guidance.
- **Free-tier LLM limits:** 15 requests per minute.
- **Hosted demo:** data resets when the app restarts, and work orders there are shared by all visitors.

## Possible next steps

- Verify in code that drafted work orders match the diagnosed severe components.
- Evaluate on a stricter split that holds out whole fault combinations.
- Tune the threshold on validation data and report the precision/recall trade-off.
- Add LLM tracing and observability.

## Run it locally

```bash
git clone https://github.com/jahnavid10/hydraulic-agent.git
cd hydraulic-agent
pip install -r requirements.txt
python -m src.download_data && python -m src.build_dataset
python -m src.train_ae && python -m src.train_clf
pytest -q
```

Run the API and dashboard with Docker (API on :8000/docs, dashboard on :8501):

```bash
docker compose up --build
```

Run the agent: create a `.env` file (never commit it) with

```
GOOGLE_API_KEY=your-key
AGENT_MODEL=gemini-3.5-flash-lite,gemini-3.1-flash-lite
```

then run `python -m src.agent 18`, or evaluate it with
`python -m eval.run_eval --n-normal 10 --n-severe 20 --sleep 20`.

## Project structure

```
src/         features, models, tools, API, agent, monitoring, work orders, evaluation helpers
dashboard/   Streamlit app (Operator and Monitoring views)
eval/        agent-vs-baseline evaluation script
tests/       pytest suite with a synthetic dataset
notes/       fault guidance used by the lookup tool
reports/     saved evaluation results
.github/     CI workflow
```

## Tech stack and credits

Python, PyTorch, scikit-learn, pandas, FastAPI, Pydantic, Streamlit, Docker, GitHub Actions, MLflow,
pytest, Google ADK, Gemini API.

Data: UCI "Condition monitoring of hydraulic systems" (Helwig, Pignanelli, Schutze, 2015), CC BY 4.0.