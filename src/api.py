"""FastAPI service exposing the detector, classifier, tools and the work-order approval gate.

Run locally:   uvicorn src.api:app --reload        then open http://127.0.0.1:8000/docs
"""
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src import tools, workorders
from src.config import ART_DIR, PROC_DIR

VERSION = "0.1.0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hydraulic-api")


# ---------- response models ----------
class ScoreResponse(BaseModel):
    cycle_id: int
    score: float
    threshold: float
    ratio: float
    is_anomaly: bool


class SensorShare(BaseModel):
    sensor: str
    meaning: str
    share_of_error: float
    most_deviating_statistic: str


class ExplainResponse(BaseModel):
    cycle_id: int
    top_sensors: list[SensorShare]


class ComponentState(BaseModel):
    state: int
    meaning: str
    confidence: float
    severe: bool


class DiagnoseResponse(BaseModel):
    cycle_id: int
    components: dict[str, ComponentState]
    severe_components: list[str]


class TriageResponse(BaseModel):
    cycle_id: int
    anomaly: ScoreResponse
    top_sensors: list[SensorShare] = []
    diagnosis: DiagnoseResponse | None = None
    recommendation: str


class WorkOrder(BaseModel):
    id: str
    cycle_id: int
    component: str
    severity: str
    priority: str
    evidence: str
    status: str
    created_at: str
    decided_at: str | None = None


# ---------- tiny in-memory request metrics (read by the monitoring dashboard) ----------
class _Metrics:
    def __init__(self):
        self.by_path = defaultdict(int)
        self.latencies_ms = deque(maxlen=500)
        self.errors = 0

    def record(self, path: str, status: int, ms: float) -> None:
        self.by_path[path] += 1
        self.latencies_ms.append(ms)
        if status >= 500:
            self.errors += 1

    def snapshot(self) -> dict:
        lat = sorted(self.latencies_ms)
        pick = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: None)
        return {"requests_total": sum(self.by_path.values()), "errors_5xx": self.errors,
                "by_path": dict(self.by_path),
                "latency_ms": {"p50": pick(0.50), "p95": pick(0.95), "n": len(lat)}}


METRICS = _Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        tools._bundle()  # warm up: load models once at startup
        log.info("models loaded")
    except Exception as exc:  # noqa: BLE001
        log.warning("models not loaded at startup: %s", exc)
    yield


app = FastAPI(title="Hydraulic Maintenance Agent API", version=VERSION, lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    METRICS.record(path, response.status_code, ms)
    log.info("%s %s -> %s in %.0f ms", request.method, request.url.path, response.status_code, ms)
    return response


def _unwrap(result: dict) -> dict:
    """Tools report problems as {"error": ...}; turn that into a clean HTTP 422."""
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ---------- system ----------
@app.get("/health", tags=["system"])
def health():
    needed = [ART_DIR / f for f in ("ae.pt", "scaler.joblib", "threshold.json", "clf.joblib")]
    needed.append(PROC_DIR / "features.csv")
    missing = [p.name for p in needed if not p.exists()]
    body = {"status": "ok" if not missing else "degraded", "missing_files": missing, "version": VERSION}
    return JSONResponse(body, status_code=200 if not missing else 503)


@app.get("/metrics", tags=["system"])
def metrics():
    return METRICS.snapshot()


# ---------- model endpoints ----------
@app.get("/cycles/{cycle_id}/score", response_model=ScoreResponse, tags=["model"])
def score(cycle_id: int):
    return _unwrap(tools.score_cycle(cycle_id))


@app.get("/cycles/{cycle_id}/explain", response_model=ExplainResponse, tags=["model"])
def explain(cycle_id: int, top_k: int = Query(5, ge=1, le=10)):
    return _unwrap(tools.explain_anomaly(cycle_id, top_k))


@app.get("/cycles/{cycle_id}/diagnose", response_model=DiagnoseResponse, tags=["model"])
def diagnose(cycle_id: int):
    return _unwrap(tools.diagnose_component(cycle_id))


@app.get("/cycles/{cycle_id}/triage", response_model=TriageResponse, tags=["model"])
def triage(cycle_id: int):
    """Fixed, rule-based pipeline (no LLM): score, then explain and diagnose if anomalous.

    Doubles as the baseline the LLM agent is compared against in the evaluation.
    """
    s = _unwrap(tools.score_cycle(cycle_id))
    if not s["is_anomaly"]:
        return {"cycle_id": s["cycle_id"], "anomaly": s,
                "recommendation": "No severe fault detected. No action needed."}
    ex = _unwrap(tools.explain_anomaly(cycle_id))
    dg = _unwrap(tools.diagnose_component(cycle_id))
    if dg["severe_components"]:
        rec = ("Severe degradation suspected in: " + ", ".join(dg["severe_components"])
               + ". Review the evidence and draft a work order.")
    else:
        rec = "Anomalous score but no component classified as severe. Inspect manually."
    return {"cycle_id": s["cycle_id"], "anomaly": s, "top_sensors": ex["top_sensors"],
            "diagnosis": dg, "recommendation": rec}


# ---------- work orders (human approval gate) ----------
@app.get("/workorders", response_model=list[WorkOrder], tags=["workorders"])
def list_workorders(status: str | None = Query(None, pattern="^(DRAFT|APPROVED|REJECTED)$")):
    return workorders.list_orders(status)


def _decide(order_id: str, status: str) -> dict:
    try:
        return workorders.set_status(order_id, status)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"work order {order_id} not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/workorders/{order_id}/approve", response_model=WorkOrder, tags=["workorders"])
def approve(order_id: str):
    return _decide(order_id, "APPROVED")


@app.post("/workorders/{order_id}/reject", response_model=WorkOrder, tags=["workorders"])
def reject(order_id: str):
    return _decide(order_id, "REJECTED")