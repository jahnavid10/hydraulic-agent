"""LLM triage agent (Google ADK + Gemini). It can read model outputs and DRAFT work orders; it cannot approve them.

Run one cycle:  python -m src.agent 964
"""
import asyncio
import json
import os
import sys
import time
import uuid

from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from .env (never commit that file)

from google.adk.agents.llm_agent import Agent                 # noqa: E402
from google.adk.runners import Runner                          # noqa: E402
from google.adk.sessions import InMemorySessionService         # noqa: E402
from google.genai import types                                 # noqa: E402

from src.tools import AGENT_TOOLS                              # noqa: E402

# Comma-separated, tried in order when a model is overloaded (503) or rate-limited (429).
# Put model names that are free for your key in AI Studio into AGENT_MODEL in .env.
MODELS = [m.strip() for m in os.getenv("AGENT_MODEL", "gemini-flash-latest,gemini-flash-lite-latest").split(",") if m.strip()]
MODEL = MODELS[0]
MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "10"))
MAX_RETRIES = 4
APP_NAME = "hydraulic_triage"

INSTRUCTION = """You are a maintenance triage assistant for a hydraulic test rig. You get one cycle id.

Procedure:
1. Call score_cycle(cycle_id).
2. ALWAYS call diagnose_component(cycle_id), even if score_cycle says normal: the anomaly detector can miss faults that the classifier sees.
3. If either one signals a problem, call explain_anomaly(cycle_id) and lookup_fault_notes with a short query about the affected component.
4. For each severe component, call draft_work_order(cycle_id, component, severity, evidence). Never draft one when nothing is severe.

Rules:
- Use only numbers returned by tools. Never invent sensor values, scores or cycle facts.
- If a tool returns an error, say so and stop; do not guess.
- If a component confidence is below 0.7, say the diagnosis is uncertain.
- You can only DRAFT work orders. Never say an order was filed, approved or sent; say it is a draft waiting for human approval.
- Keep the explanation under 120 words.

End your message with exactly one line:
RESULT: {"alert": true|false, "severe_components": ["..."], "low_confidence": ["..."]}
where severe_components lists components diagnose_component marked severe (cooler, valve, pump_leak, accumulator).
Alert is true if score_cycle says anomaly OR any component is severe; it must be true whenever severe_components is not empty."""

def build_agent(model: str) -> Agent:
    return Agent(name="hydraulic_triage_agent", model=model,
                 description="Triages one hydraulic test-rig cycle.",
                 instruction=INSTRUCTION, tools=AGENT_TOOLS)


root_agent = build_agent(MODEL)   # also what `adk web` picks up


def _is_transient(exc: Exception) -> bool:
    s = str(exc)
    return any(k in s for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "504", "DEADLINE"))


async def _run_once(cycle_id: int, agent: Agent) -> dict:
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id="operator", session_id=str(uuid.uuid4()))
    msg = types.Content(role="user", parts=[types.Part(text=f"Triage cycle {cycle_id}.")])
    calls, final_text, tokens = [], "", 0
    async for event in runner.run_async(user_id="operator", session_id=session.id, new_message=msg):
        for fc in event.get_function_calls() or []:
            calls.append({"name": fc.name, "args": dict(fc.args or {})})
        if event.usage_metadata and event.usage_metadata.total_token_count:
            tokens += event.usage_metadata.total_token_count
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)
        if len(calls) > MAX_TOOL_CALLS:
            raise RuntimeError(f"tool-call cap ({MAX_TOOL_CALLS}) exceeded")
    return {"final_text": final_text, "tool_calls": calls, "tokens": tokens}


async def run_agent_async(cycle_id: int) -> dict:
    """Never raises: failures come back as {'error': ...} so one bad case cannot stop an evaluation.
    Tries each model in MODELS; per model, retries transient errors with backoff."""
    start, last_err = time.time(), None
    for model in MODELS:
        agent = build_agent(model)
        for attempt in range(MAX_RETRIES):
            try:
                out = await _run_once(cycle_id, agent)
                out.update(error=None, latency_s=round(time.time() - start, 2), cycle_id=cycle_id, model=model)
                return out
            except Exception as exc:  # noqa: BLE001
                last_err = f"{model}: {type(exc).__name__}: {str(exc)[:300]}"
                if not _is_transient(exc):
                    return {"final_text": "", "tool_calls": [], "tokens": 0, "cycle_id": cycle_id,
                            "latency_s": round(time.time() - start, 2), "error": last_err}
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 2))   # 4, 8, 16 s
    return {"final_text": "", "tool_calls": [], "tokens": 0, "cycle_id": cycle_id,
            "latency_s": round(time.time() - start, 2), "error": last_err}

def run_agent(cycle_id: int) -> dict:
    return asyncio.run(run_agent_async(cycle_id))


if __name__ == "__main__":
    out = run_agent(int(sys.argv[1]))
    print(json.dumps(out, indent=2, default=str))