"""Small helpers for reading the agent's answer. No ADK or LLM imports, so they are easy to test."""
import json
import re

from src.config import COMPONENTS

_FILED = re.compile(
    r"\b(?:has been|have been|was|were|is now|are now|successfully)\s+(?:filed|approved|submitted|sent)\b"
    r"|\bI(?:'ve| have)?\s+(?:filed|approved|submitted)\b",
    re.IGNORECASE,
)


def parse_result(text: str) -> dict | None:
    """Return the JSON object after the last 'RESULT:' marker in the agent's message, or None."""
    if not text:
        return None
    idx = text.rfind("RESULT:")
    if idx == -1:
        return None
    tail = text[idx + len("RESULT:"):]
    start = tail.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(tail)):
        if tail[i] == "{":
            depth += 1
        elif tail[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(tail[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def normalize_result(obj: dict | None) -> dict | None:
    """Coerce the parsed RESULT into {'alert': bool, 'severe_components': sorted valid names}."""
    if obj is None or "alert" not in obj:
        return None
    severe = obj.get("severe_components") or []
    if not isinstance(severe, list):
        return None
    comps = sorted({c for c in severe if c in COMPONENTS})
    # Same rule as the rule-based triage: alert = detector OR classifier. The model sometimes writes
    # alert=false while listing severe components, so an alert is forced whenever any are listed.
    return {"alert": bool(obj["alert"]) or bool(comps),
            "severe_components": comps,
            "low_confidence": obj.get("low_confidence") or []}


def claims_filed(text: str) -> bool:
    return bool(_FILED.search(text or ""))