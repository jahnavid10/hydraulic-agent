"""Work-order storage with a human approval gate.

The agent can only call `create_draft` (via the draft_work_order tool). Moving a draft to
APPROVED or REJECTED happens through `set_status`, which only the API/dashboard calls.
Each order is one JSON file in outbox/.
"""
import json
from datetime import datetime, timezone

from src.config import ROOT

OUTBOX = ROOT / "outbox"
STATUSES = {"DRAFT", "APPROVED", "REJECTED"}


def _path(order_id: str):
    return OUTBOX / f"{order_id}.json"


def _save(order: dict) -> dict:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    _path(order["id"]).write_text(json.dumps(order, indent=2))
    return order


def create_draft(cycle_id: int, component: str, severity: str, evidence: str) -> dict:
    order = {
        "id": f"WO-{cycle_id:04d}-{component}",
        "cycle_id": cycle_id,
        "component": component,
        "severity": severity,
        "priority": "high" if severity == "severe" else "normal",
        "evidence": evidence,
        "status": "DRAFT",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return _save(order)


def get_order(order_id: str) -> dict | None:
    p = _path(order_id)
    return json.loads(p.read_text()) if p.exists() else None


def list_orders(status: str | None = None) -> list[dict]:
    if not OUTBOX.exists():
        return []
    orders = [json.loads(p.read_text()) for p in sorted(OUTBOX.glob("WO-*.json"))]
    return [o for o in orders if status is None or o["status"] == status]


def set_status(order_id: str, status: str) -> dict:
    """DRAFT -> APPROVED or REJECTED. Anything else is refused."""
    if status not in {"APPROVED", "REJECTED"}:
        raise ValueError("status must be APPROVED or REJECTED")
    order = get_order(order_id)
    if order is None:
        raise KeyError(order_id)
    if order["status"] != "DRAFT":
        raise ValueError(f"order is already {order['status']}")
    order["status"] = status
    order["decided_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _save(order)