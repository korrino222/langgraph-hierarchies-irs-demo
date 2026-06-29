"""Deterministic expense parsing and formatting tools."""

from __future__ import annotations

import json
import re
from typing import Any

_AMOUNT_RE = re.compile(r"\$?\s*([\d,]+\.\d{2})")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _error(reason: str) -> dict[str, Any]:
    return {"status": "error", "stage": "extraction", "reason": reason}


def parse_expense(raw_text: str) -> dict[str, Any]:
    """Parse a single-line expense note into structured fields."""
    text = raw_text.strip()
    if not text:
        return _error("empty input")

    amount_match = _AMOUNT_RE.search(text)
    date_match = _DATE_RE.search(text)

    missing: list[str] = []
    if not amount_match:
        missing.append("amount")
    if not date_match:
        missing.append("date")

    parts = [part.strip() for part in text.split(",")]
    vendor = parts[1] if len(parts) > 1 else ""
    category = parts[-1] if len(parts) > 3 else ""

    if not vendor:
        missing.append("vendor")
    if not category or category == vendor:
        missing.append("category")

    if missing:
        return _error(f"missing: {', '.join(missing)}")

    amount = float(amount_match.group(1).replace(",", ""))
    return {
        "status": "ok",
        "vendor": vendor,
        "amount": amount,
        "date": date_match.group(1),
        "category": category,
    }


def format_expense(artifact: dict[str, Any] | str) -> dict[str, Any]:
    """Format a structured extraction artifact into a printable card."""
    if isinstance(artifact, str):
        try:
            artifact = json.loads(artifact)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "stage": "formatting",
                "reason": "invalid artifact JSON",
            }

    if artifact.get("status") != "ok":
        return {
            "status": "error",
            "stage": "formatting",
            "reason": "cannot format non-ok extraction artifact",
        }

    card = (
        f"{artifact['vendor']} | ${artifact['amount']:.2f} | "
        f"{artifact['date']} | {artifact['category']}"
    )
    return {"status": "ok", "card": card}


def artifact_from_state(state: dict) -> dict[str, Any]:
    """Load pipeline_artifact JSON from graph state."""
    raw = state.get("pipeline_artifact") or state.get("current_agent_report") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "error", "stage": "unknown", "reason": "invalid JSON artifact"}


def extraction_status(artifact_json: str) -> str:
    """Return artifact status string for gating."""
    try:
        return json.loads(artifact_json).get("status", "error")
    except json.JSONDecodeError:
        return "error"
