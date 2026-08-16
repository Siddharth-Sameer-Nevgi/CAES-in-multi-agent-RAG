"""Evidence-sufficiency verifier — the dQ signal source.

Fires every iteration, so it dominates spend. Two consequences shape this file:
  * evidence is truncated to ~150 tokens per chunk before it is sent, which is
    the main lever on gate overhead;
  * parsing is defensive, because one unparsed response is one corrupted point
    on the coverage curve.

Parse ladder: strip fences -> json.loads -> retry once with a repair
instruction -> fall back to the previous coverage and log a warning.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import config
from agents.prompts import (
    VERIFIER_PROMPT,
    VERIFIER_REPAIR_PROMPT,
    VERIFIER_SYSTEM,
)

log = logging.getLogger("caes.verifier")

MAX_VERIFY_TOKENS = 200

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verification:
    coverage: float      # 0.0-1.0
    missing: str
    confident: bool
    raw: str = ""
    parse_failed: bool = False


def truncate_evidence(chunks: list, max_chars: int = config.VERIFIER_CHUNK_CHARS) -> str:
    """First ~150 tokens per chunk. This directly controls gate overhead."""
    parts = []
    for c in chunks:
        text = getattr(c, "text", None) or c.get("text", "")
        text = text.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        parts.append(text)
    return "\n| ".join(parts) if parts else "(no evidence retrieved)"


def _extract_json(text: str) -> dict | None:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Model wrapped the object in prose: grab the outermost braces.
    m = _OBJ_RE.search(cleaned)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _coerce(obj: dict, raw: str) -> Verification | None:
    if "coverage" not in obj:
        return None
    try:
        coverage = float(obj["coverage"])
    except (TypeError, ValueError):
        return None
    coverage = min(1.0, max(0.0, coverage))
    missing = str(obj.get("missing", "") or "").strip()
    confident = bool(obj.get("confident", coverage >= 0.85))
    return Verification(coverage=coverage, missing=missing,
                        confident=confident, raw=raw)


def verify(
    question: str,
    evidence: list,
    *,
    query_id: str = "",
    iteration: int = 0,
    policy: str = "",
    previous_coverage: float = 0.0,
) -> Verification:
    """Score how well `evidence` covers `question`."""
    import bedrock

    prompt = VERIFIER_PROMPT.format(
        question=question, evidence=truncate_evidence(evidence))
    resp = bedrock.invoke_llm(
        prompt, call_type="verify", query_id=query_id, iteration=iteration,
        max_tokens=MAX_VERIFY_TOKENS, system=VERIFIER_SYSTEM, policy=policy,
    )

    obj = _extract_json(resp.text)
    if obj is not None:
        v = _coerce(obj, resp.text)
        if v is not None:
            return v

    # One repair attempt.
    log.warning("[%s it%d] verifier JSON unparseable; retrying with repair "
                "instruction", query_id, iteration)
    repair = bedrock.invoke_llm(
        VERIFIER_REPAIR_PROMPT.format(bad_output=resp.text[:500]),
        call_type="verify", query_id=query_id, iteration=iteration,
        max_tokens=MAX_VERIFY_TOKENS, system=VERIFIER_SYSTEM, policy=policy,
    )
    obj = _extract_json(repair.text)
    if obj is not None:
        v = _coerce(obj, repair.text)
        if v is not None:
            return v

    # Give up: hold coverage flat so the gate sees zero gain and stops, rather
    # than reading a parse failure as evidence of progress.
    log.warning("[%s it%d] verifier repair also failed; holding coverage at "
                "%.2f", query_id, iteration, previous_coverage)
    return Verification(coverage=previous_coverage,
                        missing="verifier parse failure",
                        confident=False, raw=repair.text, parse_failed=True)
