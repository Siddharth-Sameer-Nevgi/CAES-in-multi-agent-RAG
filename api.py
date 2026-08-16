"""Minimal FastAPI layer.

    uvicorn api:app --port 8000
    curl -s localhost:8000/query -H 'content-type: application/json' \
         -d '{"question":"Who directed Inception?"}'

Its purpose is to substantiate the protocol's API layer for the paper. There is
deliberately no auth, no rate limiting, and no deployment tooling here.
"""
from __future__ import annotations

import logging
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config

log = logging.getLogger("caes.api")

app = FastAPI(title="CAES-RAG", version="1.0",
              description="Cost-Aware Evidence Sufficiency retrieval gate")

# Per-iteration cost is metered by diffing process-wide counters in bedrock.py,
# so concurrent graph runs would interleave and mis-attribute cost to each
# other. One lock keeps queries serialised. Fine for a demonstration endpoint;
# a real deployment would scope the meter per request instead.
_RUN_LOCK = threading.Lock()
_POLICY = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    policy: str = Field("caes", pattern="^(caes|fixed|oneshot|threshold)$")


class QueryResponse(BaseModel):
    answer: str
    iterations: int
    cost_usd: float
    latency_ms: float
    stop_reason: str
    final_coverage: float


def get_policy(name: str):
    global _POLICY
    from policies import build_policy
    if _POLICY is None or getattr(_POLICY, "name", None) != name:
        _POLICY = build_policy(name)
    return _POLICY


@app.get("/health")
def health() -> dict:
    from costs import TRACKER
    return {
        "status": "ok",
        "lambda": config.LAMBDA,
        "cumulative_spend_usd": round(TRACKER.cumulative(), 4),
        "remaining_budget_usd": round(TRACKER.remaining(), 4),
        "index_present": config.INDEX_PATH.exists(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    from costs import BudgetExceeded
    from graph import run_query, state_summary

    try:
        policy = get_policy(req.policy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with _RUN_LOCK:
            # honor_confidence is on here: this path optimises for latency, and
            # unlike the experiments it makes no cross-policy claim.
            final = run_query(req.question, policy, honor_confidence=True)
    except BudgetExceeded as exc:
        raise HTTPException(status_code=503, detail=f"Budget guard: {exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Index not built. Run `python ingest.py`. ({exc})") from exc

    s = state_summary(final)
    return QueryResponse(
        answer=s["answer"],
        iterations=s["iterations_used"],
        cost_usd=s["total_usd"],
        latency_ms=s["total_latency_ms"],
        stop_reason=s["stop_reason"],
        final_coverage=s["final_coverage"],
    )
