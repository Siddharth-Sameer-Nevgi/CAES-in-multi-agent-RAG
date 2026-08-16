"""The sole Bedrock entry point.

Every Bedrock call in this codebase goes through here. No direct boto3 calls
live anywhere else, because the cost ledger's completeness depends on that.

Call path for every request:
    cache lookup  ->  (hit: return, zero cost)
                  ->  (miss: pre-flight budget estimate -> BudgetExceeded?
                             -> call Bedrock, measure wall clock
                             -> read REAL token counts from response usage
                             -> record to CostTracker -> write cache -> return)

Measured dC is a core claim of the paper, so token counts are always read from
the response `usage` field and never estimated after the fact.

DRY_RUN=1 returns canned responses without touching the network, so graph wiring
can be tested for $0.00.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from cache import CACHE, make_key
from costs import TRACKER, BudgetExceeded  # noqa: F401  (re-exported for callers)

log = logging.getLogger("caes.bedrock")

DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

# Rough char/token ratio, used ONLY for the pre-flight affordability estimate.
# Never used for billing — that always comes from the response usage field.
_CHARS_PER_TOKEN = 3.6

_MAX_RETRIES = 5
_RETRYABLE = ("ThrottlingException", "TooManyRequestsException",
              "ServiceUnavailableException", "ModelTimeoutException",
              "InternalServerException")


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool
    usd: float            # actually billed this call (0.0 on a cache hit)
    notional_usd: float   # what the call costs at list price, cached or not


# ---------------------------------------------------------------------------
# Notional accounting
# ---------------------------------------------------------------------------
# The ledger tracks money actually spent. The CAES gate needs something else:
# what an iteration *costs*, independent of whether this particular run replayed
# it from cache. If dC collapsed to zero on a warm cache the gate would always
# see an infinitely cheap next iteration and run to MAX_ITERATIONS, so a cached
# re-run would produce different decisions from the paid one. Notional cost is
# computed from the real measured token counts (which the cache preserves), so
# results are identical cached or not.

_totals_lock = threading.Lock()
_TOTALS = {"notional_usd": 0.0, "actual_usd": 0.0, "latency_ms": 0.0,
           "calls": 0, "cache_hits": 0}


def _accrue(*, notional_usd: float, actual_usd: float, latency_ms: float,
            cached: bool) -> None:
    with _totals_lock:
        _TOTALS["notional_usd"] += notional_usd
        _TOTALS["actual_usd"] += actual_usd
        _TOTALS["latency_ms"] += latency_ms
        _TOTALS["calls"] += 1
        if cached:
            _TOTALS["cache_hits"] += 1


def totals() -> dict[str, float]:
    """Monotonic process-wide counters. Snapshot and diff to meter a span."""
    with _totals_lock:
        return dict(_TOTALS)


def notional_llm_usd(in_tokens: int, out_tokens: int) -> float:
    return (in_tokens / 1000.0 * config.PRICE_HAIKU_INPUT_PER_1K
            + out_tokens / 1000.0 * config.PRICE_HAIKU_OUTPUT_PER_1K)


_client = None


def get_client():
    """Lazily construct the bedrock-runtime client.

    Lazy so that DRY_RUN and unit tests never need AWS credentials.
    """
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
    return _client


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _invoke_with_retry(model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    from botocore.exceptions import ClientError

    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = get_client().invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(resp["body"].read())
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _RETRYABLE or attempt == _MAX_RETRIES - 1:
                raise
            last_exc = exc
            log.warning("Bedrock %s (attempt %d/%d); backing off %.1fs",
                        code, attempt + 1, _MAX_RETRIES, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable") from last_exc


# --------------------------------------------------------------------------
# DRY_RUN canned responses
# --------------------------------------------------------------------------

def _dry_run_text(call_type: str, query_id: str, iteration: int) -> str:
    """Deterministic canned output, shaped like the real thing.

    The verifier's coverage curve is synthesised with diminishing returns and
    per-query variation, so a DRY_RUN pass exercises the CAES gate realistically
    (iteration counts genuinely vary) rather than trivially.
    """
    rng = random.Random(f"{call_type}|{query_id}|{iteration}")
    if call_type == "verify":
        base_rng = random.Random(f"base|{query_id}")
        base = 0.10 + 0.45 * base_rng.random()
        spread = 0.4 + 0.6 * base_rng.random()
        coverage = base + (1.0 - base) * (1.0 - 0.55 ** max(0, iteration)) * spread
        coverage = round(min(0.98, max(0.0, coverage)), 2)
        missing = ("nothing" if coverage >= 0.9
                   else f"second-hop detail for {query_id or 'the question'}")
        return json.dumps({
            "coverage": coverage,
            "missing": missing,
            "confident": coverage >= 0.85,
        })
    if call_type == "plan":
        return f"focused sub-query {iteration} about {query_id or 'the topic'}"
    if call_type == "generate":
        return (f"[DRY_RUN] Synthetic grounded answer {rng.randint(1000, 9999)} "
                f"for {query_id or 'the question'}.")
    return "[DRY_RUN] response"


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

def invoke_llm(
    prompt: str,
    *,
    call_type: str,
    query_id: str = "",
    iteration: int = 0,
    max_tokens: int = 512,
    system: str | None = None,
    policy: str = "",
    temperature: float = 0.0,
) -> LLMResponse:
    """Invoke Claude Haiku on Bedrock through the cache and the budget gate."""
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    key = make_key(config.MODEL_LLM, body)
    hit = CACHE.get(key)
    if hit is not None:
        # A replayed call costs nothing and is never recorded to the ledger,
        # but it still accrues notional cost so the gate behaves identically.
        notional = notional_llm_usd(hit["input_tokens"], hit["output_tokens"])
        _accrue(notional_usd=notional, actual_usd=0.0,
                latency_ms=hit["latency_ms"], cached=True)
        return LLMResponse(
            text=hit["text"], input_tokens=hit["input_tokens"],
            output_tokens=hit["output_tokens"], latency_ms=hit["latency_ms"],
            cached=True, usd=0.0, notional_usd=notional,
        )

    est_in = _estimate_tokens(prompt) + (_estimate_tokens(system) if system else 0)
    TRACKER.check_affordable(TRACKER.estimate_llm_cost(est_in, max_tokens))

    t0 = time.perf_counter()
    if DRY_RUN:
        text = _dry_run_text(call_type, query_id, iteration)
        in_tokens, out_tokens = est_in, _estimate_tokens(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usd = 0.0   # nothing was billed; do not pollute the ledger
    else:
        payload = _invoke_with_retry(config.MODEL_LLM, body)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = "".join(
            blk.get("text", "") for blk in payload.get("content", [])
            if blk.get("type") == "text"
        )
        usage = payload.get("usage", {})
        if "input_tokens" not in usage or "output_tokens" not in usage:
            raise RuntimeError(
                "Bedrock response carried no usage block. Measured cost is a core "
                "claim of this work; refusing to fall back to an estimate."
            )
        in_tokens = int(usage["input_tokens"])
        out_tokens = int(usage["output_tokens"])
        usd = TRACKER.record_llm(
            call_type=call_type, in_tokens=in_tokens, out_tokens=out_tokens,
            latency_ms=latency_ms, query_id=query_id, iteration=iteration,
            policy=policy,
        )

    CACHE.set(key, {"text": text, "input_tokens": in_tokens,
                    "output_tokens": out_tokens, "latency_ms": latency_ms})
    notional = notional_llm_usd(in_tokens, out_tokens)
    _accrue(notional_usd=notional, actual_usd=usd, latency_ms=latency_ms,
            cached=False)
    return LLMResponse(text=text, input_tokens=in_tokens, output_tokens=out_tokens,
                       latency_ms=latency_ms, cached=False, usd=usd,
                       notional_usd=notional)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def _dry_run_embedding(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding so DRY_RUN retrieval is stable and free."""
    seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")
    rng = np.random.default_rng(seed % (2**32))
    v = rng.standard_normal(config.EMBED_DIM).astype("float32")
    return v / (np.linalg.norm(v) + 1e-12)


def embed(
    texts: list[str],
    *,
    query_id: str = "",
    iteration: int = 0,
    policy: str = "",
) -> np.ndarray:
    """Embed texts with Titan V2, returning L2-normalised float32 vectors.

    Titan's invoke_model takes one document per call, so callers batch for
    progress reporting and rate-limit pacing rather than for a batch endpoint.
    Cost per embedded text is tracked individually.
    """
    if isinstance(texts, str):
        texts = [texts]
    out = np.zeros((len(texts), config.EMBED_DIM), dtype="float32")

    for i, text in enumerate(texts):
        body = {
            "inputText": text,
            "dimensions": config.EMBED_DIM,
            "normalize": True,
        }
        key = make_key(config.MODEL_EMBED, body)
        hit = CACHE.get(key)
        if hit is not None:
            out[i] = np.asarray(hit["embedding"], dtype="float32")
            _accrue(
                notional_usd=TRACKER.estimate_embed_cost(hit.get("in_tokens", 0)),
                actual_usd=0.0, latency_ms=hit.get("latency_ms", 0.0),
                cached=True,
            )
            continue

        est_tokens = _estimate_tokens(text)
        TRACKER.check_affordable(TRACKER.estimate_embed_cost(est_tokens))

        t0 = time.perf_counter()
        if DRY_RUN:
            vec = _dry_run_embedding(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            in_tokens = est_tokens
            usd = 0.0   # nothing was billed; do not pollute the ledger
        else:
            payload = _invoke_with_retry(config.MODEL_EMBED, body)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            vec = np.asarray(payload["embedding"], dtype="float32")
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n   # belt and braces; normalize=True should do this
            in_tokens = int(payload.get("inputTextTokenCount", est_tokens))
            usd = TRACKER.record_embed(
                in_tokens=in_tokens, latency_ms=latency_ms, query_id=query_id,
                iteration=iteration, policy=policy,
            )

        CACHE.set(key, {"embedding": vec.tolist(), "in_tokens": in_tokens,
                        "latency_ms": latency_ms})
        _accrue(notional_usd=TRACKER.estimate_embed_cost(in_tokens),
                actual_usd=usd, latency_ms=latency_ms, cached=False)
        out[i] = vec

    return out
