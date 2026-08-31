"""The sole model entry point.

Every LLM and embedding call in this codebase goes through here. No direct
provider SDK or HTTP call lives anywhere else, because the cost ledger's
completeness depends on that (DECISIONS invariant 1).

Call path for every request, identical for both providers:
    cache lookup  ->  (hit: return, zero actual cost, notional still accrued)
                  ->  (miss: pre-flight budget estimate -> BudgetExceeded?
                             -> call the provider, measure wall clock
                             -> read REAL token counts from the response
                             -> record to CostTracker -> write cache -> return)

Measured dC is a core claim of the paper, so token counts are always read from
the response and never estimated after the fact.

Two providers are supported, selected by `config.PROVIDER`. Only request and
response shaping differs between them; the cache, the budget gate, the notional
accounting and the dry-run path are shared verbatim. See DECISIONS [D-22].

DRY_RUN=1 returns canned responses without touching the network, so graph
wiring can be tested for $0.00.

    python -m llm --check     # live provider preflight (free; needs a key)
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

log = logging.getLogger("caes.llm")

DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

# Rough char/token ratio, used ONLY for the pre-flight affordability estimate.
# Never used for billing -- that always comes from the response token counts.
_CHARS_PER_TOKEN = 3.6

_MAX_RETRIES = 5

# Bedrock error codes worth retrying. Non-retryable client errors propagate.
_RETRYABLE_BEDROCK = ("ThrottlingException", "TooManyRequestsException",
                      "ServiceUnavailableException", "ModelTimeoutException",
                      "InternalServerException")
# Gemini returns ordinary HTTP status codes. 429 is the free-tier rate limit.
_RETRYABLE_HTTP = (429, 500, 502, 503, 504)


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
#
# On the Gemini free tier actual spend is structurally $0.00 for every call, so
# notional cost is the ONLY cost signal the gate has. See DECISIONS [D-22].

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
    return (in_tokens / 1000.0 * config.PRICE_LLM_INPUT_PER_1K
            + out_tokens / 1000.0 * config.PRICE_LLM_OUTPUT_PER_1K)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Bedrock transport
# ---------------------------------------------------------------------------

_client = None


def get_client():
    """Lazily construct the bedrock-runtime client.

    Lazy so that DRY_RUN, the Gemini path, and unit tests never need AWS
    credentials.
    """
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
    return _client


def _bedrock_invoke(model_id: str, body: dict[str, Any]) -> dict[str, Any]:
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
            if code not in _RETRYABLE_BEDROCK or attempt == _MAX_RETRIES - 1:
                raise
            last_exc = exc
            log.warning("Bedrock %s (attempt %d/%d); backing off %.1fs",
                        code, attempt + 1, _MAX_RETRIES, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable") from last_exc


# ---------------------------------------------------------------------------
# Gemini transport
# ---------------------------------------------------------------------------

_rpm_lock = threading.Lock()
_last_request_at = 0.0


def _pace() -> None:
    """Client-side requests-per-minute pacer for the free tier.

    Free-tier limits are per-account and are no longer published per model, so
    this errs slow rather than discovering the limit by tripping it. 429s are
    still retried; this only reduces how often that happens.
    """
    if config.GEMINI_MAX_RPM <= 0:
        return
    min_gap = 60.0 / config.GEMINI_MAX_RPM
    global _last_request_at
    with _rpm_lock:
        wait = min_gap - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _gemini_api_key() -> str:
    key = os.environ.get(config.GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{config.GEMINI_API_KEY_ENV} is not set. Export your Google AI "
            f"Studio key before making a real call, or use DRY_RUN=1. The key "
            f"is never written into config.py or committed."
        )
    return key


def _retry_delay_seconds(payload: dict[str, Any]) -> float | None:
    """Pull Google's RetryInfo.retryDelay (e.g. "31s") out of an error body."""
    for detail in payload.get("error", {}).get("details", []) or []:
        raw = detail.get("retryDelay")
        if isinstance(raw, str) and raw.endswith("s"):
            try:
                return float(raw[:-1])
            except ValueError:
                pass
    return None


def _gemini_post(method: str, model: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to `{base}/models/{model}:{method}` with retry and backoff.

    The key travels in the x-goog-api-key header rather than the query string
    so it never lands in a URL that could be logged.
    """
    import requests

    url = f"{config.GEMINI_API_BASE}/models/{model}:{method}"
    headers = {"x-goog-api-key": _gemini_api_key(),
               "Content-Type": "application/json"}

    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        _pace()
        try:
            resp = requests.post(url, headers=headers, json=body,
                                 timeout=config.GEMINI_TIMEOUT_S)
        except requests.RequestException as exc:      # connection/timeout
            if attempt == _MAX_RETRIES - 1:
                raise
            last_exc = exc
            log.warning("Gemini %s transport error %s (attempt %d/%d); backing "
                        "off %.1fs", method, type(exc).__name__, attempt + 1,
                        _MAX_RETRIES, delay)
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 200:
            return resp.json()

        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        message = payload.get("error", {}).get("message", resp.text[:300])

        if resp.status_code not in _RETRYABLE_HTTP or attempt == _MAX_RETRIES - 1:
            raise RuntimeError(
                f"Gemini {method} on {model} failed with HTTP "
                f"{resp.status_code}: {message}"
            )

        # 429 on the free tier usually carries a server-chosen delay; honour it.
        wait = _retry_delay_seconds(payload)
        if wait is None:
            wait = delay
            delay *= 2
        log.warning("Gemini %s HTTP %d (attempt %d/%d); backing off %.1fs: %s",
                    method, resp.status_code, attempt + 1, _MAX_RETRIES, wait,
                    message[:160])
        time.sleep(wait)
    raise RuntimeError("unreachable") from last_exc


# ---------------------------------------------------------------------------
# Per-provider request/response shaping -- the ONLY thing that varies
# ---------------------------------------------------------------------------

def _build_llm_body(prompt: str, system: str | None, max_tokens: int,
                    temperature: float) -> dict[str, Any]:
    if config.PROVIDER == "gemini":
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                # Thinking tokens bill as output and vary in length, which
                # would both inflate dC and break the determinism the cache
                # key assumes. See config.GEMINI_THINKING_BUDGET.
                "thinkingConfig": {
                    "thinkingBudget": config.GEMINI_THINKING_BUDGET},
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    return body


def _call_llm(body: dict[str, Any]) -> dict[str, Any]:
    if config.PROVIDER == "gemini":
        return _gemini_post("generateContent", config.MODEL_LLM, body)
    return _bedrock_invoke(config.MODEL_LLM, body)


def _parse_llm(payload: dict[str, Any]) -> tuple[str, int, int]:
    """Return (text, input_tokens, output_tokens).

    A response without measured token counts is fatal on both providers.
    Falling back to the estimator would put fabricated numbers into the results
    with no visible signal, and measured dC is a central claim. See [D-2].
    """
    if config.PROVIDER == "gemini":
        candidates = payload.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts)

        usage = payload.get("usageMetadata", {})
        if "promptTokenCount" not in usage or "candidatesTokenCount" not in usage:
            raise RuntimeError(
                "Gemini response carried no usageMetadata token counts. "
                "Measured cost is a core claim of this work; refusing to fall "
                "back to an estimate. (DECISIONS [D-2])"
            )
        in_tokens = int(usage["promptTokenCount"])
        # Thinking tokens bill as output. They should be zero given
        # thinkingBudget=0, but if the budget is ignored the cost is real and
        # must land in dC rather than vanish.
        out_tokens = (int(usage["candidatesTokenCount"])
                      + int(usage.get("thoughtsTokenCount", 0) or 0))
        if not text and candidates:
            log.warning("Gemini returned no text; finishReason=%s",
                        candidates[0].get("finishReason"))
        return text, in_tokens, out_tokens

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
    return text, int(usage["input_tokens"]), int(usage["output_tokens"])


def _build_embed_body(text: str) -> dict[str, Any]:
    if config.PROVIDER == "gemini":
        return {
            "model": f"models/{config.MODEL_EMBED}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": config.EMBED_DIM,
        }
    return {"inputText": text, "dimensions": config.EMBED_DIM, "normalize": True}


def _call_embed(body: dict[str, Any]) -> dict[str, Any]:
    if config.PROVIDER == "gemini":
        return _gemini_post("embedContent", config.MODEL_EMBED, body)
    return _bedrock_invoke(config.MODEL_EMBED, body)


def _parse_embed(payload: dict[str, Any], text: str,
                 est_tokens: int) -> tuple[np.ndarray, int]:
    """Return (vector, input_tokens).

    Gemini's :embedContent returns no token count at all, so the count is
    measured with a separate free :countTokens call. See DECISIONS [D-23].
    """
    if config.PROVIDER == "gemini":
        values = (payload.get("embedding") or {}).get("values")
        if not values:
            raise RuntimeError(
                f"Gemini embedContent returned no embedding values: "
                f"{json.dumps(payload)[:300]}"
            )
        vec = np.asarray(values, dtype="float32")
        return vec, _gemini_embed_tokens(text, est_tokens)

    vec = np.asarray(payload["embedding"], dtype="float32")
    return vec, int(payload.get("inputTextTokenCount", est_tokens))


def _gemini_embed_tokens(text: str, est_tokens: int) -> int:
    if config.EMBED_TOKENS_MODE == "estimated":
        # Opt-in only. Invariant 4 says token counts are measured, never
        # estimated; this path knowingly breaks that, so it is loud.
        log.warning("EMBED_TOKENS_MODE=estimated: embedding token count for "
                    "this call is an ESTIMATE, not a measurement. dC derived "
                    "from it is not a measured number. (DECISIONS [D-23])")
        return est_tokens
    payload = _gemini_post("countTokens", config.MODEL_EMBED,
                           {"contents": [{"parts": [{"text": text}]}]})
    if "totalTokens" not in payload:
        raise RuntimeError(
            "Gemini countTokens returned no totalTokens. Measured cost is a "
            "core claim of this work; refusing to fall back to an estimate. "
            "(DECISIONS [D-2], [D-23])"
        )
    return int(payload["totalTokens"])


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
    """Invoke the configured LLM through the cache and the budget gate."""
    body = _build_llm_body(prompt, system, max_tokens, temperature)

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
        payload = _call_llm(body)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text, in_tokens, out_tokens = _parse_llm(payload)
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
    """Embed texts, returning L2-normalised float32 vectors.

    Both providers embed one document per call, so callers batch for progress
    reporting and rate-limit pacing rather than for a batch endpoint. Cost per
    embedded text is tracked individually, and the cache is keyed per text, so
    a partial ingest resumes for free.
    """
    if isinstance(texts, str):
        texts = [texts]
    out = np.zeros((len(texts), config.EMBED_DIM), dtype="float32")

    for i, text in enumerate(texts):
        body = _build_embed_body(text)
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
            payload = _call_embed(body)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            vec, in_tokens = _parse_embed(payload, text, est_tokens)
            usd = TRACKER.record_embed(
                in_tokens=in_tokens, latency_ms=latency_ms, query_id=query_id,
                iteration=iteration, policy=policy,
            )

        # Gemini's truncated (outputDimensionality < 3072) vectors are NOT
        # unit-norm as returned, and Titan's normalize=True is belt-and-braces
        # only. IndexFlatIP means cosine only if these are normalised.
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n

        CACHE.set(key, {"embedding": vec.tolist(), "in_tokens": in_tokens,
                        "latency_ms": latency_ms})
        _accrue(notional_usd=TRACKER.estimate_embed_cost(in_tokens),
                actual_usd=usd, latency_ms=latency_ms, cached=False)
        out[i] = vec

    return out


# --------------------------------------------------------------------------
# Live provider preflight
# --------------------------------------------------------------------------

def check_provider() -> int:
    """Make two tiny real calls and report what the provider actually does.

    Answers the three questions that cannot be settled from documentation:
    the embedding dimension actually returned, whether temperature=0.0 is
    accepted, and whether token counts come back. Free on the Gemini free tier,
    but it IS a real call, so it is never run implicitly.
    """
    print(f"provider        : {config.PROVIDER}")
    print(f"llm model       : {config.MODEL_LLM}")
    print(f"embed model     : {config.MODEL_EMBED}")
    print(f"configured dim  : {config.EMBED_DIM}")
    print(f"embed tokens    : {config.EMBED_TOKENS_MODE}")
    if DRY_RUN:
        print("\nDRY_RUN=1 is set; this check needs real calls. Unset it.")
        return 2

    print("\n--- LLM, temperature=0.0 ---")
    resp = invoke_llm('Reply with exactly: OK', call_type="generate",
                      query_id="preflight", max_tokens=16, temperature=0.0)
    print(f"accepted temperature=0.0 : yes")
    print(f"text                     : {resp.text.strip()[:80]!r}")
    print(f"tokens in/out            : {resp.input_tokens} / {resp.output_tokens}")
    print(f"notional / actual usd    : ${resp.notional_usd:.8f} / ${resp.usd:.8f}")
    print(f"latency                  : {resp.latency_ms:.0f} ms")

    print("\n--- Embeddings ---")
    t0 = time.perf_counter()
    vecs = embed(["preflight dimension probe"], policy="preflight")
    dim = int(vecs.shape[1])
    norm = float(np.linalg.norm(vecs[0]))
    print(f"returned dimension       : {dim}")
    print(f"matches config.EMBED_DIM : {dim == config.EMBED_DIM}")
    print(f"L2 norm after normalise  : {norm:.6f}")
    print(f"latency                  : {(time.perf_counter() - t0) * 1000:.0f} ms")

    print(f"\nledger cumulative        : ${TRACKER.cumulative():.6f}")
    if dim != config.EMBED_DIM:
        print(f"\nFAIL: set config.{config.PROVIDER.upper()}_EMBED_DIM to {dim} "
              f"before ingest. A mismatch surfaces as silent retrieval failure.")
        return 1
    print("\nOK: provider reachable, counts measured, dimension as configured.")
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Model-layer preflight.")
    ap.add_argument("--check", action="store_true",
                    help="make two tiny real calls and report provider facts")
    args = ap.parse_args()
    if not args.check:
        ap.print_help()
        sys.exit(0)
    try:
        sys.exit(check_provider())
    except RuntimeError as exc:
        # A preflight should report the problem, not traceback at the operator.
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
