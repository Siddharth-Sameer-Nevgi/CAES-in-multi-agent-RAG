"""The provider abstraction: shaping, parsing, pricing, and what must NOT vary.

The migration's risk is that swapping providers quietly changes something other
than request shaping and price constants. These tests pin the boundary.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import config
import llm


# ---------------------------------------------------------------------------
# Price constants: pinned to the published tables, with their source dates.
# ---------------------------------------------------------------------------

def test_published_bedrock_prices():
    """us-east-1 on-demand, confirmed 2026-08-16."""
    assert config.BEDROCK_PRICE_LLM_INPUT_PER_1K == 0.001      # $1.00 / 1M
    assert config.BEDROCK_PRICE_LLM_OUTPUT_PER_1K == 0.005     # $5.00 / 1M
    # Guards the [0.1.1] correction: v1's $0.10/1M must never come back.
    assert config.BEDROCK_PRICE_EMBED_PER_1K == 0.00002        # $0.02 / 1M


def test_published_gemini_prices():
    """ai.google.dev/gemini-api/docs/pricing, paid tier, confirmed 2026-08-31."""
    assert config.GEMINI_PRICE_LLM_INPUT_PER_1K == 0.0003      # $0.30 / 1M
    assert config.GEMINI_PRICE_LLM_OUTPUT_PER_1K == 0.0025     # $2.50 / 1M
    assert config.GEMINI_PRICE_EMBED_PER_1K == 0.00015         # $0.15 / 1M


def test_provider_settings_cover_every_varying_name(provider):
    """config.PROVIDER and the neutral names must never disagree."""
    settings = config.provider_settings(provider)
    for name, value in settings.items():
        assert getattr(config, name) == value, f"{name} did not follow PROVIDER"
    assert set(settings) == {
        "MODEL_LLM", "MODEL_EMBED", "EMBED_DIM",
        "PRICE_LLM_INPUT_PER_1K", "PRICE_LLM_OUTPUT_PER_1K",
        "PRICE_EMBED_PER_1K",
    }, "a provider-varying name was added without updating provider_settings"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        config.provider_settings("openai")


def test_embed_dims_differ_and_are_positive():
    """A dimension mismatch surfaces as silent retrieval failure, not an error."""
    assert config.BEDROCK_EMBED_DIM == 1024
    assert config.GEMINI_EMBED_DIM == 768
    assert config.GEMINI_EMBED_DIM != config.BEDROCK_EMBED_DIM


# ---------------------------------------------------------------------------
# Invariants that must hold identically on both providers
# ---------------------------------------------------------------------------

def test_token_counts_come_from_the_response_not_the_estimator(wired):
    """Invariant 4 / [D-2] on both paths."""
    fake, _, _ = wired
    fake.in_tokens, fake.out_tokens = 4242, 77
    resp = llm.invoke_llm("x" * 5000, call_type="verify")
    assert resp.input_tokens == 4242, "input tokens were not read from the response"
    assert resp.output_tokens == 77
    # The estimator would have produced ~1389 for a 5000-char prompt.
    assert resp.input_tokens != llm._estimate_tokens("x" * 5000)


def test_temperature_zero_is_sent_on_both_providers(wired):
    """[D-19]: every call is temperature 0.0, however the provider spells it."""
    fake, _, _ = wired
    llm.invoke_llm("hello", call_type="generate")
    body = fake.bodies[0]
    if config.PROVIDER == "gemini":
        assert body["generationConfig"]["temperature"] == 0.0
    else:
        assert body["temperature"] == 0.0


def test_notional_is_computed_from_the_active_price_table(wired):
    fake, _, _ = wired
    fake.in_tokens, fake.out_tokens = 1000, 1000
    resp = llm.invoke_llm("hello", call_type="generate")
    assert resp.notional_usd == pytest.approx(
        config.PRICE_LLM_INPUT_PER_1K + config.PRICE_LLM_OUTPUT_PER_1K)


def test_embeddings_are_unit_norm_and_correctly_shaped(wired):
    """IndexFlatIP means cosine only if these come back normalised."""
    _, _, _ = wired
    vecs = llm.embed(["hello world"], policy="test")
    assert vecs.shape == (1, config.EMBED_DIM)
    assert float(np.linalg.norm(vecs[0])) == pytest.approx(1.0, abs=1e-5)


def test_embedding_cost_is_recorded_from_a_token_count(wired):
    fake, tracker, _ = wired
    fake.in_tokens = 500
    llm.embed(["hello world"], policy="test")
    embed_records = [r for r in tracker.records() if r.call_type == "embed"]
    assert len(embed_records) == 1
    assert embed_records[0].in_tokens == 500
    assert embed_records[0].usd == pytest.approx(
        500 / 1000.0 * config.PRICE_EMBED_PER_1K)


# ---------------------------------------------------------------------------
# Gemini-specific shaping
# ---------------------------------------------------------------------------

def _gemini_only():
    if config.PROVIDER != "gemini":
        pytest.skip("Gemini-specific shaping")


def test_gemini_disables_thinking_and_sets_max_tokens(wired):
    _gemini_only()
    fake, _, _ = wired
    llm.invoke_llm("hello", call_type="generate", max_tokens=123)
    cfg = fake.bodies[0]["generationConfig"]
    assert cfg["maxOutputTokens"] == 123
    assert cfg["thinkingConfig"]["thinkingBudget"] == config.GEMINI_THINKING_BUDGET
    assert config.GEMINI_THINKING_BUDGET == 0, \
        "thinking tokens bill as output and would inflate dC"


def test_gemini_system_prompt_uses_system_instruction(wired):
    _gemini_only()
    fake, _, _ = wired
    llm.invoke_llm("hello", call_type="generate", system="be terse")
    assert fake.bodies[0]["systemInstruction"]["parts"][0]["text"] == "be terse"


def test_gemini_thinking_tokens_are_billed_as_output(wired):
    """If thinkingBudget is ever ignored, the cost must land in dC, not vanish."""
    _gemini_only()
    fake, _, _ = wired
    fake.in_tokens, fake.out_tokens, fake.thoughts_tokens = 100, 50, 30
    resp = llm.invoke_llm("hello", call_type="generate")
    assert resp.output_tokens == 80, "thoughtsTokenCount was dropped from dC"


def test_gemini_api_key_travels_in_a_header_not_the_url(wired):
    _gemini_only()
    fake, _, _ = wired
    llm.invoke_llm("hello", call_type="generate")
    assert "x-goog-api-key" in fake.headers[0]
    assert "key=" not in fake.urls[0], "the API key would be logged in the URL"


def test_gemini_missing_api_key_raises_before_the_network(monkeypatch, provider):
    if provider != "gemini":
        pytest.skip("Gemini-specific")
    monkeypatch.delenv(config.GEMINI_API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=config.GEMINI_API_KEY_ENV):
        llm._gemini_api_key()


def test_gemini_429_is_retried_with_the_server_supplied_delay(wired, monkeypatch):
    """Free-tier rate limits are the binding constraint; 429 must not be fatal."""
    _gemini_only()
    fake, _, _ = wired
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    fake.canned = [(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                     "retryDelay": "7s"}]}})]

    resp = llm.invoke_llm("hello", call_type="generate")
    assert resp.text == "fake answer", "the retry did not recover"
    assert fake.calls == 2
    assert slept == [7.0], "the server-supplied retryDelay was ignored"


def test_gemini_non_retryable_error_propagates(wired):
    _gemini_only()
    fake, _, _ = wired
    fake.canned = [(400, {"error": {"code": 400, "message": "bad request"}})]
    with pytest.raises(RuntimeError, match="400"):
        llm.invoke_llm("hello", call_type="generate")
    assert fake.calls == 1, "a client error was retried"


def test_gemini_embed_token_count_is_measured_not_estimated(wired):
    """[D-23]: embedContent returns no count, so countTokens supplies it."""
    _gemini_only()
    fake, tracker, _ = wired
    fake.in_tokens = 321
    llm.embed(["a text whose estimated length differs from 321"], policy="test")
    assert any(u.endswith(":countTokens") for u in fake.urls), \
        "the embedding token count was not measured"
    rec = [r for r in tracker.records() if r.call_type == "embed"][0]
    assert rec.in_tokens == 321


def test_gemini_countTokens_without_a_total_is_fatal(wired):
    _gemini_only()
    fake, _, _ = wired
    # Serve the embedding, then a countTokens response missing totalTokens.
    fake.canned = [(200, {"embedding": {"values": [0.0] * config.EMBED_DIM}}),
                   (200, {})]
    with pytest.raises(RuntimeError, match="(?i)totaltokens"):
        llm.embed(["hello"], policy="test")


def test_gemini_estimated_embed_mode_is_opt_in_and_warns(wired, monkeypatch, caplog):
    _gemini_only()
    fake, _, _ = wired
    monkeypatch.setattr(config, "EMBED_TOKENS_MODE", "estimated")
    with caplog.at_level("WARNING"):
        llm.embed(["hello world"], policy="test")
    assert not any(u.endswith(":countTokens") for u in fake.urls)
    assert any("ESTIMATE" in r.message for r in caplog.records), \
        "estimating token counts must be loud; it breaks invariant 4"


def test_gemini_requests_the_configured_output_dimensionality(wired):
    _gemini_only()
    fake, _, _ = wired
    llm.embed(["hello"], policy="test")
    embed_body = next(b for b, u in zip(fake.bodies, fake.urls)
                      if u.endswith(":embedContent"))
    assert embed_body["outputDimensionality"] == config.EMBED_DIM


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

def test_no_api_key_value_is_committed():
    """The env var NAME may appear anywhere; a key VALUE must not be tracked."""
    found = subprocess.run(
        ["git", "grep", "-InE", "AIza[0-9A-Za-z_-]{35}"],
        capture_output=True, text=True, cwd=str(config.ROOT))
    assert found.stdout.strip() == "", (
        f"a Google API key literal is tracked:\n{found.stdout}")


def test_llm_is_the_only_module_that_calls_a_provider():
    """Invariant 1. The one failure no runtime test can catch, pinned statically.

    config.py holds the endpoint constant but calls nothing; llm.py is the only
    module allowed to construct a client or issue a request against either
    provider. A direct call anywhere else is invisible to the ledger, the cache
    and the budget guard, and silently corrupts dC.
    """
    import pathlib

    allowed = {"llm.py", "config.py"}
    offenders = []
    for path in pathlib.Path(config.ROOT).rglob("*.py"):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if ("bedrock-runtime" in text
                or "generativelanguage.googleapis.com" in text
                or "GEMINI_API_BASE" in text):
            offenders.append(path.name)
    assert offenders == [], (
        f"{offenders} reach a provider directly, bypassing the ledger")
