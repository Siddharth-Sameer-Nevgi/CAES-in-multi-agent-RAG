"""Shared fixtures: fake transports for both providers.

The point of the `wired` fixture is that every transport-level test runs twice,
once per provider, against a fake that speaks that provider's real wire format.
`config.PROVIDER` selects which shaping `llm.py` uses, so a test that only ever
ran under one provider would leave the other's request building and response
parsing entirely unexercised.

No test here reaches the network: the Bedrock path is fed a fake boto3 client,
the Gemini path a fake `requests.post`.
"""
from __future__ import annotations

import json

import pytest

import cache as cache_mod
import config
import llm as llm_mod
from costs import CostTracker

PROVIDERS = ("bedrock", "gemini")


# ---------------------------------------------------------------------------
# Fake transports
# ---------------------------------------------------------------------------

class _Body:
    """Mimics the streaming body boto3 returns."""

    def __init__(self, s):
        self._s = s

    def read(self):
        return self._s


class FakeBedrockClient:
    """Stands in for bedrock-runtime. Records whether it was ever reached."""

    def __init__(self, in_tokens=100, out_tokens=50, embed_dim=None):
        self.calls = 0
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.embed_dim = embed_dim
        self.bodies: list[dict] = []

    def invoke_model(self, **kwargs):
        self.calls += 1
        body = json.loads(kwargs["body"])
        self.bodies.append(body)
        if "inputText" in body:                      # Titan embeddings
            dim = self.embed_dim or body.get("dimensions") or config.EMBED_DIM
            payload = {"embedding": [0.0] * (dim - 1) + [1.0],
                       "inputTextTokenCount": self.in_tokens}
        else:
            payload = {
                "content": [{"type": "text", "text": "fake answer"}],
                "usage": {"input_tokens": self.in_tokens,
                          "output_tokens": self.out_tokens},
            }
        return {"body": _Body(json.dumps(payload))}


class _FakeHTTPResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeGeminiTransport:
    """Stands in for `requests.post` against generativelanguage.googleapis.com.

    Dispatches on the `:method` suffix of the URL the same way the real service
    does, so request shaping is genuinely exercised rather than bypassed.
    """

    def __init__(self, in_tokens=100, out_tokens=50, embed_dim=None):
        self.calls = 0
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.embed_dim = embed_dim
        self.bodies: list[dict] = []
        self.urls: list[str] = []
        self.headers: list[dict] = []
        # Queue of (status_code, payload) to serve before normal behaviour.
        self.canned: list[tuple[int, dict]] = []
        # Set to drop a field from the next generateContent usageMetadata.
        self.omit_usage = False
        self.thoughts_tokens = 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        self.urls.append(url)
        self.headers.append(headers or {})
        self.bodies.append(json or {})

        if self.canned:
            return _FakeHTTPResponse(*self.canned.pop(0))

        if url.endswith(":generateContent"):
            usage = {"promptTokenCount": self.in_tokens,
                     "candidatesTokenCount": self.out_tokens}
            if self.thoughts_tokens:
                usage["thoughtsTokenCount"] = self.thoughts_tokens
            if self.omit_usage:
                usage.pop("candidatesTokenCount")
            return _FakeHTTPResponse(200, {
                "candidates": [{
                    "content": {"parts": [{"text": "fake answer"}],
                                "role": "model"},
                    "finishReason": "STOP",
                }],
                "usageMetadata": usage,
            })
        if url.endswith(":embedContent"):
            dim = (self.embed_dim
                   or (json or {}).get("outputDimensionality")
                   or config.EMBED_DIM)
            # Deliberately NOT unit-norm: real truncated Gemini vectors are not,
            # and llm.embed is responsible for normalising them.
            return _FakeHTTPResponse(200, {
                "embedding": {"values": [0.0] * (dim - 1) + [3.0]}})
        if url.endswith(":countTokens"):
            return _FakeHTTPResponse(200, {"totalTokens": self.in_tokens})
        raise AssertionError(f"unexpected Gemini URL: {url}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracker(tmp_path):
    return CostTracker(ledger_path=tmp_path / "ledger.json",
                       hard_budget_usd=1.00, warn_budget_usd=0.50)


def _select_provider(monkeypatch, provider: str) -> None:
    """Point config at one provider, including every name that varies with it."""
    monkeypatch.setattr(config, "PROVIDER", provider)
    for name, value in config.provider_settings(provider).items():
        monkeypatch.setattr(config, name, value)
    # The pacer would otherwise sleep 4s between calls at the default 15 RPM.
    monkeypatch.setattr(config, "GEMINI_MAX_RPM", 0)


def _forbidden(what: str):
    def _boom(*args, **kwargs):
        raise AssertionError(
            f"{what} was reached under provider={config.PROVIDER}; the provider "
            f"switch leaked."
        )
    return _boom


@pytest.fixture(params=PROVIDERS)
def wired(request, monkeypatch, tmp_path):
    """`llm` wired to a fake transport, a fresh cache and a fresh ledger.

    Runs once per provider. Returns (fake, tracker, disk); `fake.calls` counts
    network attempts on either provider, so budget assertions read identically
    for both. The provider not under test is booby-trapped, so a shaping bug
    that routed a call to the wrong transport fails loudly instead of silently
    passing.
    """
    provider_name = request.param
    _select_provider(monkeypatch, provider_name)

    trk = CostTracker(ledger_path=tmp_path / "ledger.json", hard_budget_usd=10.0)
    disk = cache_mod.DiskCache(tmp_path / "cache")
    monkeypatch.setattr(llm_mod, "TRACKER", trk)
    monkeypatch.setattr(llm_mod, "CACHE", disk)
    monkeypatch.setattr(llm_mod, "DRY_RUN", False)

    import requests
    if provider_name == "gemini":
        fake = FakeGeminiTransport()
        monkeypatch.setenv(config.GEMINI_API_KEY_ENV, "test-key-not-real")
        monkeypatch.setattr(requests, "post", fake)
        monkeypatch.setattr(llm_mod, "get_client", _forbidden("boto3"))
    else:
        fake = FakeBedrockClient()
        monkeypatch.setattr(llm_mod, "get_client", lambda: fake)
        monkeypatch.setattr(requests, "post", _forbidden("requests.post"))

    return fake, trk, disk


@pytest.fixture(params=PROVIDERS)
def provider(request, monkeypatch):
    """Select a provider without wiring a transport (for pure config tests)."""
    _select_provider(monkeypatch, request.param)
    return request.param
