"""Phase 0 acceptance: the spend guards actually guard.

Amounts are derived from `config` rather than written as literals, so the suite
means the same thing under either provider's price table. The literal published
prices are pinned separately, in test_provider.py.
"""
from __future__ import annotations

import json

import pytest

import cache as cache_mod
import config
import llm
from costs import BudgetExceeded, CostTracker
from tests.conftest import FakeBedrockClient, _Body


def _tokens_worth(usd: float) -> int:
    """Input-token count costing roughly `usd` at the active price table."""
    return int(round(usd * 1000.0 / config.PRICE_LLM_INPUT_PER_1K))


def test_estimate_applies_input_and_output_prices_to_the_right_side(tracker):
    """Catches the classic swap, without restating the price table."""
    assert tracker.estimate_llm_cost(1000, 0) == \
        pytest.approx(config.PRICE_LLM_INPUT_PER_1K)
    assert tracker.estimate_llm_cost(0, 1000) == \
        pytest.approx(config.PRICE_LLM_OUTPUT_PER_1K)
    assert tracker.estimate_llm_cost(1000, 1000) == pytest.approx(
        config.PRICE_LLM_INPUT_PER_1K + config.PRICE_LLM_OUTPUT_PER_1K)
    assert config.PRICE_LLM_OUTPUT_PER_1K > config.PRICE_LLM_INPUT_PER_1K


def test_record_accumulates_and_persists(tracker):
    expected = config.PRICE_LLM_INPUT_PER_1K + config.PRICE_LLM_OUTPUT_PER_1K
    tracker.record_llm(call_type="verify", in_tokens=1000, out_tokens=1000,
                       latency_ms=5.0, query_id="q1", iteration=1)
    assert tracker.cumulative() == pytest.approx(expected)
    raw = json.loads(tracker.ledger_path.read_text())
    assert raw["cumulative_usd"] == pytest.approx(expected)
    assert raw["records"][0]["call_type"] == "verify"


def test_ledger_survives_process_restart(tmp_path):
    """Write, drop the object (simulating a crash), reload, assert the total."""
    path = tmp_path / "ledger.json"
    t1 = CostTracker(ledger_path=path, hard_budget_usd=10.0)
    for _ in range(3):
        t1.record_llm(call_type="generate", in_tokens=1000, out_tokens=200,
                      latency_ms=1.0)
    before = t1.cumulative()
    del t1

    t2 = CostTracker(ledger_path=path, hard_budget_usd=10.0)
    assert t2.cumulative() == pytest.approx(before)
    assert len(t2.records()) == 3


def test_check_affordable_raises_at_ceiling(tracker):
    # Park spend exactly at the $1.00 ceiling the fixture sets.
    tracker.record_llm(call_type="generate", in_tokens=_tokens_worth(1.00),
                       out_tokens=0, latency_ms=1.0)
    with pytest.raises(BudgetExceeded, match="hard ceiling"):
        tracker.check_affordable(0.01)


def test_budget_exception_fires_before_the_api_call(wired, monkeypatch):
    """The exception must be raised BEFORE the call, not after paying for it.

    Runs under both providers: the pre-flight check is shared code, but a
    provider path that built its request before consulting the guard would
    still show up here as a non-zero call count.
    """
    fake, tracker, _ = wired
    monkeypatch.setattr(tracker, "hard_budget_usd", 1.00)

    # Park spend right at the ceiling.
    tracker.record_llm(call_type="generate", in_tokens=_tokens_worth(1.00),
                       out_tokens=0, latency_ms=1.0)

    with pytest.raises(BudgetExceeded):
        llm.invoke_llm("hello", call_type="verify", max_tokens=512)

    assert fake.calls == 0, \
        "the provider was reached despite the budget being blown"


def test_run_budget_bounds_a_single_invocation(tracker):
    with pytest.raises(BudgetExceeded, match="allowance"):
        with tracker.run_budget(0.0000001, name="tiny"):
            tracker.check_affordable(tracker.estimate_llm_cost(1000, 1000))


def test_run_budget_releases_on_exit(tracker):
    with tracker.run_budget(0.002, name="tiny"):
        pass
    # Outside the block only the hard ceiling applies.
    tracker.check_affordable(0.5)


def test_missing_token_counts_are_fatal_on_both_providers(wired, monkeypatch):
    """[D-2]: never silently estimate cost post-hoc; measured dC is a claim."""
    fake, tracker, disk = wired

    if config.PROVIDER == "gemini":
        fake.omit_usage = True
    else:
        class NoUsageClient(FakeBedrockClient):
            def invoke_model(self, **kwargs):
                self.calls += 1
                return {"body": _Body(json.dumps(
                    {"content": [{"type": "text", "text": "x"}]}))}

        import llm as llm_mod
        replacement = NoUsageClient()
        monkeypatch.setattr(llm_mod, "get_client", lambda: replacement)

    with pytest.raises(RuntimeError, match="(?i)usage|token"):
        llm.invoke_llm("hello", call_type="verify")


def test_summary_groups_by_call_type(tracker):
    tracker.record_llm(call_type="verify", in_tokens=100, out_tokens=10,
                       latency_ms=1.0)
    tracker.record_llm(call_type="verify", in_tokens=100, out_tokens=10,
                       latency_ms=1.0)
    tracker.record_embed(in_tokens=500, latency_ms=1.0)
    s = tracker.summary()
    assert s["n_calls"] == 3
    assert s["by_call_type"]["verify"]["calls"] == 2
    assert s["by_call_type"]["embed"]["calls"] == 1
