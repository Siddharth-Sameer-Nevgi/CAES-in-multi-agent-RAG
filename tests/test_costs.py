"""Phase 0 acceptance: the spend guards actually guard."""
from __future__ import annotations

import json

import pytest

import bedrock
import cache as cache_mod
import config
from costs import BudgetExceeded, CostTracker


@pytest.fixture()
def tracker(tmp_path):
    return CostTracker(ledger_path=tmp_path / "ledger.json",
                       hard_budget_usd=1.00, warn_budget_usd=0.50)


class FakeClient:
    """Stands in for bedrock-runtime. Records whether it was ever reached."""

    def __init__(self, in_tokens=100, out_tokens=50):
        self.calls = 0
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens

    def invoke_model(self, **kwargs):
        self.calls += 1
        payload = {
            "content": [{"type": "text", "text": "fake answer"}],
            "usage": {"input_tokens": self.in_tokens,
                      "output_tokens": self.out_tokens},
        }
        return {"body": _Body(json.dumps(payload))}


class _Body:
    def __init__(self, s):
        self._s = s

    def read(self):
        return self._s


def test_estimate_matches_price_table(tracker):
    # 1000 in, 1000 out => $0.001 + $0.005
    assert tracker.estimate_llm_cost(1000, 1000) == pytest.approx(0.006)


def test_record_accumulates_and_persists(tracker):
    tracker.record_llm(call_type="verify", in_tokens=1000, out_tokens=1000,
                       latency_ms=5.0, query_id="q1", iteration=1)
    assert tracker.cumulative() == pytest.approx(0.006)
    raw = json.loads(tracker.ledger_path.read_text())
    assert raw["cumulative_usd"] == pytest.approx(0.006)
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
    tracker.record_llm(call_type="generate", in_tokens=1_000_000,
                       out_tokens=0, latency_ms=1.0)   # $1.00, at the ceiling
    with pytest.raises(BudgetExceeded, match="hard ceiling"):
        tracker.check_affordable(0.01)


def test_budget_exception_fires_before_the_api_call(monkeypatch, tracker, tmp_path):
    """The exception must be raised BEFORE the call, not after paying for it."""
    fake = FakeClient()
    monkeypatch.setattr(bedrock, "get_client", lambda: fake)
    monkeypatch.setattr(bedrock, "TRACKER", tracker)
    monkeypatch.setattr(bedrock, "DRY_RUN", False)
    monkeypatch.setattr(bedrock, "CACHE",
                        cache_mod.DiskCache(tmp_path / "cache"))

    # Park spend right at the ceiling.
    tracker.record_llm(call_type="generate", in_tokens=1_000_000,
                       out_tokens=0, latency_ms=1.0)

    with pytest.raises(BudgetExceeded):
        bedrock.invoke_llm("hello", call_type="verify", max_tokens=512)

    assert fake.calls == 0, "Bedrock was reached despite the budget being blown"


def test_run_budget_bounds_a_single_invocation(tracker):
    with pytest.raises(BudgetExceeded, match="allowance"):
        with tracker.run_budget(0.002, name="tiny"):
            # $0.006 of estimated spend against a $0.002 allowance.
            tracker.check_affordable(tracker.estimate_llm_cost(1000, 1000))


def test_run_budget_releases_on_exit(tracker):
    with tracker.run_budget(0.002, name="tiny"):
        pass
    # Outside the block only the hard ceiling applies.
    tracker.check_affordable(0.5)


def test_missing_usage_block_is_fatal(monkeypatch, tracker, tmp_path):
    """Never silently estimate cost post-hoc; measured dC is a core claim."""
    class NoUsageClient(FakeClient):
        def invoke_model(self, **kwargs):
            self.calls += 1
            return {"body": _Body(json.dumps(
                {"content": [{"type": "text", "text": "x"}]}))}

    monkeypatch.setattr(bedrock, "get_client", lambda: NoUsageClient())
    monkeypatch.setattr(bedrock, "TRACKER", tracker)
    monkeypatch.setattr(bedrock, "DRY_RUN", False)
    monkeypatch.setattr(bedrock, "CACHE",
                        cache_mod.DiskCache(tmp_path / "cache"))

    with pytest.raises(RuntimeError, match="usage"):
        bedrock.invoke_llm("hello", call_type="verify")


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
