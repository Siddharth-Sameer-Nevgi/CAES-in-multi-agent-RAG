"""Phase 0 acceptance: cache hits are identical and free."""
from __future__ import annotations

import json

import pytest

import bedrock
import cache as cache_mod
from costs import CostTracker
from tests.test_costs import FakeClient


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    """A bedrock module wired to a fake client, fresh cache, fresh ledger."""
    fake = FakeClient()
    tracker = CostTracker(ledger_path=tmp_path / "ledger.json",
                          hard_budget_usd=10.0)
    disk = cache_mod.DiskCache(tmp_path / "cache")
    monkeypatch.setattr(bedrock, "get_client", lambda: fake)
    monkeypatch.setattr(bedrock, "TRACKER", tracker)
    monkeypatch.setattr(bedrock, "CACHE", disk)
    monkeypatch.setattr(bedrock, "DRY_RUN", False)
    return fake, tracker, disk


def test_key_is_stable_under_dict_ordering():
    a = cache_mod.make_key("m", {"x": 1, "y": [1, 2], "z": "s"})
    b = cache_mod.make_key("m", {"z": "s", "y": [1, 2], "x": 1})
    assert a == b


def test_key_changes_with_model():
    payload = {"x": 1}
    assert cache_mod.make_key("m1", payload) != cache_mod.make_key("m2", payload)


def test_hit_returns_identical_content_and_costs_nothing(wired):
    fake, tracker, disk = wired

    first = bedrock.invoke_llm("what colour is the sky?", call_type="generate",
                               query_id="q1")
    cost_after_first = tracker.cumulative()
    assert first.cached is False
    assert fake.calls == 1
    assert cost_after_first > 0

    second = bedrock.invoke_llm("what colour is the sky?", call_type="generate",
                                query_id="q1")
    assert second.cached is True
    assert second.text == first.text
    assert second.input_tokens == first.input_tokens
    assert second.output_tokens == first.output_tokens
    assert second.usd == 0.0
    assert fake.calls == 1, "cache hit still hit the network"
    assert tracker.cumulative() == pytest.approx(cost_after_first), \
        "cache hit recorded additional cost"

    assert disk.stats()["hits"] == 1
    assert disk.stats()["misses"] == 1


def test_different_prompt_misses(wired):
    fake, _, _ = wired
    bedrock.invoke_llm("a", call_type="generate")
    bedrock.invoke_llm("b", call_type="generate")
    assert fake.calls == 2


def test_system_prompt_participates_in_the_key(wired):
    fake, _, _ = wired
    bedrock.invoke_llm("a", call_type="generate", system="be terse")
    bedrock.invoke_llm("a", call_type="generate", system="be verbose")
    assert fake.calls == 2


def test_corrupt_entry_is_treated_as_a_miss(tmp_path):
    disk = cache_mod.DiskCache(tmp_path / "cache")
    disk.set("deadbeef" * 8, {"ok": True})
    path = disk._path("deadbeef" * 8)
    path.write_text("{not json", encoding="utf-8")
    assert disk.get("deadbeef" * 8) is None
    assert not path.exists()


def test_disabled_cache_never_returns(tmp_path):
    disk = cache_mod.DiskCache(tmp_path / "cache", enabled=False)
    disk.set("k" * 64, {"v": 1})
    assert disk.get("k" * 64) is None
