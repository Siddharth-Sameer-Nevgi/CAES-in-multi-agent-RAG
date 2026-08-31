"""Phase 0 acceptance: cache hits are identical and free.

Every transport-level test here runs once per provider via the `wired` fixture.
"""
from __future__ import annotations

import pytest

import cache as cache_mod
import llm


def test_key_is_stable_under_dict_ordering():
    a = cache_mod.make_key("m", {"x": 1, "y": [1, 2], "z": "s"})
    b = cache_mod.make_key("m", {"z": "s", "y": [1, 2], "x": 1})
    assert a == b


def test_key_changes_with_model():
    payload = {"x": 1}
    assert cache_mod.make_key("m1", payload) != cache_mod.make_key("m2", payload)


def test_hit_returns_identical_content_and_costs_nothing(wired):
    fake, tracker, disk = wired

    first = llm.invoke_llm("what colour is the sky?", call_type="generate",
                           query_id="q1")
    cost_after_first = tracker.cumulative()
    assert first.cached is False
    assert fake.calls == 1
    assert cost_after_first > 0

    second = llm.invoke_llm("what colour is the sky?", call_type="generate",
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


def test_cache_hit_still_accrues_notional_cost(wired):
    """[D-12]: break this and the gate changes its mind on a warm cache."""
    _, _, _ = wired
    before = llm.totals()["notional_usd"]
    first = llm.invoke_llm("notional probe", call_type="verify")
    after_first = llm.totals()["notional_usd"]
    second = llm.invoke_llm("notional probe", call_type="verify")
    after_second = llm.totals()["notional_usd"]

    assert second.cached is True
    assert second.usd == 0.0, "a replayed call must not touch the ledger"
    assert second.notional_usd == pytest.approx(first.notional_usd)
    assert after_second - after_first == pytest.approx(after_first - before), \
        "a cache hit accrued a different notional cost than the paid call"


def test_different_prompt_misses(wired):
    fake, _, _ = wired
    llm.invoke_llm("a", call_type="generate")
    llm.invoke_llm("b", call_type="generate")
    assert fake.calls == 2


def test_system_prompt_participates_in_the_key(wired):
    fake, _, _ = wired
    llm.invoke_llm("a", call_type="generate", system="be terse")
    llm.invoke_llm("a", call_type="generate", system="be verbose")
    assert fake.calls == 2


def test_temperature_participates_in_the_key(wired):
    """[D-19]: the key is the request body, so sampling changes must miss."""
    fake, _, _ = wired
    llm.invoke_llm("a", call_type="generate", temperature=0.0)
    llm.invoke_llm("a", call_type="generate", temperature=1.0)
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
