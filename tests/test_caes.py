"""Phase 4 acceptance: the gate engages, and cannot loop."""
from __future__ import annotations

import pytest

import config
from caes import (
    CAESPolicy,
    ThresholdPolicy,
    estimate_delta_c,
    estimate_delta_q,
    smooth_coverage,
)


def S(**kw):
    base = {"query_id": "q", "iteration": 1, "coverage_history": [],
            "cost_history": []}
    base.update(kw)
    return base


# --- estimators -----------------------------------------------------------

def test_delta_q_allows_one_more_when_history_is_too_short():
    assert estimate_delta_q([]) == 1.0
    assert estimate_delta_q([0.4]) == 1.0


def test_delta_q_extrapolates_with_decay():
    # last delta 0.2, decay 0.6 -> 0.12
    assert estimate_delta_q([0.3, 0.5]) == pytest.approx(0.2 * config.DECAY_FACTOR)


def test_delta_q_never_negative():
    assert estimate_delta_q([0.8, 0.5]) == 0.0


def test_delta_c_is_the_mean_of_observed_iterations():
    assert estimate_delta_c([0.001, 0.003]) == pytest.approx(0.002)
    assert estimate_delta_c([]) == 0.0


def test_smoothing_is_a_running_max():
    assert smooth_coverage([0.2, 0.6, 0.4, 0.5, 0.9]) == [0.2, 0.6, 0.6, 0.6, 0.9]


def test_smoothing_prevents_a_transient_dip_from_forcing_a_stop():
    """A dip must not read as negative gain and stop the loop prematurely."""
    raw = [0.3, 0.7, 0.5]          # new docs introduced ambiguity at it3
    assert estimate_delta_q(raw) == 0.0                    # raw would stop
    assert estimate_delta_q(smooth_coverage(raw)) == 0.0   # smoothed: flat, also stops
    # but the following iteration recovers and smoothing keeps the gain visible
    raw2 = raw + [0.8]
    assert estimate_delta_q(smooth_coverage(raw2)) > 0.0


# --- the policy -----------------------------------------------------------

def test_lambda_must_be_set_explicitly(monkeypatch):
    monkeypatch.setattr(config, "LAMBDA", None)
    with pytest.raises(ValueError, match="LAMBDA is unset"):
        CAESPolicy()


def test_hard_cap_beats_the_gate():
    p = CAESPolicy(lam=0.0, record=False)   # lam=0 => margin always positive
    state = S(iteration=config.MAX_ITERATIONS,
              coverage_history=[0.1] * 5, cost_history=[0.001] * 5)
    assert p.decide(state) == "generate"


def test_min_iterations_forces_at_least_one_retrieval():
    p = CAESPolicy(lam=1e12, record=False)  # lam huge => margin always negative
    assert p.decide(S(iteration=0)) == "retrieve"


def test_cheap_gain_continues_expensive_gain_stops():
    coverage = [0.3, 0.6]        # last delta 0.3 -> dQ = 0.18
    cost = [0.001, 0.001]        # dC = 0.001

    cheap = CAESPolicy(lam=10.0, record=False)     # 10 * 0.001 = 0.01 < 0.18
    assert cheap.decide(S(iteration=2, coverage_history=coverage,
                          cost_history=cost)) == "retrieve"

    dear = CAESPolicy(lam=1000.0, record=False)    # 1000 * 0.001 = 1.0 > 0.18
    assert dear.decide(S(iteration=2, coverage_history=coverage,
                         cost_history=cost)) == "generate"


def test_flat_coverage_stops_regardless_of_lambda():
    p = CAESPolicy(lam=0.001, record=False)
    assert p.decide(S(iteration=2, coverage_history=[0.7, 0.7],
                      cost_history=[0.001, 0.001])) == "generate"


def test_decision_log_contains_dq_dc_and_margin(tmp_path):
    import json

    from caes import DecisionLogger

    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    p = CAESPolicy(lam=100.0, logger=logger)
    p.decide(S(iteration=2, coverage_history=[0.2, 0.6],
               cost_history=[0.002, 0.002]))

    lines = (tmp_path / "decisions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for field in ("delta_q", "delta_c", "lambda_value",
                  "lambda_times_delta_c", "margin", "outcome",
                  "coverage_raw", "coverage_smoothed"):
        assert field in rec, f"decision log is missing {field}"
    assert rec["margin"] == pytest.approx(
        rec["delta_q"] - rec["lambda_times_delta_c"])


def test_iteration_counts_vary_across_queries():
    """If every query stops at the same iteration the gate is not engaging."""
    p = CAESPolicy(lam=100.0, record=False)
    trajectories = {
        "easy":  [0.95, 0.96],           # tiny gain -> stop
        "hard":  [0.10, 0.45],           # big gain  -> continue
        "medium": [0.40, 0.55],          # moderate
    }
    outcomes = {
        name: p.decide(S(query_id=name, iteration=2, coverage_history=cov,
                         cost_history=[0.001, 0.001]))
        for name, cov in trajectories.items()
    }
    assert len(set(outcomes.values())) > 1, \
        f"gate gave the same answer to every trajectory: {outcomes}"


# --- fallback policy ------------------------------------------------------

def test_threshold_policy_stops_on_small_delta_above_coverage_floor():
    p = ThresholdPolicy(record=False)
    assert p.decide(S(iteration=2, coverage_history=[0.72, 0.74])) == "generate"


def test_threshold_policy_continues_while_gaining():
    p = ThresholdPolicy(record=False)
    assert p.decide(S(iteration=2, coverage_history=[0.30, 0.60])) == "retrieve"


def test_threshold_policy_continues_below_the_coverage_floor():
    p = ThresholdPolicy(record=False)
    # flat, but coverage is too low to be worth stopping on
    assert p.decide(S(iteration=2, coverage_history=[0.40, 0.41])) == "retrieve"
