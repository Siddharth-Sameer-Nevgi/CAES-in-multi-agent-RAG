"""Phase 3 acceptance: the baselines are faithful representations."""
from __future__ import annotations

import pytest

import config
from policies import (
    FixedPolicy,
    OneShotPolicy,
    build_policy,
    complexity_score,
    score_to_depth,
)


def S(**kw):
    base = {"query_id": "q", "question": "Who directed Inception?",
            "iteration": 0, "coverage_history": [], "cost_history": []}
    base.update(kw)
    return base


# --- B1: fixed ------------------------------------------------------------

def test_fixed_runs_exactly_n_iterations():
    p = FixedPolicy(n=3)
    assert [p.decide(S(iteration=i)) for i in range(5)] == [
        "retrieve", "retrieve", "retrieve", "generate", "generate"]


def test_fixed_ignores_the_verifier_entirely():
    p = FixedPolicy(n=3)
    # Perfect coverage after one iteration: a fixed policy must not care.
    assert p.decide(S(iteration=1, coverage_history=[1.0])) == "retrieve"


def test_fixed_rejects_out_of_range_n():
    with pytest.raises(ValueError):
        FixedPolicy(n=0)
    with pytest.raises(ValueError):
        FixedPolicy(n=config.MAX_ITERATIONS + 1)


# --- B2: one-shot routing -------------------------------------------------

def test_complexity_score_is_bounded_and_ordered():
    simple = "Who wrote Hamlet?"
    complex_q = ("The director of the 1994 film Speed, which starred Keanu "
                 "Reeves, also directed which 1996 disaster film featuring "
                 "Helen Hunt?")
    for q in (simple, complex_q):
        assert 0.0 <= complexity_score(q) <= 1.0
    assert complexity_score(simple) < complexity_score(complex_q)


def test_score_to_depth_is_monotone_and_clamped():
    depths = [score_to_depth(s) for s in (0.0, 0.35, 0.55, 0.9)]
    assert depths == sorted(depths)
    assert all(config.MIN_ITERATIONS <= d <= config.MAX_ITERATIONS
               for d in depths)


def test_oneshot_commits_before_iteration_one():
    """Depth is chosen up front and never revisited, whatever the verifier says."""
    p = OneShotPolicy()
    state = S(query_id="q1", question="Who wrote Hamlet?")
    depth = p.depth_for(state)

    # Feed it a wildly contradictory verifier signal mid-run.
    for i in range(depth):
        d = p.decide(S(query_id="q1", question="Who wrote Hamlet?",
                       iteration=i, coverage_history=[0.01] * (i + 1)))
        assert d == "retrieve", "one-shot changed its mind mid-run"
    assert p.decide(S(query_id="q1", question="Who wrote Hamlet?",
                      iteration=depth, coverage_history=[0.01] * depth)) \
        == "generate"


def test_oneshot_routes_different_questions_to_different_depths():
    p = OneShotPolicy()
    simple = S(query_id="a", question="Who wrote Hamlet?")
    hard = S(query_id="b", question=(
        "The director of the 1994 film Speed, which starred Keanu Reeves, "
        "also directed which 1996 disaster film featuring Helen Hunt?"))
    assert p.depth_for(simple) < p.depth_for(hard), \
        "one-shot routing collapsed to a constant depth"


def test_oneshot_decision_is_stable_across_repeated_calls():
    p = OneShotPolicy()
    state = S(query_id="q1")
    assert len({p.depth_for(state) for _ in range(5)}) == 1


# --- factory --------------------------------------------------------------

def test_build_policy_returns_the_right_types():
    assert build_policy("fixed", n=2).name == "fixed"
    assert build_policy("oneshot").name == "oneshot"
    assert build_policy("caes", lam=100.0).name == "caes"
    assert build_policy("threshold").name == "threshold"


def test_build_policy_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown policy"):
        build_policy("magic")
