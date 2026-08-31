"""Graph-level guarantees, exercised against a stub retriever and stub agents.

The point of these tests is the one property the spec calls non-negotiable:
MAX_ITERATIONS is enforced by the graph itself, so a broken gate cannot loop.
"""
from __future__ import annotations

import pytest

import config
import graph as graph_mod
from retrieval import Chunk


class StubRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, k=5, **kw):
        self.calls += 1
        # Fresh chunk ids each call so evidence genuinely grows.
        return [Chunk(chunk_id=f"c{self.calls}-{i}", title=f"T{i}",
                      text=f"passage {self.calls}-{i}", score=1.0 / (i + 1))
                for i in range(k)]


class RunawayPolicy:
    """A deliberately broken gate that always says 'retrieve'."""
    name = "runaway"

    def decide(self, state):
        return "retrieve"


@pytest.fixture()
def stubbed(monkeypatch):
    from agents import generator, planner, verifier

    retriever = StubRetriever()
    monkeypatch.setattr(graph_mod, "get_retriever", lambda: retriever)
    monkeypatch.setattr(planner, "plan",
                        lambda q, *a, **kw: f"{q} (refined)")
    monkeypatch.setattr(
        generator, "generate",
        lambda q, ev, **kw: f"answer from {len(ev)} chunks")

    coverages = iter([0.2, 0.4, 0.55, 0.65, 0.7, 0.72, 0.73, 0.74])

    def fake_verify(question, evidence, **kw):
        return verifier.Verification(
            coverage=next(coverages, 0.75), missing="more", confident=False)

    monkeypatch.setattr(verifier, "verify", fake_verify)
    return retriever


def test_max_iterations_is_enforced_against_a_broken_gate(stubbed):
    """The headline safety property: a gate bug cannot loop."""
    final = graph_mod.run_query("q?", RunawayPolicy(), query_id="runaway-1")
    assert final["iteration"] == config.MAX_ITERATIONS
    assert final["stop_reason"] == "max_iter"
    assert stubbed.calls == config.MAX_ITERATIONS


def test_fixed_policy_completes_end_to_end(stubbed):
    from policies import FixedPolicy

    final = graph_mod.run_query("q?", FixedPolicy(n=3), query_id="fx-1")
    assert final["iteration"] == 3
    assert final["stop_reason"] == "fixed"
    assert final["answer"].startswith("answer from")


def test_per_iteration_cost_and_coverage_are_logged_for_every_iteration(stubbed):
    from policies import FixedPolicy

    final = graph_mod.run_query("q?", FixedPolicy(n=3), query_id="fx-2")
    assert len(final["coverage_history"]) == 3
    assert len(final["cost_history"]) == 3
    assert len(final["latency_history"]) == 3


def test_evidence_is_deduplicated_across_iterations(monkeypatch, stubbed):
    from policies import FixedPolicy

    # Force the retriever to return the same chunks every time.
    fixed_hits = [Chunk(chunk_id="same", title="T", text="p", score=1.0)]
    monkeypatch.setattr(graph_mod, "get_retriever",
                        lambda: type("R", (), {"search": lambda s, *a, **k: fixed_hits})())

    final = graph_mod.run_query("q?", FixedPolicy(n=3), query_id="dedup-1")
    assert len(final["evidence"]) == 1, "duplicate chunks accumulated"


def test_state_summary_shape(stubbed):
    from policies import FixedPolicy

    final = graph_mod.run_query("q?", FixedPolicy(n=2), query_id="sum-1")
    s = graph_mod.state_summary(final)
    for field in ("query_id", "policy", "iterations_used", "stop_reason",
                  "total_usd", "total_latency_ms", "final_coverage", "answer"):
        assert field in s


def test_confidence_short_circuit_is_off_by_default(monkeypatch, stubbed):
    """Baselines must not inherit a stopping rule CAES has (or vice versa)."""
    from agents import verifier
    from policies import FixedPolicy

    monkeypatch.setattr(
        verifier, "verify",
        lambda q, ev, **kw: verifier.Verification(
            coverage=1.0, missing="nothing", confident=True))

    final = graph_mod.run_query("q?", FixedPolicy(n=3), query_id="conf-off")
    assert final["iteration"] == 3
    assert final["stop_reason"] == "fixed"


def test_confidence_short_circuit_works_when_enabled(monkeypatch, stubbed):
    from agents import verifier
    from policies import FixedPolicy

    monkeypatch.setattr(
        verifier, "verify",
        lambda q, ev, **kw: verifier.Verification(
            coverage=1.0, missing="nothing", confident=True))

    final = graph_mod.run_query("q?", FixedPolicy(n=3), query_id="conf-on",
                                honor_confidence=True)
    assert final["iteration"] == 1
    assert final["stop_reason"] == "confident"


def test_gold_titles_are_invisible_to_the_gate():
    """Instrumentation must not leak into the decision.

    `gold_recall` answers DECISIONS open question 3, but the supporting-fact
    titles are ground truth: a policy that could see them would be choosing its
    depth from the answer key, which would invalidate every result.
    """
    import inspect

    import graph

    for name in ("evaluate_gate", "route_from_state", "make_verify_node"):
        src = inspect.getsource(getattr(graph, name))
        assert "gold" not in src, (
            f"graph.{name} references gold titles; the gate must never see "
            f"ground truth")

    # And the recall figure itself is derivable without touching the policy.
    class Chunk:
        def __init__(self, title):
            self.title = title

    assert graph.gold_recall([Chunk("A"), Chunk("B")], ["A", "B"]) == 1.0
    assert graph.gold_recall([Chunk("A")], ["A", "B"]) == 0.5
    assert graph.gold_recall([Chunk("Z")], ["A", "B"]) == 0.0
    assert graph.gold_recall([Chunk("A")], None) == -1.0, \
        "an unlabelled run must be distinguishable from a failed retrieval"


def test_each_iteration_adds_new_evidence(monkeypatch):
    """An iteration that adds no evidence cannot move coverage.

    Retrieving exactly TOP_K and filtering seen chunks afterwards can return
    nothing new, which makes dQ structurally zero and every later iteration
    dead. Retrieval must search deep enough to yield TOP_K unseen chunks.
    """
    import config
    import graph
    from retrieval import Chunk

    class RankedRetriever:
        """Returns the same global ranking every time, as a real index would
        for a query whose planner output barely changed."""

        def __init__(self):
            self.depths = []

        def search(self, query, k=5, **kw):
            self.depths.append(k)
            return [Chunk(chunk_id=str(i), title=f"T{i}", text=f"body {i}",
                          score=1.0 - i / 100) for i in range(k)]

    r = RankedRetriever()
    monkeypatch.setattr(graph, "get_retriever", lambda: r)
    monkeypatch.setattr(config, "TOP_K", 2)

    state = graph.initial_state("q", query_id="q1")
    state["query"] = "q"
    for it in range(1, 4):
        state["iteration"] = it
        state.update(graph.node_retrieve(state))

    assert state["new_chunks_history"] == [2, 2, 2], (
        f"iterations added {state['new_chunks_history']} chunks; an iteration "
        f"adding 0 is dead and makes dQ structurally zero")
    assert len(state["evidence"]) == 6
    assert len(set(state["seen_chunk_ids"])) == 6, "duplicate chunks in evidence"
    assert r.depths == [2, 4, 6], (
        "search depth must grow with what has been seen, or the filter can "
        "return nothing new")


def test_run_driver_split_selection_is_explicit_and_namespaced():
    """A tune-split run must never write to the test-split artifact.

    The driver hardcoded test_set(), so an instruction to run baselines on the
    tune split silently produced test-split results that were then compared
    against a tune-split lambda sweep. See DECISIONS [D-29].
    """
    from experiments.run import raw_path

    assert raw_path("fixed", "test").name == "fixed_raw.jsonl"
    assert raw_path("fixed", "tune").name == "fixed_tune_raw.jsonl"
    assert raw_path("fixed", "test") != raw_path("fixed", "tune")

    # Diagnostic runs at different lambda must not collide with each other.
    assert raw_path("caes", "tune", 1000).name == "caes_tune_lam1000_raw.jsonl"
    assert raw_path("caes", "tune", 3000) != raw_path("caes", "tune", 1000)
    # A frozen config.LAMBDA (no override) keeps the plain name.
    assert raw_path("caes", "test", None).name == "caes_raw.jsonl"


def test_splits_are_disjoint_and_correctly_sized():
    from splits import test_set, tune_set
    import config

    t, s = tune_set(), test_set()
    assert len(t) == config.N_TUNE
    assert len(s) == config.N_TEST
    assert not ({q["id"] for q in t} & {q["id"] for q in s}), \
        "tune and test splits overlap; invariant 6"
