"""LangGraph state machine:  plan -> retrieve -> verify -> [GATE] -> generate

The gate is a conditional edge returning "retrieve" or "generate".

MAX_ITERATIONS is enforced inside the graph itself, checked before the active
policy is consulted at all. A gate bug must be incapable of looping.

Note on where the gate runs: the decision is computed at the end of `verify`
and written into state as `_route` / `stop_reason`; the conditional edge is a
pure read of that field. A conditional-edge function that tried to record its
reasoning by mutating state would lose it — LangGraph does not merge writes made
inside an edge back into the graph state.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, TypedDict

import config
from retrieval import get_retriever

log = logging.getLogger("caes.graph")


class State(TypedDict, total=False):
    # --- spec fields ---
    question: str
    query_id: str
    iteration: int
    evidence: list                 # list[Chunk]
    coverage_history: list[float]  # raw verifier coverage, per iteration
    cost_history: list[float]      # USD per retrieval iteration (notional)
    missing: str | None
    answer: str | None
    stop_reason: str               # "caes" | "max_iter" | "fixed" | "confident"

    # --- bookkeeping ---
    policy: str
    query: str                     # retrieval query for the current iteration
    seen_chunk_ids: list[str]
    latency_history: list[float]
    confident: bool
    parse_failures: int
    # --- instrumentation only, never read by the gate ---
    gold_titles: list[str]          # HotpotQA supporting_facts titles
    gold_recall_history: list[float]
    total_usd: float
    total_latency_ms: float
    _route: str                    # "retrieve" | "generate", set by verify
    _iter_usd_mark: float
    _iter_latency_mark: float
    _query_usd_mark: float
    _query_latency_mark: float


def initial_state(question: str, query_id: str = "", policy: str = "",
                  gold_titles: list[str] | None = None) -> State:
    import llm
    t = llm.totals()
    return State(
        question=question,
        query_id=query_id or f"q-{uuid.uuid4().hex[:8]}",
        iteration=0,
        evidence=[],
        coverage_history=[],
        cost_history=[],
        missing=None,
        answer=None,
        stop_reason="",
        policy=policy,
        query=question,
        seen_chunk_ids=[],
        latency_history=[],
        confident=False,
        parse_failures=0,
        gold_titles=list(gold_titles or []),
        gold_recall_history=[],
        total_usd=0.0,
        total_latency_ms=0.0,
        _route="",
        _query_usd_mark=t["notional_usd"],
        _query_latency_mark=t["latency_ms"],
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_plan(state: State) -> dict[str, Any]:
    import llm
    from agents.planner import plan

    iteration = state["iteration"] + 1
    t = llm.totals()
    query = plan(
        state["question"],
        prior_evidence=state["evidence"],
        missing=state.get("missing"),
        query_id=state["query_id"],
        iteration=iteration,
        policy=state.get("policy", ""),
    )
    log.debug("[%s it%d] query: %s", state["query_id"], iteration, query)
    return {
        "iteration": iteration,
        "query": query,
        "_iter_usd_mark": t["notional_usd"],
        "_iter_latency_mark": t["latency_ms"],
    }


def node_retrieve(state: State) -> dict[str, Any]:
    hits = get_retriever().search(
        state["query"], k=config.TOP_K, query_id=state["query_id"],
        iteration=state["iteration"], policy=state.get("policy", ""),
    )
    seen = set(state["seen_chunk_ids"])
    fresh = [c for c in hits if c.chunk_id not in seen]
    evidence = state["evidence"] + fresh
    log.debug("[%s it%d] retrieved %d hits, %d new",
              state["query_id"], state["iteration"], len(hits), len(fresh))
    return {
        "evidence": evidence,
        "seen_chunk_ids": state["seen_chunk_ids"] + [c.chunk_id for c in fresh],
        "gold_recall_history": (state.get("gold_recall_history", [])
                                + [gold_recall(evidence, state.get("gold_titles"))]),
    }


def gold_recall(evidence: list, gold_titles) -> float:
    """Fraction of the question's supporting passages now in evidence.

    **Instrumentation only. The gate must never read this.** It is the answer
    to DECISIONS open question 3: without it, a low F1 is ambiguous between
    "the gate stopped too early" and "retrieval never surfaced the passage",
    and those call for opposite fixes. Recording it per iteration separates
    them directly.

    Returns -1.0 when no gold titles are known, so an unlabelled run is
    distinguishable from a run that retrieved nothing.
    """
    if not gold_titles:
        return -1.0
    have = {getattr(c, "title", None) or c.get("title", "") for c in evidence}
    return sum(1 for t in gold_titles if t in have) / len(gold_titles)


def make_verify_node(policy, honor_confidence: bool = False) -> Callable:
    """Verify, then evaluate the gate and record its decision into state.

    `honor_confidence` adds a universal short-circuit when the verifier reports
    high coverage AND confidence. It is OFF by default and stays off for the
    headline three-way comparison: giving CAES a second, orthogonal stopping
    rule the baselines lack would confound the contribution. It exists because
    "confident" is part of the stop_reason vocabulary and is useful for the API
    layer, where latency matters more than attribution.
    """

    def node_verify(state: State) -> dict[str, Any]:
        import llm
        from agents.verifier import verify

        prev = state["coverage_history"][-1] if state["coverage_history"] else 0.0
        v = verify(
            state["question"], state["evidence"], query_id=state["query_id"],
            iteration=state["iteration"], policy=state.get("policy", ""),
            previous_coverage=prev,
        )
        t = llm.totals()
        iter_usd = t["notional_usd"] - state["_iter_usd_mark"]
        iter_latency = t["latency_ms"] - state["_iter_latency_mark"]

        updates: dict[str, Any] = {
            "coverage_history": state["coverage_history"] + [v.coverage],
            "cost_history": state["cost_history"] + [iter_usd],
            "latency_history": state["latency_history"] + [iter_latency],
            "missing": v.missing,
            "confident": v.confident,
            "parse_failures": state["parse_failures"] + int(v.parse_failed),
        }

        # The gate sees the post-verification view of the world.
        merged: dict[str, Any] = {**state, **updates}
        route, reason = evaluate_gate(merged, policy, honor_confidence)
        updates["_route"] = route
        updates["stop_reason"] = reason if route == "generate" else ""

        log.debug("[%s it%d] coverage=%.2f cost=$%.5f -> %s (%s)",
                  state["query_id"], state["iteration"], v.coverage, iter_usd,
                  route, reason)
        return updates

    return node_verify


def node_generate(state: State) -> dict[str, Any]:
    import llm
    from agents.generator import generate

    answer = generate(
        state["question"], state["evidence"], query_id=state["query_id"],
        iteration=state["iteration"], policy=state.get("policy", ""),
    )
    t = llm.totals()
    return {
        "answer": answer,
        "total_usd": t["notional_usd"] - state["_query_usd_mark"],
        "total_latency_ms": t["latency_ms"] - state["_query_latency_mark"],
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def evaluate_gate(state, policy, honor_confidence: bool = False) -> tuple[str, str]:
    """Return (route, reason). The only place a stop decision is made."""
    # Hard cap, checked BEFORE the policy. Independent of gate correctness.
    if state["iteration"] >= config.MAX_ITERATIONS:
        return "generate", "max_iter"

    if honor_confidence and state.get("confident") and \
            state.get("coverage_history") and state["coverage_history"][-1] >= 0.9:
        return "generate", "confident"

    if policy.decide(state) == "generate":
        return "generate", getattr(policy, "name", "policy")
    return "retrieve", ""


def route_from_state(state: State) -> str:
    """Conditional edge: a pure read of the decision `verify` already made."""
    return state.get("_route") or "generate"


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def build_graph(policy, honor_confidence: bool = False):
    """Compile the LangGraph state machine for a given policy."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(State)
    g.add_node("plan", node_plan)
    g.add_node("retrieve", node_retrieve)
    g.add_node("verify", make_verify_node(policy, honor_confidence))
    g.add_node("generate", node_generate)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_conditional_edges("verify", route_from_state,
                            {"retrieve": "plan", "generate": "generate"})
    g.add_edge("generate", END)
    return g.compile()


def _run_manual(state: State, policy, honor_confidence: bool) -> State:
    """Fallback executor with identical semantics, used when LangGraph is absent.

    Same nodes, same order, same hard cap. Exists so the pipeline stays testable
    and runnable without the extra dependency; the compiled graph is the
    production path.
    """
    verify_node = make_verify_node(policy, honor_confidence)
    while True:
        state.update(node_plan(state))
        state.update(node_retrieve(state))
        state.update(verify_node(state))
        if route_from_state(state) == "generate":
            break
        if state["iteration"] >= config.MAX_ITERATIONS:   # belt and braces
            state["stop_reason"] = "max_iter"
            break
    state.update(node_generate(state))
    return state


def _compiled_for(policy, honor_confidence: bool):
    """Compile once per policy object; recompiling per query is wasteful."""
    attr = f"_caes_graph_{int(honor_confidence)}"
    cached = getattr(policy, attr, "missing")
    if cached == "missing":
        try:
            cached = build_graph(policy, honor_confidence=honor_confidence)
        except ImportError:
            log.warning("LangGraph not installed; using the equivalent manual "
                        "executor. `pip install langgraph` for the real thing.")
            cached = None
        try:
            setattr(policy, attr, cached)
        except AttributeError:                      # slotted policy object
            pass
    return cached


def run_query(
    question: str,
    policy,
    *,
    query_id: str = "",
    honor_confidence: bool = False,
    gold_titles: list[str] | None = None,
) -> State:
    """Run one question end to end and return the final state.

    `gold_titles` is recorded for retrieval diagnostics only and is never
    visible to the policy or the gate -- see `gold_recall`.
    """
    state = initial_state(question, query_id=query_id,
                          policy=getattr(policy, "name", "unknown"),
                          gold_titles=gold_titles)
    t0 = time.perf_counter()

    compiled = _compiled_for(policy, honor_confidence)
    if compiled is None:
        final = _run_manual(state, policy, honor_confidence)
    else:
        final = compiled.invoke(
            state, config={"recursion_limit": 4 * config.MAX_ITERATIONS + 10})

    final["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    if not final.get("stop_reason"):
        # Only reachable if the graph exited without passing through the gate.
        final["stop_reason"] = "max_iter"
    return final


def state_summary(state: State) -> dict[str, Any]:
    """Flatten a final state into the per-query record the experiments log."""
    return {
        "query_id": state["query_id"],
        "question": state["question"],
        "policy": state.get("policy", ""),
        "iterations_used": state["iteration"],
        "stop_reason": state["stop_reason"],
        "total_usd": round(state.get("total_usd", 0.0), 8),
        "total_latency_ms": round(state.get("total_latency_ms", 0.0), 2),
        "wall_ms": round(state.get("wall_ms", 0.0), 2),
        "final_coverage": (state["coverage_history"][-1]
                           if state["coverage_history"] else 0.0),
        "coverage_history": state["coverage_history"],
        "cost_history": [round(c, 8) for c in state["cost_history"]],
        # Per-iteration latency, alongside the per-iteration cost and coverage
        # series. Needed to publish latency per iteration to CloudWatch, and
        # useful for the same reason the other two series are recorded.
        "latency_history": [round(l, 2) for l in state.get("latency_history", [])],
        "n_evidence": len(state["evidence"]),
        "parse_failures": state.get("parse_failures", 0),
        "gold_recall_history": state.get("gold_recall_history", []),
        "final_gold_recall": (state["gold_recall_history"][-1]
                              if state.get("gold_recall_history") else -1.0),
        "answer": state.get("answer") or "",
    }
