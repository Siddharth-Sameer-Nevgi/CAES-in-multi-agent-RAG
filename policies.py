"""Three interchangeable retrieval-depth gates behind one interface.

  FixedPolicy    B1  always N iterations
  OneShotPolicy  B2  CA-RAG style: depth chosen up front from a complexity
                     score, then committed regardless of what the verifier says
  CAESPolicy         the contribution (defined in caes.py)
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Protocol, runtime_checkable

import config
from caes import CAESPolicy, ThresholdPolicy  # noqa: F401  (re-exported)

log = logging.getLogger("caes.policies")

Decision = Literal["retrieve", "generate"]


@runtime_checkable
class Policy(Protocol):
    name: str

    def decide(self, state) -> Decision: ...


class FixedPolicy:
    """B1: always run exactly n iterations, whatever the verifier reports."""

    name = "fixed"

    def __init__(self, n: int = 3) -> None:
        if not 1 <= n <= config.MAX_ITERATIONS:
            raise ValueError(
                f"FixedPolicy n={n} must be within 1..{config.MAX_ITERATIONS}")
        self.n = n

    def decide(self, state) -> Decision:
        iteration = state.get("iteration", 0)
        if iteration >= min(self.n, config.MAX_ITERATIONS):
            return "generate"
        return "retrieve"


# ---------------------------------------------------------------------------
# One-shot routing (B2)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "which", "who", "what", "when", "where", "how",
    "did", "does", "do", "that", "this", "with", "by", "from", "as", "it",
}

_MULTIHOP_MARKERS = (
    "also", "both", "same", "which one", "who is older", "either",
    "as well as", "in common", "respectively",
)


def complexity_score(question: str) -> float:
    """Cheap, deterministic complexity estimate in [0, 1].

    Deliberately does not consult the verifier or any retrieved evidence: the
    whole point of the baseline is that depth is chosen with no feedback.
    """
    words = question.split()
    length_term = min(1.0, len(words) / 30.0)

    # Capitalised tokens not at sentence start, as a proxy for named entities.
    entities = {
        w.strip(".,?;:'\"") for i, w in enumerate(words)
        if i > 0 and w[:1].isupper() and w.strip(".,?;:'\"").lower() not in _STOPWORDS
    }
    entity_term = min(1.0, len(entities) / 4.0)

    lowered = question.lower()
    marker_term = 1.0 if any(m in lowered for m in _MULTIHOP_MARKERS) else 0.0

    clause_term = min(1.0, len(re.findall(r"[,;]|\bwho\b|\bwhich\b|\bthat\b",
                                          lowered)) / 3.0)

    return (0.30 * length_term + 0.35 * entity_term
            + 0.20 * marker_term + 0.15 * clause_term)


def score_to_depth(score: float, max_depth: int = config.MAX_ITERATIONS) -> int:
    """Map a complexity score to a committed iteration count."""
    if score < 0.30:
        depth = 1
    elif score < 0.50:
        depth = 2
    elif score < 0.70:
        depth = 3
    else:
        depth = 4
    return max(config.MIN_ITERATIONS, min(depth, max_depth))


class OneShotPolicy:
    """B2: choose depth BEFORE iteration 1, then commit.

    The fidelity of this baseline is what makes the three-way comparison
    credible, so the decision is made once, from information available before
    any retrieval, and is never revisited.
    """

    name = "oneshot"

    def __init__(self, max_depth: int = config.MAX_ITERATIONS) -> None:
        self.max_depth = max_depth
        self._committed: dict[str, int] = {}

    def depth_for(self, state) -> int:
        query_id = state.get("query_id", "") or state.get("question", "")
        if query_id not in self._committed:
            score = complexity_score(state["question"])
            depth = score_to_depth(score, self.max_depth)
            self._committed[query_id] = depth
            log.debug("[%s] one-shot route: complexity=%.3f -> depth=%d",
                      query_id, score, depth)
        return self._committed[query_id]

    def decide(self, state) -> Decision:
        iteration = state.get("iteration", 0)
        depth = self.depth_for(state)
        if iteration >= min(depth, config.MAX_ITERATIONS):
            return "generate"
        return "retrieve"


# ---------------------------------------------------------------------------

def build_policy(name: str, **kwargs) -> Policy:
    """Factory used by the experiment driver and the API."""
    name = name.lower()
    if name == "fixed":
        return FixedPolicy(n=kwargs.get("n", 3))
    if name == "oneshot":
        return OneShotPolicy(max_depth=kwargs.get("max_depth",
                                                  config.MAX_ITERATIONS))
    if name == "caes":
        return CAESPolicy(lam=kwargs.get("lam"))
    if name == "threshold":
        return ThresholdPolicy()
    raise ValueError(
        f"Unknown policy {name!r}. Choose from: fixed, oneshot, caes, threshold.")
