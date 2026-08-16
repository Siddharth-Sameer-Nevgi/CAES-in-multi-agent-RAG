"""The Cost-Aware Evidence Sufficiency gate.

Decision rule:  continue retrieving only while  dQ - lambda * dC > 0

  dQ  estimated marginal evidence-quality gain of the NEXT iteration,
      extrapolated from the observed coverage trajectory.
  dC  measured marginal execution cost of the next iteration, taken from the
      real per-iteration spend already observed for this query.

Every decision is logged with dQ, dC, lambda*dC, the margin, and the outcome.
That log is the source of the paper's central figure.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path

import config

log = logging.getLogger("caes.gate")

DECISION_LOG_PATH = config.RESULTS_DIR / "caes_decisions.jsonl"


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def smooth_coverage(coverage_history: list[float]) -> list[float]:
    """Running max of the raw coverage trajectory.

    Coverage genuinely decreases sometimes: a new document introduces a second
    plausible entity and the verifier rightly gets less certain. Differencing
    the raw series would read that transient dip as negative gain and force a
    premature stop, so the gate differentiates the running max instead. Both
    series are logged.
    """
    out: list[float] = []
    best = float("-inf")
    for c in coverage_history:
        best = max(best, c)
        out.append(best)
    return out


def estimate_delta_q(coverage_history: list[float]) -> float:
    """Extrapolate the gain the NEXT iteration would deliver."""
    if len(coverage_history) < 2:
        return 1.0                      # unknown -> allow one more
    last_delta = coverage_history[-1] - coverage_history[-2]
    return max(0.0, last_delta * config.DECAY_FACTOR)


def estimate_delta_c(cost_history: list[float]) -> float:
    """Next iteration's cost -- mean of observed iterations (near-constant)."""
    if not cost_history:
        return 0.0
    return sum(cost_history) / len(cost_history)


# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    query_id: str
    iteration: int
    policy: str
    coverage_raw: list[float] = field(default_factory=list)
    coverage_smoothed: list[float] = field(default_factory=list)
    delta_q: float = 0.0
    delta_c: float = 0.0
    lambda_value: float = 0.0
    lambda_times_delta_c: float = 0.0
    margin: float = 0.0
    outcome: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionLogger:
    def __init__(self, path: Path | str = DECISION_LOG_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, d: GateDecision) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(d.to_dict()) + "\n")


DECISIONS = DecisionLogger()


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------

class CAESPolicy:
    """The contribution: a per-iteration cost-aware stopping gate."""

    name = "caes"

    def __init__(self, lam: float | None = None,
                 logger: DecisionLogger | None = None,
                 record: bool = True) -> None:
        lam = config.LAMBDA if lam is None else lam
        if lam is None:
            raise ValueError(
                "LAMBDA is unset. Run `python tune_lambda.py` and write the "
                "chosen value into config.py before running CAES. Guessing it "
                "would make the headline number meaningless."
            )
        self.lam = float(lam)
        self.logger = logger or DECISIONS
        self.record = record

    def decide(self, state) -> str:
        query_id = state.get("query_id", "")
        iteration = state.get("iteration", 0)
        raw = list(state.get("coverage_history", []))
        costs = list(state.get("cost_history", []))
        smoothed = smooth_coverage(raw)

        d = GateDecision(
            query_id=query_id, iteration=iteration, policy=self.name,
            coverage_raw=raw, coverage_smoothed=smoothed,
            lambda_value=self.lam,
        )

        # Hard bounds first. A gate bug must be incapable of looping.
        if iteration >= config.MAX_ITERATIONS:
            d.outcome, d.reason = "generate", "max_iter"
            self._emit(d)
            return "generate"
        if iteration < config.MIN_ITERATIONS:
            d.outcome, d.reason = "retrieve", "min_iter"
            self._emit(d)
            return "retrieve"

        d.delta_q = estimate_delta_q(smoothed)
        d.delta_c = estimate_delta_c(costs)
        d.lambda_times_delta_c = self.lam * d.delta_c
        d.margin = d.delta_q - d.lambda_times_delta_c

        if d.margin > 0:
            d.outcome, d.reason = "retrieve", "positive_margin"
        else:
            d.outcome, d.reason = "generate", "caes"
        self._emit(d)
        return d.outcome

    def _emit(self, d: GateDecision) -> None:
        log.debug("[%s it%d] dQ=%.4f dC=%.6f l*dC=%.4f margin=%+.4f -> %s",
                  d.query_id, d.iteration, d.delta_q, d.delta_c,
                  d.lambda_times_delta_c, d.margin, d.outcome)
        if self.record:
            self.logger.log(d)


class ThresholdPolicy:
    """Insurance fallback, kept behind a flag.

    Stops when coverage_delta < 0.05 AND coverage > 0.7. Weaker as a
    contribution -- it is closer to RAGentA and has no cost term at all -- but
    it is a working system that still produces real cost data if
    estimate_delta_q proves unusable.
    """

    name = "threshold"

    def __init__(self, delta_threshold: float = 0.05,
                 coverage_threshold: float = 0.7,
                 logger: DecisionLogger | None = None,
                 record: bool = True) -> None:
        self.delta_threshold = delta_threshold
        self.coverage_threshold = coverage_threshold
        self.logger = logger or DECISIONS
        self.record = record

    def decide(self, state) -> str:
        iteration = state.get("iteration", 0)
        raw = list(state.get("coverage_history", []))
        smoothed = smooth_coverage(raw)

        d = GateDecision(
            query_id=state.get("query_id", ""), iteration=iteration,
            policy=self.name, coverage_raw=raw, coverage_smoothed=smoothed,
        )
        if iteration >= config.MAX_ITERATIONS:
            d.outcome, d.reason = "generate", "max_iter"
        elif iteration < config.MIN_ITERATIONS:
            d.outcome, d.reason = "retrieve", "min_iter"
        else:
            coverage = smoothed[-1] if smoothed else 0.0
            delta = (smoothed[-1] - smoothed[-2]) if len(smoothed) >= 2 else 1.0
            d.delta_q = delta
            if delta < self.delta_threshold and coverage > self.coverage_threshold:
                d.outcome, d.reason = "generate", "threshold"
            else:
                d.outcome, d.reason = "retrieve", "below_threshold"

        if self.record:
            self.logger.log(d)
        return d.outcome
