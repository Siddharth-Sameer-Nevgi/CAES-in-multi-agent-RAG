"""Spend tracking with a persistent ledger and hard pre-flight budget stops.

Design invariants (Phase 0 acceptance criteria):
  1. Cumulative spend is flushed to disk after EVERY recorded call, and reloaded
     on construction. A crash cannot reset the running total.
  2. BudgetExceeded is raised BEFORE the API call, from an estimate. We never
     discover we are over budget after paying for it.
  3. run_budget(max_usd) bounds a single experiment invocation.
  4. Every call is itemised for the paper's cost analysis.
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import config

log = logging.getLogger("caes.costs")


class BudgetExceeded(Exception):
    """Raised before any call that would breach a spend ceiling."""


@dataclass
class CostRecord:
    timestamp: str
    model: str
    call_type: str        # plan | verify | generate | embed
    in_tokens: int
    out_tokens: int
    latency_ms: float
    usd: float
    query_id: str = ""
    iteration: int = 0
    policy: str = ""


@dataclass
class _RunBudget:
    name: str
    max_usd: float
    start_cumulative: float

    def spent(self, cumulative: float) -> float:
        return cumulative - self.start_cumulative


class CostTracker:
    def __init__(
        self,
        ledger_path: Path | str = config.LEDGER_PATH,
        hard_budget_usd: float = config.HARD_BUDGET_USD,
        warn_budget_usd: float = config.WARN_BUDGET_USD,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.hard_budget_usd = hard_budget_usd
        self.warn_budget_usd = warn_budget_usd
        self._lock = threading.RLock()
        self._records: list[CostRecord] = []
        self._cumulative: float = 0.0
        self._warned = False
        self._run_budgets: list[_RunBudget] = []
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error(
                "Ledger at %s is unreadable (%s). Refusing to start with a zeroed "
                "total; move or repair the file.", self.ledger_path, exc)
            raise
        self._cumulative = float(raw.get("cumulative_usd", 0.0))
        self._records = [CostRecord(**r) for r in raw.get("records", [])]
        log.info("Loaded ledger: $%.4f across %d calls",
                 self._cumulative, len(self._records))

    def _flush(self) -> None:
        payload = {
            "cumulative_usd": self._cumulative,
            "n_records": len(self._records),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": [asdict(r) for r in self._records],
        }
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.ledger_path)   # atomic replace; survives a mid-write crash

    # ---------- estimation ----------

    def estimate_llm_cost(self, in_tokens: int, out_tokens_est: int) -> float:
        return (
            in_tokens / 1000.0 * config.PRICE_HAIKU_INPUT_PER_1K
            + out_tokens_est / 1000.0 * config.PRICE_HAIKU_OUTPUT_PER_1K
        )

    def estimate_embed_cost(self, in_tokens: int) -> float:
        return in_tokens / 1000.0 * config.PRICE_TITAN_EMBED_PER_1K

    # ---------- the gate ----------

    def check_affordable(self, estimated_usd: float) -> None:
        """Raise BudgetExceeded if this call would breach any active ceiling.

        Called before the network request, never after.
        """
        with self._lock:
            projected = self._cumulative + estimated_usd
            if projected > self.hard_budget_usd:
                raise BudgetExceeded(
                    f"Call would take cumulative spend to ${projected:.4f}, over "
                    f"the hard ceiling of ${self.hard_budget_usd:.2f}. "
                    f"Refusing to call Bedrock."
                )
            for rb in self._run_budgets:
                run_projected = rb.spent(projected)
                if run_projected > rb.max_usd:
                    raise BudgetExceeded(
                        f"Run {rb.name!r} would spend ${run_projected:.4f}, over "
                        f"its allowance of ${rb.max_usd:.2f}."
                    )

    # ---------- recording ----------

    def _record(self, rec: CostRecord) -> float:
        with self._lock:
            self._records.append(rec)
            self._cumulative += rec.usd
            self._flush()
            if not self._warned and self._cumulative >= self.warn_budget_usd:
                self._warned = True
                log.warning(
                    "*** SPEND WARNING: cumulative $%.2f has passed the warn "
                    "threshold of $%.2f (hard stop at $%.2f) ***",
                    self._cumulative, self.warn_budget_usd, self.hard_budget_usd,
                )
        return rec.usd

    def record_llm(
        self, *, call_type: str, in_tokens: int, out_tokens: int,
        latency_ms: float, query_id: str = "", iteration: int = 0,
        policy: str = "", model: str = config.MODEL_LLM,
    ) -> float:
        usd = (
            in_tokens / 1000.0 * config.PRICE_HAIKU_INPUT_PER_1K
            + out_tokens / 1000.0 * config.PRICE_HAIKU_OUTPUT_PER_1K
        )
        return self._record(CostRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model, call_type=call_type, in_tokens=in_tokens,
            out_tokens=out_tokens, latency_ms=latency_ms, usd=usd,
            query_id=query_id, iteration=iteration, policy=policy,
        ))

    def record_embed(
        self, *, in_tokens: int, latency_ms: float,
        query_id: str = "", iteration: int = 0, policy: str = "",
    ) -> float:
        usd = self.estimate_embed_cost(in_tokens)
        return self._record(CostRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=config.MODEL_EMBED, call_type="embed", in_tokens=in_tokens,
            out_tokens=0, latency_ms=latency_ms, usd=usd,
            query_id=query_id, iteration=iteration, policy=policy,
        ))

    # ---------- introspection ----------

    def cumulative(self) -> float:
        with self._lock:
            return self._cumulative

    def remaining(self) -> float:
        return max(0.0, self.hard_budget_usd - self.cumulative())

    def records(self) -> list[CostRecord]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, dict[str, float]] = {}
            for r in self._records:
                b = by_type.setdefault(
                    r.call_type,
                    {"calls": 0, "usd": 0.0, "in_tokens": 0, "out_tokens": 0})
                b["calls"] += 1
                b["usd"] += r.usd
                b["in_tokens"] += r.in_tokens
                b["out_tokens"] += r.out_tokens
            return {
                "cumulative_usd": self._cumulative,
                "n_calls": len(self._records),
                "by_call_type": by_type,
            }

    # ---------- export ----------

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([asdict(r) for r in self.records()])

    def to_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    # ---------- per-run ceiling ----------

    @contextmanager
    def run_budget(self, max_usd: float, name: str = "run") -> Iterator[_RunBudget]:
        """Bound the spend of one experiment invocation.

        Enforced on the same pre-flight path as the hard ceiling, so a call that
        would breach the allowance never reaches Bedrock.
        """
        with self._lock:
            rb = _RunBudget(name=name, max_usd=max_usd,
                            start_cumulative=self._cumulative)
            self._run_budgets.append(rb)
        try:
            yield rb
        finally:
            with self._lock:
                self._run_budgets.remove(rb)
                log.info("Run %r spent $%.4f of its $%.2f allowance",
                         name, rb.spent(self._cumulative), max_usd)


# Process-wide tracker. Everything shares one ledger.
TRACKER = CostTracker()
