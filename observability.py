"""Optional CloudWatch publishing for experiment runs.

Off unless `--cloudwatch` is passed, so experiments stay runnable offline and
with no AWS credentials at all.

**This is not decoration.** METHODOLOGY §3.2 defines ΔC as *measured* marginal
cost, metered per iteration. Publishing cost, latency, coverage and depth per
iteration makes ΔC **observable per iteration in the deployment** rather than
merely computed inside the process — which is the methodological point, and the
reason the AWS-native claim survives the move of model serving to Gemini.

Nothing here touches the ledger, the gate, the cache, or any result. A
CloudWatch failure is logged and swallowed: an observability backend must never
be able to fail an experiment that is otherwise producing valid data.

Metric cardinality, because CloudWatch bills per unique metric beyond the free
tier: four metric names x one `Policy` dimension = **four metrics per policy**,
so a full three-policy run creates twelve. The free tier covers ten. Pass
`--cloudwatch-no-dimensions` to collapse to four total if that matters more than
separating the arms.
"""
from __future__ import annotations

import logging
from typing import Any

import config

log = logging.getLogger("caes.observability")

# PutMetricData accepts at most 1000 MetricDatum per request.
_MAX_DATUMS_PER_CALL = 1000


class CloudWatchPublisher:
    """Buffers per-iteration metrics and flushes them in batches.

    Construct once per run and call `record_query` after each query completes;
    `flush` is idempotent and is also called by the context manager on exit.
    """

    def __init__(self, namespace: str = config.CLOUDWATCH_NAMESPACE, *,
                 policy: str = "", dimensions: bool = True,
                 flush_every: int = 200, client=None) -> None:
        self.namespace = namespace
        self.policy = policy
        self.dimensions = dimensions
        self.flush_every = min(flush_every, _MAX_DATUMS_PER_CALL)
        self._client = client
        self._buffer: list[dict[str, Any]] = []
        self.published = 0
        self.failures = 0

    # ---------- client ----------

    def client(self):
        """Lazily construct the CloudWatch client.

        Lazy so that importing this module never requires credentials, and so
        an offline run that never passes --cloudwatch never builds one.
        """
        if self._client is None:
            import boto3
            self._client = boto3.client("cloudwatch",
                                        region_name=config.AWS_REGION)
        return self._client

    def _dims(self) -> list[dict[str, str]]:
        """`Policy` only, deliberately.

        Adding `Iteration` as a dimension would let you slice by iteration
        index, but it multiplies cardinality by MAX_ITERATIONS: three
        per-iteration metrics x 5 iterations x 3 policies is 45 unique metrics
        against a free allowance of 10. Each iteration already emits its own
        datapoint, so the per-iteration series is observable either way; only
        the by-index breakdown is given up.
        """
        if not self.dimensions:
            return []
        return [{"Name": "Policy", "Value": self.policy or "unknown"}]

    # ---------- recording ----------

    def record_query(self, rec: dict[str, Any]) -> None:
        """Buffer the metrics for one completed query record.

        `rec` is a `graph.state_summary` result: it carries the per-iteration
        `cost_history`, `latency_history` and `coverage_history` series.
        """
        costs = rec.get("cost_history") or []
        latencies = rec.get("latency_history") or []
        coverages = rec.get("coverage_history") or []

        for usd in costs:
            self._add("IterationCost", float(usd), "None")
        for ms in latencies:
            self._add("IterationLatency", float(ms), "Milliseconds")
        for cov in coverages:
            self._add("IterationCoverage", float(cov), "None")

        self._add("IterationsUsed", float(rec.get("iterations_used", 0)), "Count")

        if len(self._buffer) >= self.flush_every:
            self.flush()

    def _add(self, name: str, value: float, unit: str) -> None:
        self._buffer.append({
            "MetricName": name,
            "Dimensions": self._dims(),
            "Value": value,
            "Unit": unit,
        })

    # ---------- flushing ----------

    def flush(self) -> None:
        """Send everything buffered. Never raises."""
        while self._buffer:
            batch, self._buffer = (self._buffer[:_MAX_DATUMS_PER_CALL],
                                   self._buffer[_MAX_DATUMS_PER_CALL:])
            try:
                self.client().put_metric_data(Namespace=self.namespace,
                                              MetricData=batch)
                self.published += len(batch)
            except Exception as exc:                 # noqa: BLE001 - see docstring
                self.failures += len(batch)
                log.warning(
                    "CloudWatch put_metric_data failed for %d datums (%s: %s). "
                    "The run continues; results are unaffected.",
                    len(batch), type(exc).__name__, str(exc)[:200])

    def summary(self) -> str:
        return (f"cloudwatch: {self.published} datums published to "
                f"{self.namespace}"
                + (f", {self.failures} failed" if self.failures else ""))

    # ---------- context manager ----------

    def __enter__(self) -> CloudWatchPublisher:
        return self

    def __exit__(self, *exc_info) -> bool:
        self.flush()
        return False
