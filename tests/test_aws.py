"""The AWS layers that survived the provider migration: S3 and CloudWatch.

Bedrock was only the model-serving layer. S3 (corpus), EC2 (host) and
CloudWatch (per-iteration observability) are unaffected by the invocation block
and stay in the deployment.

Neither test reaches AWS. The S3 path is checked with botocore's Stubber, which
validates the request against the real service model — so a malformed
`put_object` fails here rather than at ingest time. CloudWatch is checked
against a recording double.
"""
from __future__ import annotations

import json

import pytest

import config
from observability import CloudWatchPublisher


# ---------------------------------------------------------------------------
# S3 — ingest.py --upload-s3
# ---------------------------------------------------------------------------

def test_upload_to_s3_sends_well_formed_jsonl(monkeypatch):
    """Validated against the real S3 service model, without touching AWS."""
    import boto3
    from botocore.stub import ANY, Stubber

    import ingest

    client = boto3.client("s3", region_name=config.AWS_REGION,
                          aws_access_key_id="test", aws_secret_access_key="test")
    stubber = Stubber(client)
    stubber.add_response("put_object", {"ETag": '"abc"'},
                         {"Bucket": "caes-test-bucket",
                          "Key": "caes-rag/corpus/passages.jsonl",
                          "Body": ANY})
    stubber.activate()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)

    passages = [{"title": "Speed (1994 film)", "text": "a film"},
                {"title": "Jan de Bont", "text": "a director"}]
    ingest.upload_to_s3("caes-test-bucket", passages)

    stubber.assert_no_pending_responses()


def test_upload_to_s3_body_round_trips(monkeypatch):
    """The uploaded object must be readable back as one JSON object per line."""
    import boto3

    import ingest

    captured = {}

    class Recording:
        def put_object(self, **kwargs):
            captured.update(kwargs)
            return {"ETag": '"abc"'}

    monkeypatch.setattr(boto3, "client", lambda *a, **k: Recording())
    passages = [{"title": "A", "text": "one"}, {"title": "B", "text": "two"}]
    ingest.upload_to_s3("bucket", passages)

    lines = captured["Body"].decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == passages
    assert captured["Key"] == "caes-rag/corpus/passages.jsonl"


# ---------------------------------------------------------------------------
# CloudWatch — experiments/run.py --cloudwatch
# ---------------------------------------------------------------------------

class RecordingCloudWatch:
    def __init__(self, fail=False):
        self.batches = []
        self.fail = fail

    def put_metric_data(self, **kwargs):
        if self.fail:
            raise RuntimeError("AccessDenied")
        self.batches.append(kwargs)
        return {}


def _record(iterations=3, policy="caes"):
    return {
        "policy": policy,
        "iterations_used": iterations,
        "cost_history": [0.001] * iterations,
        "latency_history": [120.0] * iterations,
        "coverage_history": [0.4, 0.6, 0.7][:iterations],
    }


def test_publishes_the_four_per_iteration_metrics():
    cw = RecordingCloudWatch()
    with CloudWatchPublisher(policy="caes", client=cw) as pub:
        pub.record_query(_record(iterations=3))

    data = [d for b in cw.batches for d in b["MetricData"]]
    names = [d["MetricName"] for d in data]
    assert names.count("IterationCost") == 3, "one datum per iteration"
    assert names.count("IterationLatency") == 3
    assert names.count("IterationCoverage") == 3
    assert names.count("IterationsUsed") == 1, "one datum per query"
    assert all(b["Namespace"] == config.CLOUDWATCH_NAMESPACE for b in cw.batches)
    assert pub.published == len(data)


def test_policy_dimension_separates_the_arms():
    cw = RecordingCloudWatch()
    with CloudWatchPublisher(policy="fixed", client=cw) as pub:
        pub.record_query(_record(policy="fixed"))
    data = [d for b in cw.batches for d in b["MetricData"]]
    assert all(d["Dimensions"] == [{"Name": "Policy", "Value": "fixed"}]
               for d in data)


def test_no_dimensions_mode_collapses_cardinality():
    """4 unique metrics total rather than 4 per policy, for the free tier."""
    cw = RecordingCloudWatch()
    with CloudWatchPublisher(policy="fixed", dimensions=False,
                             client=cw) as pub:
        pub.record_query(_record())
    data = [d for b in cw.batches for d in b["MetricData"]]
    assert all(d["Dimensions"] == [] for d in data)
    assert len({d["MetricName"] for d in data}) == 4


def test_batches_respect_the_put_metric_data_limit():
    cw = RecordingCloudWatch()
    with CloudWatchPublisher(policy="caes", flush_every=10_000,
                             client=cw) as pub:
        for _ in range(200):                 # 200 x 10 datums = 2000
            pub.record_query(_record(iterations=3))
    assert all(len(b["MetricData"]) <= 1000 for b in cw.batches), \
        "PutMetricData accepts at most 1000 datums per request"
    assert sum(len(b["MetricData"]) for b in cw.batches) == 2000


def test_a_cloudwatch_failure_never_breaks_the_run():
    """An observability backend must not be able to fail a valid experiment."""
    cw = RecordingCloudWatch(fail=True)
    pub = CloudWatchPublisher(policy="caes", client=cw)
    pub.record_query(_record())
    pub.flush()                              # must not raise
    assert pub.failures == 10
    assert pub.published == 0


def test_publisher_is_off_unless_the_flag_is_passed():
    """--cloudwatch defaults off so experiments stay runnable offline."""
    parsed = _parse(["--policy", "caes"])
    assert parsed.cloudwatch is False
    assert parsed.cloudwatch_no_dimensions is False
    assert _parse(["--policy", "caes", "--cloudwatch"]).cloudwatch is True


def _parse(argv):
    """Reach the driver's parser without running it."""
    import argparse
    import experiments.run as run_mod

    captured = {}
    real = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        captured["ns"] = ns
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            run_mod.main(argv)
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured["ns"]


# ---------------------------------------------------------------------------
# Ingest resume — a multi-day free-tier ingest is restarted many times
# ---------------------------------------------------------------------------

def test_ingest_resume_costs_no_quota(monkeypatch, tmp_path):
    """Ingest spans days on a free-tier quota, so it is resumed repeatedly.

    A resume must issue NO new provider requests for chunks already embedded --
    including the batched countTokens calls, which are easy to forget and would
    otherwise burn ~112 requests of the daily allowance on finished work.
    """
    import numpy as np

    import cache as cache_mod
    import ingest
    import llm as llm_mod

    disk = cache_mod.DiskCache(tmp_path / "cache")
    monkeypatch.setattr(llm_mod, "CACHE", disk)
    monkeypatch.setattr(llm_mod, "DRY_RUN", True)

    chunks = [{"chunk_id": str(i), "title": f"T{i}", "part": 0,
               "text": f"Title {i}: passage body number {i} with some words."}
              for i in range(120)]

    first_vecs, first_tokens = ingest.embed_chunks(chunks, batch=50)
    writes_after_first = disk.stats()["writes"]
    assert writes_after_first > 0

    second_vecs, second_tokens = ingest.embed_chunks(chunks, batch=50)
    assert disk.stats()["writes"] == writes_after_first, \
        "a resumed ingest re-issued provider requests for finished work"
    assert second_tokens == first_tokens, "the token total drifted across resume"
    assert np.array_equal(first_vecs, second_vecs), \
        "a resumed ingest produced different vectors"


def test_count_tokens_is_free_under_dry_run(monkeypatch):
    """Every network path must be exercisable for $0.00, or a multi-day ingest
    cannot be rehearsed before it is committed to."""
    import config as config_mod
    import llm as llm_mod
    from tests.conftest import _select_provider

    _select_provider(monkeypatch, "gemini")
    monkeypatch.setattr(llm_mod, "DRY_RUN", True)
    monkeypatch.delenv(config_mod.GEMINI_API_KEY_ENV, raising=False)
    assert llm_mod.count_tokens(["some text", "more text"]) > 0
