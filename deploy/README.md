# AWS deployment runbook

Four of the six protocol layers run on AWS at zero cost. Model serving is the
one that does not — Bedrock invocation is blocked account-wide on this account
(`ValidationException: Operation not allowed`, re-confirmed 2026-08-31), so
Gemini serves the models. See [DECISIONS.md](../docs/DECISIONS.md) **[D-22]**.

| Layer | Service | Cost | State |
|---|---|---|---|
| Corpus archive | S3 standard | free tier | permissions OK, no bucket yet |
| Per-iteration metrics | CloudWatch custom metrics | free tier (10 metrics) | **`PutMetricData` denied** |
| Index + orchestration + API | EC2 t3.micro | free tier, 750 h/mo | **not authorised, not provisioned** |
| Model serving | Google Gemini free tier | free | running |

**Nothing here changes a single number in the results.** S3 archives the corpus
and is never read on the query path; CloudWatch is write-only observability and
a failed publish is swallowed by design; EC2 is where the same code runs. The
experiment is reproducible without any of it.

---

## Order matters

**Do step 1 and 2 before Phase 5 (the test-split runs), not after.** CloudWatch
metrics can only be emitted while a run is happening. Adding them afterwards
means re-running the experiment, and on a 500-requests/day free tier that costs
days — not money.

---

## 1. Grant the missing permissions

The current user (`iam::099868052312:user/sid_nevgi`) has S3 but not CloudWatch
or EC2. [`iam-policy.json`](iam-policy.json) is least-privilege: S3 scoped to
`caes-rag-*` buckets, `PutMetricData` scoped by condition to the `CAES-RAG`
namespace only, and EC2 limited to standing up one host.

Console → IAM → Users → `sid_nevgi` → Add permissions → Create inline policy →
JSON → paste → name it `caes-rag-deploy`.

Or, with credentials that can write IAM:

```bash
aws iam put-user-policy \
  --user-name sid_nevgi \
  --policy-name caes-rag-deploy \
  --policy-document file://deploy/iam-policy.json
```

Verify:

```bash
python -c "
import boto3
boto3.client('cloudwatch', region_name='us-east-1').put_metric_data(
    Namespace='CAES-RAG',
    MetricData=[{'MetricName':'Probe','Value':1.0,'Unit':'Count'}])
print('PutMetricData OK')
"
```

## 2. Create the corpus bucket and upload

Bucket names are globally unique, so pick a suffix. It must match the
`caes-rag-*` pattern the policy scopes to.

```bash
aws s3api create-bucket --bucket caes-rag-corpus-<suffix> --region us-east-1
aws s3api put-public-access-block --bucket caes-rag-corpus-<suffix> \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

Then, from the repo root:

```bash
python ingest.py --upload-s3 caes-rag-corpus-<suffix>
```

Ingest refuses to re-embed when `data/index.faiss` exists, so this uploads the
4,965 deduplicated passages without re-spending a single embedding request.

## 3. Publish per-iteration metrics during Phase 5

```bash
python -m experiments.run --policy caes --n 150 --yes --cloudwatch
```

This is the methodological point rather than decoration: METHODOLOGY §3.2
defines ΔC as *measured* marginal cost metered per iteration, and publishing it
makes that observable **in the deployment** rather than only inside the process.

Cardinality: four metric names × a `Policy` dimension = four metrics per policy,
so three policies create twelve against a free allowance of ten. Pass
`--cloudwatch-no-dimensions` to collapse to four if that matters more than
separating the arms. A denied publish is logged, counted in the run summary,
and **cannot fail the run**.

Read the metrics back:

```bash
aws cloudwatch list-metrics --namespace CAES-RAG
```

## 4. (Optional) EC2 t3.micro host

Only needed to demonstrate the serving layer; the experiment does not require
it. `t3.micro` is free-tier for 750 h/month in the first 12 months.

The 768-dimension embedding choice was made partly for this box: a flat FAISS
index over 5,552 chunks is ~17 MB resident at 768 dims against ~68 MB at 3072,
which fits 1 GB RAM alongside Python, FAISS and FastAPI comfortably.

```bash
# Amazon Linux 2023, t3.micro, in the default VPC
sudo dnf install -y python3.11 git
git clone <your-repo> && cd caes
pip install -r requirements.txt
export GEMINI_API_KEY=...          # never bake this into an AMI or user-data
python ingest.py                   # or: aws s3 cp the corpus down
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open port 8000 to **your IP only**. `api.py` has no auth, no rate limiting and
no TLS by design — it exists to substantiate the protocol's API layer, not to
be exposed.

**Stop the instance when not demonstrating.** Free tier is 750 hours/month;
one instance running continuously is 730, so a second one starts costing money.

---

## What this does not do

Bedrock stays blocked, so model serving stays on Gemini. The Bedrock code path
is retained and tested (`CAES_PROVIDER=bedrock` passes the full suite) so a
two-provider robustness result is available if the block ever clears — but
switching providers mid-experiment would invalidate the verifier calibration
and the tuned λ, since both are properties of a specific model. That is
invariant 9, and it is why the switch is a *later* experiment rather than a
configuration change.
