# CAES-RAG — Cost-Aware Evidence Sufficiency

A multi-agent Retrieval-Augmented Generation system on AWS Bedrock whose novel
component is a **Cost-Aware Evidence Sufficiency (CAES) gate**: at every
retrieval iteration it decides whether to retrieve again or stop, by comparing
estimated marginal evidence-quality gain against measured marginal execution
cost.

**Decision rule:** continue retrieving only while `ΔQ − λ·ΔC > 0`

The deliverable is a working system plus a three-way experimental comparison
(CAES vs. fixed-iteration vs. one-shot routing) on HotpotQA.

> **New to this codebase?** Read [DECISIONS.md](DECISIONS.md) first. It explains
> the mental model, the reading order, the invariants that must not be broken,
> and why several deliberately unusual choices are load-bearing. This file
> covers *how to run* the system; that one covers *how it thinks*.
> [CHANGELOG.md](CHANGELOG.md) tracks progress phase by phase.

---

## ⚠️ Cost constraints are architectural

This project is built to run inside **$100 of AWS credits** with a hard stop at
**$40**. The guards are not preferences; removing them breaks the design.

### Services that must NEVER be used

| Forbidden | Reason |
|---|---|
| Bedrock Knowledge Bases | Provisions OpenSearch Serverless, $345+/month minimum, bills at zero traffic |
| OpenSearch Serverless / Service | Same — minimum-capacity billing |
| Bedrock Agents | ~5× token amplification vs. direct orchestration |
| SageMaker endpoints | Always-on hourly billing |
| NAT Gateway | ~$32/month standing charge |
| Elastic IP (unattached) | Billed when not associated |
| Provisioned Throughput | Hourly commitment, $40–200/hr |

**Permitted:** Bedrock on-demand invoke (Claude Haiku 4.5 + Titan Text
Embeddings V2 only), S3 standard storage, a single EC2 t3.micro (or fully
local), CloudWatch logs within free tier.

**Model selection is fixed.** All LLM calls use `anthropic.claude-haiku-4-5` —
never Sonnet, never Opus. The verification agent fires every iteration and
dominates spend. All embeddings use `amazon.titan-embed-text-v2:0`.

### How spend is actually prevented

* `costs.py` keeps a **persistent ledger** (`.spend_ledger.json`), flushed after
  every call and reloaded on startup. A crash cannot reset the running total.
* `BudgetExceeded` is raised **before** the network call, from a pre-flight
  estimate — never after paying for it.
* `run_budget(max_usd)` bounds a single experiment invocation on the same gate.
* `cache.py` makes re-runs free. Re-running an experiment after a bug fix costs
  nothing for anything already computed.

Check spend at any time:

```bash
python -c "from costs import TRACKER; print(TRACKER.summary())"
```

---

## Prerequisites

1. **Python 3.11+** and `pip install -r requirements.txt`
2. **AWS credentials** with `bedrock:InvokeModel`, in `us-east-1`
   (`aws configure`, or an instance role).
3. **Bedrock model access must be requested and approved** in the AWS console
   before anything works — Bedrock → Model access → enable:
   * Anthropic Claude Haiku 4.5
   * Amazon Titan Text Embeddings V2

   Without this, every call fails with `AccessDeniedException`.
4. **Pricing constants** in `config.py` were verified 2026-08-16
   (Haiku 4.5 $1/$5 per 1M; Titan Embeddings **V2** $0.02 per 1M — note v1/G1
   bills 5× that, at $0.10). Re-check before the final runs: those three floats
   drive every ΔC number in the paper. The AWS pricing page renders its tables
   in JS, so confirm in the Bedrock console rather than by scraping the page.

---

## Workflow

Run the phases in order. Each one gates the next.

### 0. Free wiring check (do this first, always)

```bash
DRY_RUN=1 python -m smoke      # full graph, canned responses, $0.00
pytest -q                      # 60+ tests, no network, no credentials
```

`DRY_RUN=1` returns canned responses without touching the network. The smoke
test builds a tiny synthetic index if no real one exists, so it works on a clean
checkout, and asserts the ledger did not move.

To exercise the *whole* pipeline (experiments, tuning, figures) for free:

```bash
DRY_RUN=1 python devdata.py                 # synthetic corpus + 260 questions
DRY_RUN=1 python -m experiments.run --policy fixed --n 40 --yes
DRY_RUN=1 python -m experiments.run --policy caes --n 40 --lam 40 --yes
DRY_RUN=1 python -m experiments.analyze
```

`devdata.py` stamps `data/meta.json` with `"synthetic": true` so synthetic data
is never mistaken for the real corpus. Delete `data/` before the real ingest.

### 1. Build the index (~$0.05, one time)

```bash
python ingest.py                        # HotpotQA distractor, 2000 questions
python ingest.py --upload-s3 my-bucket  # optional; not on the query path
```

Guarded: if `data/index.faiss` exists it refuses to re-embed. `--force`
overrides and re-spends.

### 2. Calibrate the verifier (~$1) — **this is a blocker**

```bash
python calibrate_verifier.py --n 30
```

The verifier's coverage scores *are* the ΔQ signal. If they do not spread across
the range, ΔQ is noise and the method fails. This script fails the build if
JSON parsing is imperfect or if scores cluster in a narrow band, and tells you
to sharpen `VERIFIER_PROMPT` in `agents/prompts.py`. **Do not proceed to Phase 3
on a flat coverage signal.**

### 3. Baselines (~$2)

```bash
python -m experiments.run --policy fixed   --n 150 --max-usd 5 --yes
python -m experiments.run --policy oneshot --n 150 --max-usd 5 --yes
```

### 4. Tune λ (~$5) — on held-out data only

```bash
python tune_lambda.py --yes
```

Sweeps λ over `[1, 3, 10, 30, 100, 300, 1000]`, then refines log-spaced around
the knee, over the **50-question tuning split** — disjoint by construction from the 150-question
test split (`splits.py` asserts it). Emits `results/lambda_sweep.csv`.

Then **write the chosen value into `config.py`**:

```python
LAMBDA = 40.0    # from tune_lambda.py — do not re-tune
```

`CAESPolicy` refuses to construct while `LAMBDA is None`. Tuning on test data
invalidates the result, so the test split is never loaded by the tuner.

### 5. Experiments (~$15)

```bash
python -m experiments.run --policy caes --n 150 --max-usd 5 --yes
python -m experiments.analyze
```

`run.py` checkpoints after **every** query to `results/{policy}_raw.jsonl`, so a
crash at query 130 does not lose the first 129. Resume with `--resume`. If the
run-budget guard trips mid-run, the partial results are already on disk — raise
the allowance and `--resume`.

### 6. Serve (optional)

```bash
uvicorn api:app --port 8000
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question":"Who directed Inception?"}'
```

---

## Cost table

| Phase | What | Estimated |
|---|---|---|
| 0 | Wiring check (`DRY_RUN=1`) | **$0.00** |
| 1 | Index build (~18k chunks embedded) | ~$0.05 |
| 2 | Verifier calibration, 30 questions | ~$1 |
| 3 | Baselines: fixed + one-shot, 150 questions each | ~$2 |
| 4 | λ sweep on 50 held-out questions | ~$5 |
| 5 | CAES test run + analysis | ~$15 |
| | **Projected total** | **~$24** of the $40 ceiling |

Re-runs cost far less: the disk cache replays everything already computed.

Embeddings are a rounding error on the query path — about **0.03%** of a
three-iteration query. ΔC is, in practice, almost entirely the verifier and
planner LLM calls, which is exactly why the verifier's evidence truncation
(`VERIFIER_CHUNK_CHARS`) is the main lever on gate overhead.

---

## Design notes worth knowing

**Notional vs. actual cost.** The ledger tracks money actually spent; a cache hit
spends nothing. But the CAES gate needs the *cost of an iteration*, independent
of whether this particular run replayed it. If ΔC collapsed to zero on a warm
cache, the gate would see an infinitely cheap next iteration and run to
`MAX_ITERATIONS` — a cached re-run would produce different decisions from the
paid one. So `bedrock.py` accrues **notional cost** from the real measured token
counts (which the cache preserves), and the gate reads that. Results are
identical cached or not.

**The gate runs in `verify`, not in the edge.** LangGraph does not merge state
written inside a conditional-edge function, so a gate that recorded its
reasoning there would silently lose it. The decision is computed at the end of
`verify` and written to state; the edge is a pure read.

**`MAX_ITERATIONS` is enforced by the graph**, checked before the policy is
consulted at all. `tests/test_graph.py` proves it with a deliberately broken
gate that always says "retrieve".

**Coverage smoothing.** Coverage genuinely dips sometimes — a new document
introduces a second plausible entity and the verifier rightly gets less certain.
The gate differentiates a *running max* so a transient dip does not force a
premature stop. Both raw and smoothed series are logged.

**The confidence short-circuit is off by default.** Stopping early when the
verifier reports high coverage *and* confidence is available
(`honor_confidence=True`) and is used by the API, but stays **off** for the
three-way comparison: giving CAES a second, orthogonal stopping rule the
baselines lack would confound the contribution. The API turns it on because it
optimises for latency and makes no cross-policy claim.

**Abstentions score zero.** A policy that stops too early and honestly says
"insufficient evidence" is penalised for it, otherwise the cost saving would
look free.

**Fallback policy.** If `estimate_delta_q` proves unusable, `ThresholdPolicy`
(stop when `coverage_delta < 0.05 AND coverage > 0.7`) is kept behind
`--policy threshold`. Weaker as a contribution — closer to RAGentA, and with no
cost term at all — but it is a working system with real cost data.

---

## Layout

```
caes/
├── DECISIONS.md            # START HERE if you are new — design rationale
├── CHANGELOG.md            # what landed, phase by phase
├── config.py               # all tunables: prices, budgets, model ids, λ
├── costs.py                # CostTracker, BudgetExceeded, persistent ledger
├── cache.py                # disk cache; hits never touch the tracker
├── bedrock.py              # SOLE Bedrock entry point; DRY_RUN mode
├── ingest.py               # Phase 1: corpus → chunks → embeddings → FAISS
├── retrieval.py            # Retriever.search()
├── splits.py               # deterministic, disjoint tune/test splits
├── metrics.py              # SQuAD-style EM / F1
├── agents/
│   ├── prompts.py          # all prompts as named constants
│   ├── planner.py          # iteration 1 is free (question verbatim)
│   ├── verifier.py         # CRITICAL: the ΔQ signal, defensive parsing
│   └── generator.py        # grounded answer, abstains rather than speculates
├── caes.py                 # ΔQ / ΔC estimators, CAESPolicy, decision log
├── policies.py             # FixedPolicy, OneShotPolicy, factory
├── graph.py                # LangGraph state machine + hard iteration cap
├── calibrate_verifier.py   # Phase 2 acceptance gate
├── tune_lambda.py          # Phase 4 λ sweep on held-out data
├── devdata.py              # synthetic corpus for free pipeline testing
├── smoke.py                # DRY_RUN end-to-end check
├── api.py                  # minimal FastAPI layer
├── experiments/
│   ├── run.py              # checkpointing driver with --resume
│   └── analyze.py          # table, 4 figures, headline number
├── tests/
└── results/
    ├── {policy}_raw.jsonl  # per-query records
    ├── caes_decisions.jsonl# ΔQ, ΔC, λ·ΔC, margin, outcome per iteration
    ├── lambda_sweep.csv
    ├── main_table.csv
    ├── headline.json
    └── figures/
```

## Outputs

| Artifact | What it shows |
|---|---|
| `main_table.csv` | mean cost, latency, EM, F1, mean iterations per policy |
| `fig1_cost_vs_quality.png` | the headline figure |
| `fig2_iteration_histogram.png` | CAES spread across 1–5 vs. Fixed flat at N |
| `fig3_coverage_vs_iteration.png` | empirical diminishing returns |
| `fig4_lambda_sweep.png` | the cost/quality tradeoff curve from tuning |
| `headline.json` | % cost reduction vs. Fixed, paired t-test + bootstrap CI |
| `caes_decisions.jsonl` | every gate decision — the paper's central figure |

Figures use a CVD-validated three-colour categorical palette assigned to
policies (never to rank), with direct labels so identity never rests on colour
alone.
