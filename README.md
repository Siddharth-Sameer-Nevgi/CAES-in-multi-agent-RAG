# CAES-RAG — Cost-Aware Evidence Sufficiency

A multi-agent Retrieval-Augmented Generation system, AWS-hosted and provider-
pluggable, whose novel component is a **Cost-Aware Evidence Sufficiency (CAES)
gate**: at every
retrieval iteration it decides whether to retrieve again or stop, by comparing
estimated marginal evidence-quality gain against measured marginal execution
cost.

**Decision rule:** continue retrieving only while `ΔQ − λ·ΔC > 0`

The deliverable is a working system plus a three-way experimental comparison
(CAES vs. fixed-iteration vs. one-shot routing) on HotpotQA.

> **New to this codebase?** Read [DECISIONS.md](docs/DECISIONS.md) first. It explains
> the mental model, the reading order, the invariants that must not be broken,
> and why several deliberately unusual choices are load-bearing. This file
> covers *how to run* the system; that one covers *how it thinks*.
> [CHANGELOG.md](docs/CHANGELOG.md) tracks progress phase by phase.
>
> Two companion documents go deeper:
> [METHODOLOGY.md](docs/METHODOLOGY.md) — the research design: hypothesis, the ΔQ/ΔC
> estimators, experimental controls, splits, λ tuning protocol, statistics, and
> threats to validity.
> [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — the code tour: module map, query
> lifecycle, cost accounting, and how to extend the system safely.

---

## ⚠️ Cost constraints are architectural

This project was built to run inside **$100 of AWS credits** with a hard stop
at **$40**. The guards are not preferences; removing them breaks the design.

> **Since [0.2.0] the model provider is Google Gemini, not Bedrock.** Bedrock
> model invocation is blocked account-wide on this AWS account
> (`ValidationException: Operation not allowed`, for every model and through the
> console too); a support case is open. Both `gemini-2.5-flash` and
> `gemini-embedding-001` are **free of charge on the free tier**, so **actual
> spend is $0.00** and the dollar ceiling never binds. **Rate limiting** binds
> instead. The ledger and the ceiling are kept anyway — the Bedrock path is
> retained and still tested, and a guard that only exists when needed is a guard
> that is wrong when it is needed. Every cost figure this project reports is
> **list-price notional**, computed from measured token counts, **not billed**.
> See [DECISIONS.md](docs/DECISIONS.md) **[D-22]**.

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

**Permitted:** Gemini free-tier calls; Bedrock on-demand invoke (Claude Haiku
4.5 + Titan Text Embeddings V2 only) if the block ever clears; S3 standard
storage; a single EC2 t3.micro (or fully local); CloudWatch within free tier.

**Model selection is fixed.** All LLM calls use one model —
`gemini-3.5-flash-lite` on the default provider, `anthropic.claude-haiku-4-5` on
the Bedrock path. Never a larger tier: the verification agent fires every
iteration and dominates spend. Embeddings use `gemini-embedding-001` (768 dims)
or `amazon.titan-embed-text-v2:0` (1024 dims) respectively.

### ⏱ The real budget is requests per day, not dollars

Free-tier quotas for this account, read 2026-08-31:

| Model | RPM | RPD |
|---|---:|---:|
| `gemini-3.5-flash-lite` (in use) | 15 | **500** |
| `gemini-2.5-flash` | 5 | 20 |
| `gemini-embedding-001` | 100 | **1,000** |

`gemini-3.5-flash-lite` lists at the same price as `gemini-2.5-flash` but allows
25× the daily requests — that single fact is the difference between a 15-day
experiment and a 360-day one. **The corpus was cut from 2000 questions to 500
for the same reason**; see the note under *Build the index*, and
[DECISIONS.md](docs/DECISIONS.md) **[D-24]**.

Every long-running phase can be interrupted by the daily cap and resumed:
`QuotaExhausted` exits cleanly with instructions, and the disk cache replays
completed work for free, so **no quota is ever re-spent on work already done**.
Quotas are per model *and* per account — if you change either, re-read them from
[aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit) and
update `config.py`. The test suite refuses to run a model whose RPM is not
recorded.

### AWS still hosts four of the six layers

Dropping Bedrock did not drop AWS. Bedrock was only the model-serving layer;
the rest of the protocol is unchanged and still free-tier:

| Layer | Where it runs | Cost |
|---|---|---|
| Corpus storage | **S3** standard (`ingest.py --upload-s3 BUCKET`) | free tier |
| Index + orchestration + API | **EC2 t3.micro** — FAISS, LangGraph, FastAPI | free tier (750 h/mo, 12 mo) |
| Per-iteration observability | **CloudWatch** custom metrics (`--cloudwatch`) | free tier (10 metrics) |
| Model serving | Google Gemini free tier | free |

The t3.micro is the intended host: FAISS at 768 dims over ~20k chunks is ~60 MB
resident, which fits its 1 GB comfortably — one of the reasons the embedding
dimension is truncated rather than left at the default 3072.

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
2. **A Google AI Studio API key**, exported as `GEMINI_API_KEY`:

   ```bash
   export GEMINI_API_KEY=...        # never commit this; never put it in config.py
   ```

   Everything except the key is in `config.py`. Without the key, real calls fail
   immediately with a `RuntimeError` naming the variable — before any network
   request. `DRY_RUN=1` needs no key at all.

3. **Verify the provider before spending a request budget:**

   ```bash
   python -m llm --check
   ```

   Two tiny real calls. Reports the embedding dimension actually returned
   (must match `config.EMBED_DIM`, or retrieval fails *silently*), whether
   `temperature=0.0` was accepted, and the measured token counts.

4. **AWS credentials** are still needed for S3 and CloudWatch (`aws configure`,
   or an instance role) — but **not** for model access. To run the Bedrock path
   instead, set `CAES_PROVIDER=bedrock`; it additionally needs
   `bedrock:InvokeModel` in `us-east-1` and approved model access for Claude
   Haiku 4.5 and Titan Text Embeddings V2.

5. **Pricing constants** in `config.py` carry their source and verification
   date — Gemini verified 2026-08-31 against
   `ai.google.dev/gemini-api/docs/pricing`, Bedrock 2026-08-16. Re-check before
   the final runs: those floats drive every ΔC number in the paper. Note the
   free tier bills **nothing** for either Gemini model; these are list prices
   used for notional accounting only.

---

## Workflow

Run the phases in order. Each one gates the next.

### 0. Free wiring check (do this first, always)

```bash
DRY_RUN=1 python -m smoke      # full graph, canned responses, $0.00
pytest -q                      # 127 tests, no network, no credentials

CAES_PROVIDER=bedrock pytest -q                 # the retained provider path
CAES_PROVIDER=bedrock DRY_RUN=1 python -m smoke
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

### 1. Build the index (one time, ~9 days of free-tier quota)

```bash
python ingest.py                        # HotpotQA distractor, 500 questions
python ingest.py --upload-s3 my-bucket  # optional; not on the query path
```

Guarded: if `data/index.faiss` exists it refuses to re-embed. `--force`
overrides and re-spends.

~4,500 chunks × 2 requests each (the embedding plus its `countTokens`
measurement, **[D-23]**) against a 1,000/day quota. It will stop when the day's
allowance is spent and tell you so; **re-run it the next day and it continues
from where it stopped**, replaying finished chunks from cache for free.

> **Why 500 questions and not 2000.** Purely a quota decision — 2000 is ~36 days
> of ingest. The evaluation split is unchanged (50 tune + 150 test); what
> shrinks is the *distractor pool*. That makes retrieval easier, so **absolute
> F1 will read high against published HotpotQA numbers and should not be
> compared to them.** It inflates F1 for all three arms equally, so the relative
> claim is unaffected. See METHODOLOGY §10 (External) and **[D-24]**.

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

Add `--cloudwatch` to publish per-iteration cost, latency, coverage and depth to
the `CAES-RAG` namespace. **Off by default**, so experiments stay runnable
offline and with no AWS credentials at all. This is not decoration: it is what
makes ΔC *observable per iteration in the deployment* rather than merely
computed in-process.

```bash
python -m experiments.run --policy caes --n 150 --yes --cloudwatch
```

Requires `cloudwatch:PutMetricData` on the calling identity. If the call is
refused the run continues normally and the results are unaffected — an
observability backend must never be able to fail a valid experiment. Cardinality
is four metric names × a `Policy` dimension, so a three-policy run creates
twelve unique metrics against a free allowance of ten; `--cloudwatch-no-dimensions`
collapses that to four.

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

**Billed cost on the current provider is $0.00 at every phase** — the Gemini
free tier charges nothing. The figures below are **list-price notional**: what
these runs *would* cost on a paid account, computed from measured token counts.
They are what the gate reads and what the paper reports; they are not an
invoice. Bedrock-path estimates are given for comparison, since that path is
retained.

| Phase | What | Notional (Gemini) | Notional (Bedrock) |
|---|---|---|---|
| 0 | Wiring check (`DRY_RUN=1`) | $0.00 | $0.00 |
| 1 | Index build (~18k chunks embedded) | ~$0.06 | ~$0.05 |
| 2 | Verifier calibration, 30 questions | ~$0.40 | ~$1 |
| 3 | Baselines: fixed + one-shot, 150 questions each | ~$0.80 | ~$2 |
| 4 | λ sweep on 50 held-out questions | ~$2 | ~$5 |
| 5 | CAES test run + analysis | ~$6 | ~$15 |
| | **Projected total** | **~$9 notional, $0.00 billed** | ~$24 |

Gemini figures scale the Bedrock estimates by the published price ratio (input
~3× cheaper, output ~2× cheaper) and are projections, not measurements — no run
has happened yet. The λ sweep and calibration costs are the ones to watch, since
[D-21] made the sweep deliberately dense.

**The real budget is requests, not dollars.** Free-tier rate limits are
per-account and unpublished per model, so `GEMINI_MAX_RPM` paces the client and
`429`s are retried with the server's delay. Re-runs cost far less on either
axis: the disk cache replays everything already computed.

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
├── METHODOLOGY.md          # research design: hypothesis, controls, statistics
├── IMPLEMENTATION.md       # code tour: modules, query lifecycle, extension points
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
