# IMPLEMENTATION.md — how the system is built

A code-level tour: what each module owns, how a query flows through the system,
where cost is measured, and where to make changes safely.

For *what is being tested and why the test is valid*, see
[METHODOLOGY.md](METHODOLOGY.md). For *why* individual choices were made, see
[DECISIONS.md](DECISIONS.md). For *how to run* the phases, see
[README.md](README.md).

---

## 1. Architecture at a glance

```
                        ┌─────────────────────────────────────┐
   question ──────────► │  graph.py — LangGraph state machine │
                        │                                     │
                        │   plan → retrieve → verify ─┐       │
                        │     ▲                       │       │
                        │     └──── "retrieve" ◄─ GATE│       │
                        │                             ▼       │
                        │                        "generate"   │
                        └───────┬─────────────────────┬───────┘
                                │                     │
              agents/{planner,verifier,generator}   caes.py / policies.py
                                │                     (the stopping rule)
                                ▼
                            llm.py  ── the sole model-provider entry point
                            │   │   │
             cache.py ◄─────┘   │   └─────► costs.py (ledger + budget gate)
        (free replays)          ▼
                    retrieval.py → FAISS index (built by ingest.py)
```

Three properties hold everywhere and everything else follows from them:

1. **Every model call goes through `llm.py`.** No direct `boto3` call or
   provider HTTP request exists anywhere else, because the ledger's completeness
   depends on that. `config.PROVIDER` (`"bedrock" | "gemini"`) selects request
   shaping *inside* `llm.py`; nothing downstream knows which provider is
   active. See [DECISIONS.md](DECISIONS.md) **[D-22]**.
2. **Every tunable lives in `config.py`.** No price, model id, or loop bound is
   hardcoded elsewhere.
3. **The stopping decision is made in exactly one place** — `evaluate_gate` in
   `graph.py`, at the end of the `verify` node.

## 2. Module map

| Module | Owns | Key exports |
|---|---|---|
| [config.py](config.py) | every tunable: provider, prices, budgets, model ids, λ, paths | module-level constants, `provider_settings` |
| [costs.py](costs.py) | persistent spend ledger, pre-flight budget gate | `CostTracker`, `TRACKER`, `BudgetExceeded`, `run_budget` |
| [cache.py](cache.py) | content-addressed disk cache for responses | `DiskCache`, `CACHE`, `make_key` |
| [llm.py](llm.py) | the only model client, either provider; notional accounting; dry-run | `invoke_llm`, `embed`, `totals`, `DRY_RUN`, `check_provider` |
| [ingest.py](ingest.py) | corpus → chunks → embeddings → FAISS index | `main`, `chunk_text`, `build_index` |
| [retrieval.py](retrieval.py) | dense search over the index | `Retriever`, `get_retriever`, `Chunk` |
| [splits.py](splits.py) | deterministic disjoint tune/test splits | `make_splits`, `tune_set`, `test_set` |
| [metrics.py](metrics.py) | SQuAD-style EM / F1, abstention handling | `score`, `exact_match`, `f1`, `is_abstention` |
| [agents/prompts.py](agents/prompts.py) | every prompt, as named constants | `VERIFIER_PROMPT`, … |
| [agents/planner.py](agents/planner.py) | the retrieval query for each iteration | `plan` |
| [agents/verifier.py](agents/verifier.py) | coverage scoring — the ΔQ signal | `verify`, `Verification` |
| [agents/generator.py](agents/generator.py) | grounded answer or honest abstention | `generate` |
| [caes.py](caes.py) | ΔQ / ΔC estimators, `CAESPolicy`, decision log | `CAESPolicy`, `estimate_delta_q`, `DECISIONS` |
| [policies.py](policies.py) | baselines and the factory | `FixedPolicy`, `OneShotPolicy`, `build_policy` |
| [graph.py](graph.py) | the state machine, the gate, the hard iteration cap | `run_query`, `evaluate_gate`, `state_summary` |
| [calibrate_verifier.py](calibrate_verifier.py) | Phase 2 acceptance gate on the ΔQ signal | `main` |
| [tune_lambda.py](tune_lambda.py) | Phase 4 λ sweep on held-out data | `evaluate_lambda`, `find_knee` |
| [experiments/run.py](experiments/run.py) | checkpointing experiment driver | `main`, `load_completed` |
| [experiments/analyze.py](experiments/analyze.py) | table, figures, headline number | `headline`, `main_table` |
| [devdata.py](devdata.py) | synthetic corpus for free pipeline testing | `main` |
| [smoke.py](smoke.py) | dry-run end-to-end wiring check | `main` |
| [api.py](api.py) | minimal FastAPI serving layer | `app` |

## 3. The query lifecycle

`run_query(question, policy)` in [graph.py](graph.py) builds an initial `State`,
runs the compiled graph, and returns the final state. One iteration is
**plan → retrieve → verify**, and the gate at the end of `verify` decides whether
to loop or fall through to `generate`.

### 3.1 State

`State` is a `TypedDict`. The fields that carry the experiment:

| Field | Meaning |
|---|---|
| `iteration` | retrieval iterations completed so far |
| `evidence` | accumulated `Chunk` list, de-duplicated by `chunk_id` |
| `coverage_history` | raw verifier coverage, one entry per iteration |
| `cost_history` | notional USD per iteration, one entry per iteration |
| `missing` | what the verifier said is still needed — feeds the next planner call |
| `stop_reason` | `caes` \| `fixed` \| `oneshot` \| `max_iter` \| `confident` |
| `_route` | `retrieve` \| `generate`, written by `verify`, read by the edge |
| `_iter_usd_mark` | counter snapshot taken at the start of the iteration |
| `_query_usd_mark` | counter snapshot taken at the start of the query |

The `_*_mark` fields are how cost is metered: `llm.totals()` returns
monotonic process-wide counters, and a span's cost is the difference between two
snapshots. `plan` takes the iteration mark; `verify` diffs it; `initial_state`
takes the query mark and `generate` diffs it.

### 3.2 Nodes

**`node_plan`** increments the iteration and produces the retrieval query.
Iteration 1 returns the question **verbatim with no LLM call** — rewriting a
question that has not yet been searched is pure overhead. Later iterations ask
the planner for a focused sub-query targeting the verifier's `missing` field,
falling back to `question + missing` if the model returns nothing.

**`node_retrieve`** embeds the query (a real, metered cost), searches FAISS for
`TOP_K` hits, filters out chunks already seen, and appends the fresh ones.

**`node_verify`** calls the verifier, appends coverage / cost / latency to their
histories, then evaluates the gate against the **post-verification** view of the
state and writes the route into `_route`.

**`node_generate`** produces the grounded answer and records total query cost and
latency by diffing the query marks.

### 3.3 Where the gate runs, and why it matters

The decision is computed at the end of `verify` and written into state; the
LangGraph conditional edge (`route_from_state`) is a **pure read** of that field.

This is not stylistic. LangGraph does not merge writes made inside a
conditional-edge function back into graph state, so a gate that recorded its
reasoning from the edge would silently lose it — and that reasoning is the
paper's central figure.

`evaluate_gate` is the only place a stop decision is made:

```python
if state["iteration"] >= config.MAX_ITERATIONS:      # BEFORE the policy
    return "generate", "max_iter"
if honor_confidence and state["confident"] and coverage >= 0.9:
    return "generate", "confident"
if policy.decide(state) == "generate":
    return "generate", policy.name
return "retrieve", ""
```

The hard cap is checked **before** the policy is consulted, so no policy —
including a broken one — can loop. `tests/test_graph.py` proves this with a gate
that always answers "retrieve".

`honor_confidence` is off by default and off for every experiment; the API turns
it on. See [METHODOLOGY.md §4.1](METHODOLOGY.md).

### 3.4 The manual executor

If LangGraph is not installed, `_run_manual` executes the same nodes in the same
order with the same hard cap. It exists so the pipeline stays runnable and
testable without the extra dependency; the compiled graph is the production
path. The compiled graph is cached on the policy object, so it is built once per
policy rather than once per query.

## 4. Policies

All policies satisfy one `Protocol`: `decide(state) -> "retrieve" | "generate"`,
plus a `name`. `build_policy(name, **kwargs)` is the factory used by the
experiment driver and the API.

| Policy | Where | Behaviour |
|---|---|---|
| `FixedPolicy(n=3)` | `policies.py` | retrieve until `iteration >= n` |
| `OneShotPolicy` | `policies.py` | compute `complexity_score` once, map to depth 1–4, commit |
| `CAESPolicy(lam)` | `caes.py` | `ΔQ − λ·ΔC > 0` |
| `ThresholdPolicy` | `caes.py` | `coverage_delta < 0.05 AND coverage > 0.7` |

`CAESPolicy` refuses to construct while λ is unset (`config.LAMBDA is None`),
with an error naming the tuning script. Failing closed here is deliberate: a
silently defaulted λ would produce a plausible-looking but meaningless headline.

Every CAES decision is appended as one JSON line to
`results/caes_decisions.jsonl`:

```json
{"query_id": "...", "iteration": 2, "policy": "caes",
 "coverage_raw": [0.35, 0.62], "coverage_smoothed": [0.35, 0.62],
 "delta_q": 0.162, "delta_c": 0.00121, "lambda_value": 40.0,
 "lambda_times_delta_c": 0.0484, "margin": 0.1136,
 "outcome": "retrieve", "reason": "positive_margin"}
```

Pass `record=False` to suppress logging — the λ sweep does, since sweep
decisions are not experiment decisions.

## 5. The model layer

`llm.py` is the only module that talks to a model provider. Every request
follows one path, identical for both providers:

```
make_key(model, payload)
   ├─ cache hit  → accrue notional cost, return; ledger untouched, $0.00
   └─ cache miss → pre-flight estimate → TRACKER.check_affordable()  ← may raise
                 → invoke with retry/backoff, measure wall clock
                 → read REAL token counts from response `usage`
                 → TRACKER.record_*()  → cache.set()  → return
```

Points worth knowing:

* **The cache key is `sha256(model + json.dumps(payload, sort_keys=True))`.**
  Any change to a prompt, temperature, or `max_tokens` produces a new key, so
  edits never silently reuse stale responses.
* **A response without token counts is fatal.** Measured cost is a core claim
  of the work, so the code raises rather than falling back to an estimate.
  Bedrock reports them in `usage`, Gemini in `usageMetadata`. Gemini's
  *embedding* endpoint reports none at all, so the count is measured with a
  separate `countTokens` call rather than estimated — **[D-23]**.
* **Retries** cover throttling, service-unavailable, timeout, and internal
  errors, with exponential backoff (5 attempts). Non-retryable client errors
  propagate immediately. On Gemini these are HTTP status codes rather than
  error names, and a `429` honours the server-supplied `retryDelay` when one is
  offered — free-tier rate limiting is the binding constraint now, so this path
  is load-bearing rather than defensive.
* **Estimates are used for exactly one thing:** the pre-flight affordability
  check, via a rough characters-per-token ratio. They never reach a result.
* **`DRY_RUN=1`** returns canned responses with no network access. Verifier
  output is synthesised as a diminishing-returns coverage curve with per-query
  variation, so a dry run exercises the gate *realistically* — iteration counts
  genuinely vary — rather than trivially. Dry-run calls record `$0.00` to the
  ledger so it is never polluted with fake spend.
* **Embeddings** go one text per call on both providers; batching in callers is
  for progress reporting and rate-limit pacing, not a batch endpoint. The cache
  is keyed per text, so a partial ingest resumes for free.
* **Only request/response shaping and price constants vary by provider.** The
  cache, the budget gate, notional accounting and the dry-run path are shared
  verbatim. The seam is six small functions — `_build_llm_body`, `_call_llm`,
  `_parse_llm`, `_build_embed_body`, `_call_embed`, `_parse_embed`.
* **`python -m llm --check`** makes two tiny real calls and reports the
  embedding dimension actually returned, whether `temperature=0.0` was accepted,
  and the measured token counts. Run it before any ingest.

### 5.1 Two kinds of cost

| | Tracked in | Meaning | Read by |
|---|---|---|---|
| **actual** | `costs.TRACKER` → `.spend_ledger.json` | money really spent; a cache hit adds nothing | budget guards, spend reporting |
| **notional** | `llm._TOTALS` (in-process) | list price of the measured tokens, cached or not | the CAES gate, `total_usd` in results |

Keeping these separate is what makes a cached re-run reproduce the paid run
exactly. See [METHODOLOGY.md §3.3](METHODOLOGY.md).

`llm.totals()` returns `{notional_usd, actual_usd, latency_ms, calls,
cache_hits}` — monotonic counters, meant to be snapshotted and diffed.

On the Gemini free tier **actual is always $0.00**, so the two columns are not
merely different views of one number — notional is the only cost signal the gate
has. See **[D-22]**.

### 5.2 Budget enforcement

`CostTracker` is a process-wide singleton over a JSON ledger that is:

* **flushed after every recorded call** via atomic temp-file replace, and
  reloaded on construction — a crash cannot reset the running total;
* **refused rather than zeroed** if unreadable — starting fresh on a corrupt
  ledger would silently discard the spend history that the guards depend on.

Three ceilings, all enforced on the same pre-flight path
(`check_affordable`, called *before* the network request). They are inert on the
free tier and retained deliberately — **[D-22]**:

| Ceiling | Constant | Effect |
|---|---|---|
| hard budget | `HARD_BUDGET_USD` | raises `BudgetExceeded` |
| warn threshold | `WARN_BUDGET_USD` | logs a loud warning once |
| per-run allowance | `run_budget(max_usd)` | bounds one experiment invocation |

`run_budget` is a context manager; allowances nest and are all checked.

## 6. The agents

All three are thin functions over `llm.invoke_llm`, with prompts held as
named constants in [agents/prompts.py](agents/prompts.py) so they can be tuned
without touching orchestration.

**Planner** (`max_tokens=100`) — skips the LLM entirely on iteration 1 or when
`missing` is empty/"nothing". Otherwise builds a sub-query from the question,
the titles already retrieved, and the missing fact.

**Verifier** (`max_tokens=200`) — the ΔQ signal source, and the dominant cost
because it fires every iteration. Two consequences shape the code:

* evidence is truncated to `VERIFIER_CHUNK_CHARS = 600` (~150 tokens) per chunk
  before being sent. **This is the main lever on gate overhead** — raising it
  raises ΔC on every iteration of every query;
* parsing is defensive, because one unparsed response is one corrupted point on
  the coverage curve. The ladder is: strip code fences → `json.loads` → extract
  the outermost `{...}` from surrounding prose → one repair call with the bad
  output quoted back → give up and hold the previous coverage with
  `parse_failed=True`.

**Generator** (`max_tokens=256`) — sees more of each chunk than the verifier
(`GENERATOR_CHUNK_CHARS = 1200`), because it has to produce the answer, not just
judge whether one is derivable. Instructed to reply exactly
`insufficient evidence` rather than speculate, so a stopped-too-early gate shows
up as an honest abstention rather than a confident hallucination — and
`metrics.score` scores that abstention as zero.

## 7. Retrieval and ingest

**Ingest** (one paid run, guarded): loads HotpotQA distractor validation, builds
questions and title-deduplicated passages, chunks on whitespace words
(`CHUNK_TOKENS`/`CHUNK_OVERLAP_TOKENS` converted at a words-per-token ratio, so
tokens stay the unit of configuration), prepends the title to each chunk, embeds
in batches, and writes a flat inner-product FAISS index plus `chunks.jsonl`,
`questions.jsonl`, and a `meta.json` provenance stamp.

It **refuses to re-embed** if `data/index.faiss` exists, and refuses more loudly
if `meta.json` says the index is the synthetic dev corpus. `--force` overrides
and re-spends.

**Retriever** loads the index and chunks once as a process-wide singleton, and
asserts that vector count matches chunk count — a mismatch means the two files
came from different builds, which would silently return wrong text for right
hits. `search()` embeds the query through `llm.embed` (so query embeddings
are counted in ΔC like everything else) and returns `Chunk` objects.

## 8. Experiments

### 8.1 Driver

```bash
python -m experiments.run --policy caes --n 150 --max-usd 5 --yes
python -m experiments.run --policy caes --resume
```

* **Pre-flight**: prints projected cost, run allowance, cumulative spend, and
  remaining budget, then stops unless `--yes` is passed.
* **Checkpointing**: one JSON line appended and flushed to
  `results/{policy}_raw.jsonl` after **every** query. A crash at query 130 keeps
  the first 129.
* **`--resume`**: reads the existing file, skips completed query ids, and
  tolerates one torn final line from a mid-write crash.
* **Budget stop**: `BudgetExceeded` and `KeyboardInterrupt` are caught, the
  partial results are already on disk, and the message tells you to raise the
  allowance and resume. Exit code 1 signals an early stop.
* Refuses to append to an existing output file without `--resume`, so two runs
  never silently interleave.

### 8.2 Analysis

`python -m experiments.analyze` reads the raw files (de-duplicating query ids,
keeping the first occurrence, since a resumed run can repeat one) and emits:

| Output | Content |
|---|---|
| `results/main_table.csv` | mean cost, latency, EM, F1, coverage, abstention, iterations per policy |
| `results/headline.json` | paired cost reduction, F1 delta, t-test, bootstrap CIs, parity verdict |
| `figures/fig1_cost_vs_quality.png` | the headline figure, 95% CIs on both axes |
| `figures/fig2_iteration_histogram.png` | where each policy actually stops |
| `figures/fig3_coverage_vs_iteration.png` | empirical diminishing returns |
| `figures/fig4_lambda_sweep.png` | the tuning tradeoff curve |
| stdout | main table + stop-reason breakdown per policy |

Figures use a CVD-validated three-colour categorical palette assigned to
**policy identity, never to rank**, so re-ordering the table never repaints a
series, plus direct labels so identity never rests on colour alone. They are
light-mode only: the destination is a printed paper, so a single committed look
is correct rather than a theme-aware pair.

`scipy` is optional — without it the bootstrap CI is still reported and the
t-test is skipped with a notice.

## 9. Serving

`api.py` is a minimal FastAPI layer: `GET /health` (λ, spend, remaining budget,
index presence) and `POST /query`. Deliberately no auth, rate limiting, or
deployment tooling — its purpose is to substantiate the protocol's API layer.

Two implementation notes:

* Queries are **serialised behind a lock**. Per-iteration cost is metered by
  diffing process-wide counters, so concurrent runs would interleave and
  mis-attribute cost to each other. A real deployment would scope the meter per
  request instead.
* `honor_confidence=True` here, unlike the experiments: this path optimises for
  latency and makes no cross-policy claim.

`BudgetExceeded` maps to 503, a missing index to 503 with build instructions, an
unknown policy to 400.

## 10. Tests

```bash
pytest -q          # 113 tests, no network, no credentials
DRY_RUN=1 python -m smoke

CAES_PROVIDER=bedrock pytest -q               # the other provider, same suite
CAES_PROVIDER=bedrock DRY_RUN=1 python -m smoke
```

Transport-level tests are parametrised over both providers: each runs once
against a fake speaking that provider's real wire format, with the other
provider's transport booby-trapped so a mis-routed call fails loudly rather
than passing silently.

| File | Covers |
|---|---|
| `test_costs.py` | ledger persistence, pre-flight refusal, run budgets, itemisation |
| `test_cache.py` | key stability, hit/miss accounting, corrupt-entry recovery |
| `test_caes.py` | ΔQ/ΔC estimators, smoothing, decision arithmetic, decision log |
| `test_policies.py` | fixed depth, complexity scoring, one-shot commitment |
| `test_graph.py` | node wiring, cost metering, and the hard cap against a broken gate |
| `test_verifier.py` | the parse ladder end to end (fences, prose, repair, give-up), evidence truncation, and the EM/F1/abstention metrics |
| `test_tune_lambda.py` | knee finding, log refinement, flat-F1 and spread detection |
| `test_provider.py` | the provider seam: published prices, request shaping, `temperature=0.0` and measured token counts on both paths, Gemini thinking/429/`countTokens` handling, and a static check that nothing outside `llm.py` calls a provider |
| `conftest.py` | the two fake transports and the provider-parametrised `wired` fixture |

`smoke.py` runs the full graph under all three policies against a tiny synthetic
index (built in-process if no real index exists, so it works on a clean
checkout), then **asserts the ledger did not move**. It refuses to run outside
dry-run mode.

## 11. Extending the system

**Add a policy.** Implement `decide(state) -> "retrieve" | "generate"` with a
`name` attribute, then register it in `build_policy` and in the `--policy`
choices of `experiments/run.py`. The graph, cost metering, and hard cap need no
changes.

**Change a prompt.** Edit the constant in `agents/prompts.py`. This changes the
cache key, so the next run re-spends for the affected calls — expected, and the
reason prompts are versioned in one file. Re-run `calibrate_verifier.py` after
any change to `VERIFIER_PROMPT`.

**Change the cost model.** Update the price constants in `config.py` only. Both
the ledger and the notional accounting derive from them.

**Change the gate's overhead.** `VERIFIER_CHUNK_CHARS` is the lever. Embeddings
are a rounding error on the query path — roughly 0.03% of a three-iteration
query — so ΔC is in practice almost entirely the verifier and planner calls.

### Invariants not to break

1. No `boto3` call or provider HTTP request outside `llm.py`.
2. No prices, model ids, or loop bounds outside `config.py`.
3. Budget checks stay pre-flight; token counts stay measured, never estimated.
4. The gate stays in `verify`; the conditional edge stays a pure read.
5. `MAX_ITERATIONS` stays checked before the policy is consulted.
6. Cache hits never touch the ledger; the gate never reads actual spend.
7. λ stays fail-closed, and the tuner never loads the test split.

[DECISIONS.md §4](DECISIONS.md) carries the full invariant list and the checks
that catch violations.
