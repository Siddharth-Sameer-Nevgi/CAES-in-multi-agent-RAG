# DECISIONS.md — onboarding and design rationale

Read this before changing anything. [README.md](README.md) tells you how to
*run* the system; this file tells you how it *thinks*, why it is built the way
it is, and which parts will silently produce wrong research results if you
"clean them up".

[CHANGELOG.md](CHANGELOG.md) records what landed when, and cross-references the
**[D-n]** decision records below.

---

## 1. What this project is, in 60 seconds

A multi-agent RAG pipeline whose research contribution is a **stopping rule**.

Most iterative-retrieval systems either retrieve a fixed number of times, or
pick a depth up front from the question. This one decides *at every iteration*
whether another retrieval round is worth paying for:

```
continue retrieving only while   ΔQ − λ·ΔC > 0
```

- **ΔQ** — estimated marginal *evidence-quality gain* of the next iteration,
  extrapolated from how much the verifier's coverage score has been improving.
- **ΔC** — measured marginal *execution cost* of the next iteration, in dollars,
  taken from what the previous iterations of this same query actually cost.
- **λ** — the exchange rate between quality and money. Tuned once on held-out
  data, then frozen.

The deliverable is the working system **plus** a three-way comparison on
HotpotQA: CAES vs. fixed-iteration (B1) vs. one-shot routing (B2). The headline
claim is a percentage cost reduction against fixed-depth **at statistically
indistinguishable F1**.

Two consequences follow from "this is a research artifact, not a product", and
they explain most of the unusual code:

1. **Measurement fidelity outranks convenience.** Cost and latency numbers are
   claims in a paper, so the code refuses to estimate what it can measure.
2. **Baseline fidelity outranks CAES looking good.** Several places
   deliberately withhold an advantage from CAES so the comparison stays honest.

---

## 2. The mental model

### Per-query control flow

```
                 ┌──────────────────────── retrieve again ─────────┐
                 │                                                 │
   question ─→ plan ─→ retrieve ─→ verify ─→ [GATE: ΔQ − λ·ΔC > 0?]┘
                 ▲                              │
                 │                              └─ stop ─→ generate ─→ answer
                 └── iteration 1 is free (question used verbatim)
```

- **plan** — iteration 1 returns the question unchanged, with no LLM call.
  Later iterations ask Haiku for a focused sub-query targeting whatever the
  verifier said was `missing`.
- **retrieve** — embed the query, FAISS search, append chunks not already seen.
- **verify** — score how well the accumulated evidence covers the question
  (0.0–1.0), name what is still missing, then **evaluate the gate** and record
  the decision into state.
- **generate** — grounded answer from evidence only, or `"insufficient
  evidence"`.

### What one iteration costs, and what `total_usd` means

This trips people up, so it is worth being precise:

| Quantity | Spans | Used for |
|---|---|---|
| `cost_history[i]` | plan + retrieve + verify of iteration *i* | **ΔC** — the marginal cost of retrieving *again* |
| `total_usd` | the whole query, generation included | the results table and the headline number |

Generation is deliberately **excluded** from ΔC. The gate is deciding "is
another retrieval round worth it?", and generation happens exactly once either
way, so including it would inflate the marginal cost of a decision it does not
depend on. Do not "fix" this.

Mechanically: `node_plan` snapshots `bedrock.totals()`, `node_verify` diffs it.
`initial_state` snapshots for the whole-query figure, `node_generate` diffs it.

### Cost bookkeeping has two separate meanings

| | What it is | Who reads it |
|---|---|---|
| **Actual** (`CostTracker`, `.spend_ledger.json`) | money genuinely spent; a cache hit is $0.00 | budget guards, "am I about to blow $40" |
| **Notional** (`bedrock.totals()["notional_usd"]`) | list-price cost of the call, from real token counts, **cached or not** | the CAES gate, `cost_history`, `total_usd`, the paper |

See **[D-12]** for why this split is load-bearing rather than fussy.

---

## 3. Where to start reading

In this order. You can stop after step 4 and understand 80% of the system.

| # | File | Why it matters | Lines worth reading closely |
|---|---|---|---|
| 1 | `config.py` | Every tunable. Nothing is hardcoded elsewhere. | all of it, it's short |
| 2 | `caes.py` | **The contribution.** ΔQ, ΔC, smoothing, the decision. | `estimate_delta_q`, `smooth_coverage`, `CAESPolicy.decide` |
| 3 | `graph.py` | How a query actually flows; where the hard cap lives. | `make_verify_node`, `evaluate_gate` |
| 4 | `bedrock.py` | The only place that talks to AWS. Cache + budget + metering. | `invoke_llm`, the notional-accounting block |
| 5 | `costs.py` | Ledger, pre-flight gate, `run_budget`. | `check_affordable`, `_flush` |
| 6 | `agents/prompts.py` | The verifier rubric **is** the ΔQ signal. | `VERIFIER_PROMPT` |
| 7 | `agents/verifier.py` | The parse ladder that protects that signal. | `verify`, `_extract_json` |
| 8 | `policies.py` | The two baselines you are being compared against. | `OneShotPolicy` |
| 9 | `experiments/run.py`, `experiments/analyze.py` | How results are produced and read. | `headline()` |

**Fastest way to build intuition:** run the whole thing for free and read the
decision log.

```bash
DRY_RUN=1 python devdata.py
DRY_RUN=1 python -m experiments.run --policy caes --n 20 --lam 40 --yes
head -3 results/caes_decisions.jsonl | python -m json.tool
```

Each line shows ΔQ, ΔC, λ·ΔC, the margin, and the outcome for one gate
evaluation. That file is the paper's central figure.

---

## 4. Invariants — break these and the results are wrong

These are not style preferences. Each one, if violated, produces a system that
still runs and still emits numbers, but the numbers no longer mean what the
paper says they mean.

1. **All Bedrock traffic goes through `bedrock.py`.** A direct `boto3` call
   anywhere else is invisible to the ledger, the cache, and the budget guard,
   and silently corrupts ΔC.
2. **Budget checks happen before the network call, never after.** Enforced by
   `test_budget_exception_fires_before_the_api_call`, which asserts the mocked
   client's call count stays at zero.
3. **Cache hits never record cost to the ledger, and always accrue notional
   cost.** Break the first and re-runs cost money on paper; break the second and
   the gate changes its mind on a warm cache.
4. **Token counts come from the response `usage` field.** Never estimate
   post-hoc. `_estimate_tokens()` exists solely for the pre-flight affordability
   check.
5. **`MAX_ITERATIONS` is enforced in the graph, before the policy is
   consulted.** A gate bug must be incapable of looping.
6. **The tune and test splits never mix.** `splits.py` asserts disjointness;
   `tune_lambda.py` never imports `test_set`.
7. **λ is set once, by tuning, and written into `config.py`.** Re-tuning after
   seeing test results invalidates the headline claim.
8. **Baselines get no capability CAES lacks, and vice versa.** See **[D-11]**.
9. **Prompts live in `agents/prompts.py`.** Editing the verifier rubric changes
   the ΔQ signal — that is a *research* change, so it invalidates any λ tuned
   before it and requires re-running `calibrate_verifier.py`.

### How to check you haven't broken anything

```bash
pytest -q                        # 62 tests, no network, no credentials
DRY_RUN=1 python -m smoke        # must end with "OK: full graph exercised for $0.00"
```

The two highest-value tests to keep passing:

- `tests/test_graph.py::test_max_iterations_is_enforced_against_a_broken_gate`
- `tests/test_costs.py::test_budget_exception_fires_before_the_api_call`

---

## 5. Decision records

Format: **context → decision → why → what was rejected → consequences.**

---

### [D-1] Budget enforcement is pre-flight, from an estimate

**Context.** A cost guard that checks after the call has already spent the
money it was meant to prevent.

**Decision.** `CostTracker.check_affordable(estimated_usd)` runs before every
request and raises `BudgetExceeded`. The estimate uses a crude
`len(text)/3.6` token proxy and the full `max_tokens` as the output bound.

**Why.** Deliberately pessimistic: assuming maximum output means the guard errs
toward stopping early, which is the safe direction when the ceiling is a hard
credit limit.

**Rejected.** Post-hoc checking (too late by definition); a token-counting API
call before each real call (doubles request count to save cents).

**Consequences.** The guard trips slightly before the true ceiling. Accepted —
under-spending is not a failure mode here.

---

### [D-2] A response without a `usage` block is fatal

**Context.** Bedrock normally returns real input/output token counts. If it
didn't, the obvious move would be to fall back to the estimator.

**Decision.** Raise `RuntimeError` instead.

**Why.** Measured ΔC is a central claim. A silent fallback to estimated tokens
would put fabricated numbers into the results with no visible signal.

**Consequences.** A Bedrock response-shape change breaks the run loudly. That is
the intended behaviour. Covered by
`test_missing_usage_block_is_fatal`.

---

### [D-3] A verifier parse failure holds the previous coverage

**Context.** The verifier must return strict JSON. Sometimes models don't.
Parse ladder: strip fences → `json.loads` → regex the outermost `{...}` → one
repair retry → fallback.

**Decision.** On total failure, return `coverage=previous_coverage`,
`parse_failed=True`, and log a warning.

**Why.** Consider the alternatives. Returning `0.0` fabricates a large negative
delta (running-max smoothing absorbs it, but the raw series is corrupted).
Returning `1.0` fabricates success. Holding flat means ΔQ ≈ 0, so the gate reads
"no gain available" and stops — the *conservative* reading of "we don't know".

**Consequences.** Parse failures bias toward stopping early, and are counted in
`parse_failures` on every record so the rate is auditable. If that count is
non-trivial in a real run, fix the rubric rather than the fallback.

---

### [D-4] Abstentions score zero rather than being excluded

**Context.** The generator says `"insufficient evidence"` instead of
speculating. Those responses could be dropped from the metric.

**Decision.** `metrics.score()` returns EM = F1 = 0 and flags `abstained=1.0`.

**Why.** Excluding them makes stopping early look free: a policy could cut cost
by abstaining more and its mean F1 over *answered* questions would go *up*. That
would be a measurement artifact presented as a result.

**Consequences.** Abstention rate is reported as its own column, so the effect
is visible rather than hidden inside F1.

---

### [D-5] `MAX_ITERATIONS` is a graph property, not a policy property

**Context.** Each policy could enforce its own cap.

**Decision.** `evaluate_gate()` checks `iteration >= MAX_ITERATIONS` **before**
calling `policy.decide()`. The manual executor re-checks it after routing.

**Why.** The gate is the experimental variable. An experimental variable must
not also be the safety mechanism — a bug in the contribution would otherwise
become an unbounded spend loop.

**Rejected.** Relying on LangGraph's `recursion_limit` alone: it is a framework
backstop with an opaque error, not a domain guarantee.

**Consequences.** `stop_reason == "max_iter"` is ambiguous between "the policy
wanted more" and "the policy happened to stop exactly there" — read it together
with the policy name. Proven by the runaway-gate test.

---

### [D-6] The gate runs at the end of `verify`, not inside the conditional edge

**Context.** The spec describes the gate as a conditional edge. Implemented
literally, every `stop_reason` came out as `max_iter`.

**Decision.** `make_verify_node` computes the decision and writes `_route` and
`stop_reason` into state. `route_from_state` is a pure read.

**Why.** LangGraph does not merge state mutated inside an edge function back
into graph state. The edge is still a conditional edge — it just reads a
decision made one node earlier.

**Consequences.** The gate needs the *post-verification* view of state, so
`node_verify` merges its own updates before evaluating:
`merged = {**state, **updates}`. If you add a state field the gate reads, it
must be in `updates` before that merge.

---

### [D-7] Coverage is smoothed with a running max before differencing

**Context.** Coverage genuinely decreases sometimes — new documents introduce a
second plausible entity and the verifier correctly becomes less certain.

**Decision.** `smooth_coverage()` takes a running max; ΔQ is computed on the
smoothed series. Both series are logged on every decision.

**Why.** Differencing the raw series reads a transient dip as negative gain and
forces a premature stop on exactly the queries that are still making progress.

**Rejected.** Clamping ΔQ at zero only (loses the recovery signal on the
following iteration); a moving average (adds a window-size parameter and lags
real gains).

**Consequences.** ΔQ can never be negative, so the gate stops on *flat* rather
than on *declining* coverage. Logging both series means the smoothing is
auditable rather than hidden.

---

### [D-8] `tune_lambda.py` refuses to recommend on a flat curve

**Context.** The knee is found by maximising F1-per-dollar. If F1 is identical
at every λ, that reduces to "pick the cheapest".

**Decision.** `f1_is_flat()` (spread < 0.01) → print the diagnosis, exit 1,
recommend nothing.

**Why.** Silently returning the cheapest λ would recommend a value that stops at
one iteration on every query, and it would look like a tuned parameter.

**Consequences.** Under `DRY_RUN` the sweep always reports degenerate, because
canned answers make F1 meaningless. That is correct — the sweep is only
meaningful with real responses.

---

### [D-9] Figures use a validated three-colour palette keyed to policy identity

**Context.** Three series, print destination, colourblind readers.

**Decision.** Slots 1–3 of a CVD-validated categorical palette
(`#2a78d6` CAES, `#eb6834` Fixed, `#1baf7a` One-shot), mapped to **policy
identity, never to rank**. Direct labels on every series in addition to the
legend. Light mode only. One y-axis per chart.

**Why.** The all-pairs colourblind check passes for exactly three slots — which
is exactly the number of policies. Mapping by rank would repaint series when a
table is re-sorted, misleading anyone who learned "CAES is blue". One slot sits
below 3:1 contrast on the light surface, and direct labels are the required
relief, so identity never depends on colour alone.

**Consequences.** Adding a fourth policy to the figures is **not** a matter of
appending a colour — the fourth slot fails the all-pairs floors. Facet into
small multiples instead. Light-mode-only is deliberate: these are paper figures,
not a web dashboard.

---

### [D-10] The API serialises queries behind a lock

**Context.** Per-iteration cost is metered by diffing process-wide counters in
`bedrock.py`. Concurrent graph runs would interleave and mis-attribute cost.

**Decision.** One `threading.Lock` around `run_query` in `api.py`, with the
reason in a comment.

**Why.** The API exists to substantiate the protocol's API layer for the paper.
Correct cost attribution matters more than throughput; a per-request meter is
the right fix but is not needed for the claim.

**Consequences.** Throughput is one query at a time. Listed under "next" in the
changelog.

---

### [D-11] The confidence short-circuit is off for experiments, on for the API

**Context.** `stop_reason` includes `"confident"`. The verifier can report high
coverage *and* confidence, which is an obvious cheap early exit.

**Decision.** `honor_confidence` defaults to `False`. `graph.run_query` exposes
it; `experiments/run.py` never sets it; `api.py` sets it `True`.

**Why.** Giving CAES a second, orthogonal stopping rule that Fixed and One-shot
lack would confound the contribution — the cost reduction could then be
attributed to the confidence check rather than to the ΔQ − λ·ΔC gate. The API
makes no cross-policy claim, so it can prefer latency.

**Rejected.** Giving every policy the short-circuit — that would make
`FixedPolicy` no longer fixed, destroying baseline B1.

**Consequences.** CAES's measured advantage is attributable to the gate alone.
Two tests pin both halves of this behaviour.

---

### [D-12] Notional cost accounting exists alongside the ledger

**Context.** The spec says ΔC is *measured* cost. The cache makes re-runs free.
Those two facts collide.

**Decision.** `bedrock.py` maintains process-wide counters accruing **notional**
cost — list price computed from the real measured token counts, which the cache
persists — separately from the ledger's **actual** spend. The gate, `cost_history`
and `total_usd` read notional; budget guards read actual.

**Why.** Without this, on a warm cache ΔC → 0, so `λ·ΔC` → 0, so the margin is
always positive, so the gate runs to `MAX_ITERATIONS` on every query. A cached
re-run would produce **different decisions and different results** than the paid
run. Notional cost makes the experiment reproducible from cache — which is the
entire point of having the cache.

**Rejected.** Disabling the cache during experiments (throws away the main cost
control); recording cache hits to the ledger (makes the budget guard fire on
money never spent).

**Consequences.** Two cost numbers exist and they mean different things. Papers
quote notional; the credit balance follows actual. Cached embeddings persist
`in_tokens` specifically so their notional cost survives a cache hit — do not
drop that field.

---

### [D-13] Raw `boto3` rather than the Anthropic Bedrock SDK client

**Context.** `AnthropicBedrockMantle` is the newer path for Claude on Bedrock.

**Decision.** A single `boto3.client("bedrock-runtime")` wrapper.

**Why.** Titan Text Embeddings V2 is not an Anthropic model and is only
reachable through `bedrock-runtime`. Using both clients would mean two entry
points, and the ledger's completeness depends on there being exactly one.

**Consequences.** Request/response bodies are hand-built
(`anthropic_version: "bedrock-2023-05-31"`). If Claude access moves to a client
the embeddings model also supports, revisit.

---

### [D-14] Splits are derived deterministically, not stored

**Context.** Tune/test disjointness is a correctness property of the research,
not a convenience.

**Decision.** `splits.py` sorts by id, shuffles with a fixed seed
(`SPLIT_SEED = 20240917`), slices 50 then 150, and **asserts** disjointness on
every call.

**Why.** A stored split file can drift from the code, be regenerated by
accident, or be edited. Derivation plus assertion cannot.

**Consequences.** Changing `SPLIT_SEED`, `N_TUNE`, `N_TEST`, or the question set
silently changes both splits — so it invalidates any λ tuned before the change.

---

### [D-15] Iteration 1 spends no planner call

**Context.** A planner that rewrites the question before anything has been
retrieved has no information to rewrite it *with*.

**Decision.** `plan()` returns the question verbatim when `iteration <= 1`, or
when there is no prior evidence, or when `missing` is empty/`"nothing"`.

**Why.** Pure overhead otherwise. The verifier fires every iteration and already
dominates spend; adding a no-information LLM call to every query inflates the
cost of *all three* policies equally, which is not free — it shrinks the
measurable difference between them.

**Consequences.** A single-iteration query costs one embed + one verify + one
generate. That is the floor for every policy.

---

### [D-16] `LAMBDA = None` fails closed

**Context.** λ has no defensible default.

**Decision.** `config.LAMBDA` ships as `None`; `CAESPolicy.__init__` raises
`ValueError` with instructions.

**Why.** Any placeholder would eventually be run as though it were tuned. A
guessed λ makes the headline number meaningless while looking completely normal.

**Consequences.** CAES cannot run before tuning. `tune_lambda.py` and `smoke.py`
pass λ explicitly; both say in a comment that it is illustrative.

---

### [D-17] A manual executor mirrors the LangGraph path

**Context.** LangGraph is a heavy dependency for a state machine this small.

**Decision.** Keep LangGraph as the production path; `_run_manual()` runs the
same node functions in the same order, with the same hard cap, and is used only
if the import fails.

**Why.** Keeps the pipeline runnable and testable on a minimal install without
maintaining two divergent implementations — both call the identical node
functions and the identical `evaluate_gate`.

**Consequences.** If you add a node, add it to both. The duplication is ~10
lines and deliberate.

---

### [D-18] The verifier sees less evidence than the generator

**Context.** Both need the retrieved chunks.

**Decision.** Verifier truncates to `VERIFIER_CHUNK_CHARS = 600` (~150 tokens
per chunk); generator uses `GENERATOR_CHUNK_CHARS = 1200`.

**Why.** The verifier fires *every iteration* and is the dominant cost, so its
input size is the main lever on gate overhead. It only has to judge whether an
answer is derivable. The generator fires once and has to actually produce the
answer, so it gets more.

**Consequences.** The verifier can in principle under-score evidence whose
decisive fact sits past 600 characters. Raising `VERIFIER_CHUNK_CHARS` raises ΔC
for every policy — treat it as a research parameter, not a tuning knob.

---

### [D-19] `temperature=0.0` on every LLM call

**Decision.** Default in `invoke_llm`.

**Why.** Reproducibility, and cache effectiveness — the cache key is the request
body, so deterministic sampling maximises hit rate across re-runs. Haiku 4.5
accepts sampling parameters (newer Claude models reject them).

**Consequences.** If the model is ever changed to one that rejects
`temperature`, this must be removed. Model choice is fixed by the cost
constraints, so that is not imminent.

---

### [D-20] Synthetic development data is a separate script with a loud stamp

**Context.** The full pipeline needs a corpus, but downloading HotpotQA and
embedding it costs money and time.

**Decision.** `devdata.py` generates a synthetic corpus and question set, writes
the same filenames `ingest.py` writes, and stamps `data/meta.json` with
`"synthetic": true`. It refuses to run outside `DRY_RUN`. `ingest.py` detects
the stamp and exits non-zero with a specific message.

**Why.** Without the stamp-and-detect pair, a developer who dry-ran the pipeline
would later find the real ingest refusing to build with the generic "index
already exists" message, and might `--force` past it or waste an afternoon.

**Consequences.** `data/` is single-tenant — synthetic and real cannot coexist.
Delete `data/` between the two.

---

### [D-21] λ grid resolution is a research-correctness issue, not tuning convenience

**Context.** The original grid was `[0.1, 1, 10, 100, 1000]` — decade-spaced.
Sweeping it under `DRY_RUN` produced uniform iteration counts at almost every
value: λ=20 → all 40 queries stop at iteration 3; λ=90, 200 → all stop at 2;
λ=2000 → all stop at 1. Only λ≈60 showed real spread.

**Decision.** Coarse grid is now half-decade-spaced,
`[1, 3, 10, 30, 100, 300, 1000]`. Refinement is **log-spaced** about the knee
(`centre × [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]`) rather than linear. Every swept
λ records its iteration distribution and largest-bucket share in
`results/lambda_sweep.csv`, and a degenerate spread at the recommended λ prints
a warning naming the best-spread alternative.

**Why this is correctness, not polish.** The gate's sensitive band — where
`λ·ΔC` is the same order as ΔQ, so the decision actually depends on the query —
is narrow. Outside it, the margin's sign is constant and CAES degenerates into a
fixed-depth policy that happens to be spelled differently. A decade-spaced grid
steps clean over that band.

The failure is silent and it corrupts the *claim*, not the code. At a degenerate
λ the pipeline runs, F1 and cost are real, and the headline cost reduction may
look excellent. But the paper's central figure — CAES spread across 1–5 versus
Fixed flat at N — becomes a single bar, which is visually identical to a fixed
policy and destroys the per-iteration-granularity argument. The system would be
demonstrating the opposite of its thesis while reporting a healthy number.

**Why log-spaced refinement.** The coarse grid is log-spaced, so the knee is
located to within a multiplicative factor. Linear refinement around such a
centre oversamples above it and undersamples below. Measured on the synthetic
corpus, the spread band is λ∈[40,70] with the best spread at λ=50 (55% largest
bucket). The coarse grid brackets that band without landing in it — but
`refine_grid(30)` reaches 42 and 60, and `refine_grid(100)` reaches 50 and 70.
Log refinement from *either* bracketing point covers the band; linear refinement
from λ=30 would have topped out at 120 and missed the lower half entirely.

**Why the spread guard warns instead of blocking.** `f1_is_flat()` blocks
because a flat quality curve means there is genuinely nothing to tune. A
degenerate spread is different: the λ may be legitimately optimal, and real data
may spread where synthetic data does not. Blocking would substitute the script's
judgement for the researcher's on a question — what the experiment should
demonstrate versus what it costs — that is not the script's to make. So it
reports both candidates and stops there.

**Rejected.** Optimising directly for spread (it is a property of the figure,
not of the method — maximising it would be tuning to make the picture look
good); auto-selecting the best-spread λ (same objection, silently applied);
widening the grid further (more paid sweep points for resolution that
refinement already provides).

**Consequences.** The coarse sweep is 7 points instead of 5, and refinement adds
up to 6 more, so a full sweep costs roughly 2.5× the original. That is the main
cost driver in Phase 4 and it is deliberate: λ is the one number the whole
contribution hangs on. Guarded by `tests/test_tune_lambda.py`, which pins the
half-decade grid spacing, the log-spacing of refinement, and both degeneracy
checks.

---

## 6. Traps

Things that will cost you time, in rough order of likelihood.

| Trap | Symptom | Fix |
|---|---|---|
| Bedrock model access not requested | `AccessDeniedException` on the first real call | Enable Claude Haiku 4.5 **and** Titan Embeddings V2 in the Bedrock console, `us-east-1` |
| Synthetic data still in `data/` | `ingest.py` exits 2 mentioning devdata | `rm -rf data/` |
| `CAESPolicy` raises `ValueError` | λ never tuned | Run `tune_lambda.py`, write the value into `config.py` |
| `experiments/run.py` exits 2 immediately | output file exists and `--resume` not passed | Add `--resume`, or delete `results/{policy}_raw.jsonl` |
| Nothing happens, exits 0 | pre-flight confirmation | Add `--yes` |
| Everything says `$0.00` | `DRY_RUN=1` is still exported | `unset DRY_RUN` |
| Every CAES query stops at the same iteration | λ is far outside the sensitive region | Check `caes_decisions.jsonl`: if `lambda_times_delta_c` dwarfs `delta_q` everywhere, λ is too high |
| λ sweep says "DEGENERATE" | F1 flat — usually still in `DRY_RUN` | Run with real responses |
| λ sweep warns "degenerate on spread" | recommended λ puts >90% of queries in one iteration bucket | Compare against the best-spread λ it names; see **[D-21]** |
| Coverage all lands in one band | rubric not discriminating | Sharpen `VERIFIER_PROMPT`; this is a Phase 2 blocker, not cosmetic |
| Ledger seems stuck | it is cumulative and persistent, by design | `python -c "from costs import TRACKER; print(TRACKER.summary())"` |

### The one that is hardest to notice

If you add a Bedrock call that bypasses `bedrock.py`, everything keeps working.
Tests pass. Figures render. The only symptom is that ΔC is understated, so the
gate retrieves more than it should, and the paper's cost numbers are quietly
wrong. There is no test that can catch this — it is why invariant 1 exists.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **ΔQ / `delta_q`** | Estimated coverage gain the *next* iteration would deliver. `max(0, last_delta × DECAY_FACTOR)`, on the smoothed series. Returns `1.0` when history is too short — "unknown, allow one more". |
| **ΔC / `delta_c`** | Mean observed per-iteration notional cost for this query, in USD. |
| **λ / `LAMBDA`** | Quality-per-dollar exchange rate. Higher λ ⇒ cost matters more ⇒ stops sooner. |
| **margin** | `ΔQ − λ·ΔC`. Positive ⇒ retrieve again. |
| **coverage** | Verifier's 0–1 judgement of whether the evidence answers the question. The ΔQ signal. |
| **notional cost** | List-price cost from real token counts, charged whether or not the call was cached. What the gate and the paper use. |
| **actual cost** | Money genuinely spent. What the ledger and budget guards use. |
| **B1 / Fixed** | Baseline: always N iterations. |
| **B2 / One-shot** | Baseline: depth chosen before iteration 1 from a complexity score, then committed. |
| **stop_reason** | `caes` \| `fixed` \| `oneshot` \| `threshold` \| `max_iter` \| `confident`. |
| **abstention** | Generator returned `"insufficient evidence"`. Scored zero, counted separately. |

---

## 8. Open questions

Genuinely undecided, flagged so nobody assumes they were settled.

1. **Is `DECAY_FACTOR = 0.6` right?** It was taken from the spec and never
   fitted. The honest version is to fit the decay against observed coverage
   trajectories from a real run, and report the sensitivity.
2. **ΔC is a mean, so it is backward-looking.** Iteration 4 may cost more than
   iteration 1 because the evidence set — and therefore the verifier prompt —
   has grown. A monotone cost model might be more accurate. Check
   `cost_history` from a real run before assuming it matters.
3. **Retrieval failure and gate failure are currently indistinguishable.** A low
   F1 could mean the gate stopped too early *or* that retrieval never surfaced
   the supporting passage. Recording precision-at-k against HotpotQA's
   `supporting_titles` would separate them. This is the highest-value addition
   to the results section.
4. **One-shot's complexity score is hand-built.** The spec permitted a small
   Haiku call instead. The current heuristic is free and deterministic, but if
   B2 routes nearly every question to the same depth on real data, the baseline
   is weak and a reviewer will say so — check the depth distribution.
5. **`FixedPolicy(n=3)` is the headline baseline.** Whether N=3 is the fair
   comparison, or whether the N that matches CAES's mean iteration count is
   fairer, is a framing decision worth making explicitly before writing up.
