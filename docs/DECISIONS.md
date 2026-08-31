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
| 4 | `llm.py` | The only place that talks to a model provider. Cache + budget + metering. | `invoke_llm`, the notional-accounting block |
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

1. **All model traffic goes through `llm.py`.** A direct `boto3` call or
   HTTP request to a provider anywhere else is invisible to the ledger, the
   cache, and the budget guard, and silently corrupts ΔC. This holds for both
   providers; `config.PROVIDER` selects request shaping *inside* `llm.py` and
   nowhere else. See **[D-22]**.
2. **Budget checks happen before the network call, never after.** Enforced by
   `test_budget_exception_fires_before_the_api_call`, which asserts the mocked
   client's call count stays at zero.
3. **Cache hits never record cost to the ledger, and always accrue notional
   cost.** Break the first and re-runs cost money on paper; break the second and
   the gate changes its mind on a warm cache.
4. **Token counts come from the response.** Never estimate post-hoc.
   `_estimate_tokens()` exists solely for the pre-flight affordability check.
   Bedrock reports them in `usage`, Gemini in `usageMetadata`; Gemini's
   embedding endpoint reports none, which is why **[D-23]** exists.
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
pytest -q                        # 113 tests, no network, no credentials
DRY_RUN=1 python -m smoke        # must end with "OK: full graph exercised for $0.00"

# and the same two under the other provider
CAES_PROVIDER=bedrock pytest -q
CAES_PROVIDER=bedrock DRY_RUN=1 python -m smoke
```

Transport-level tests run once per provider against a fake that speaks that
provider's real wire format, and the provider *not* under test is booby-trapped,
so a shaping bug that routed a call to the wrong transport fails loudly.

The two highest-value tests to keep passing:

- `tests/test_graph.py::test_max_iterations_is_enforced_against_a_broken_gate`
- `tests/test_costs.py::test_budget_exception_fires_before_the_api_call`
- `tests/test_provider.py::test_llm_is_the_only_module_that_calls_a_provider`

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

**Context.** Both providers return real input/output token counts — Bedrock in
`usage`, Gemini in `usageMetadata`. If one didn't, the obvious move would be to
fall back to the estimator.

**Decision.** Raise `RuntimeError` instead.

**Why.** Measured ΔC is a central claim. A silent fallback to estimated tokens
would put fabricated numbers into the results with no visible signal.

**Consequences.** A response-shape change on either provider breaks the run
loudly. That is the intended behaviour. Covered by
`test_missing_token_counts_are_fatal_on_both_providers`, which runs against both.
Gemini's *embedding* endpoint returns no count at all, which is a separate
problem — see **[D-23]**.

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

**Decision.** `llm.py` maintains process-wide counters accruing **notional**
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
drop that field. Since the move to the Gemini free tier, actual is *always*
zero, which makes this separation load-bearing rather than merely useful — see
**[D-22]**.

---

### [D-13] Raw `boto3` rather than the Anthropic Bedrock SDK client

**Context.** `AnthropicBedrockMantle` is the newer path for Claude on Bedrock.

**Decision.** A single `boto3.client("bedrock-runtime")` wrapper.

**Why.** Titan Text Embeddings V2 is not an Anthropic model and is only
reachable through `bedrock-runtime`. Using both clients would mean two entry
points, and the ledger's completeness depends on there being exactly one.

**Consequences.** Request/response bodies are hand-built
(`anthropic_version: "bedrock-2023-05-31"`). If Claude access moves to a client
the embeddings model also supports, revisit. The same reasoning applied again at
**[D-22]**: the Gemini path is raw REST rather than `google-generativeai`, so
there is still exactly one entry point and the cache key is still the request
body we built.

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
accepts sampling parameters (newer Claude models reject them). Gemini takes it
as `generationConfig.temperature`.

**Consequences.** If the model is ever changed to one that rejects
`temperature`, this must be removed — and that is a methodology change, not a
config tweak, because cache correctness depends on determinism. On
`gemini-2.5-flash` there is a second determinism risk that `temperature` does
not cover: thinking is on by default and its length varies, so **[D-22]**
disables it explicitly. Pinned for both providers by
`test_temperature_zero_is_sent_on_both_providers`.

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

### [D-22] The evaluation provider is Google Gemini, and actual spend is now structurally zero

**Context.** Bedrock model invocation is blocked account-wide on this AWS
account. `ListFoundationModels` succeeds and returns 120 models, but
`InvokeModel` fails with `ValidationException: Operation not allowed` for
`amazon.titan-embed-text-v2:0`, `amazon.nova-lite-v1:0`, **and** through the
AWS console playground — so it is not model-specific, not SDK-related, and not
fixable from our side. A support case is open and unresolved.

**Decision.** `bedrock.py` becomes `llm.py`, gaining a `config.PROVIDER` switch
(`"bedrock" | "gemini"`, defaulting to `"gemini"`). Only request/response
shaping and price constants vary by provider; the cache, the pre-flight budget
gate, notional accounting, the dry-run path and the single-entry-point
invariant are shared verbatim. The Bedrock path is kept fully working.

**Why the contribution is unaffected.** The contribution is the stopping rule.
ΔQ − λ·ΔC needs measured token counts and published per-token prices; every
commercial provider supplies both. The provider is a threat-to-validity line
([METHODOLOGY.md](METHODOLOGY.md) §10, External), not part of the claim.

**Why the Bedrock path stays.** The support case may resolve, and a two-provider
robustness result would strengthen the paper. A dead code path would be a cost;
a *tested* one is an asset — `tests/conftest.py` runs every transport-level test
once per provider against a fake speaking that provider's wire format, and
booby-traps the other transport so a mis-routed call fails loudly.

**The consequence that actually matters: actual spend is now $0.00, always.**
Both `gemini-2.5-flash` and `gemini-embedding-001` bill nothing on the free
tier. So:

* **ΔC is entirely list-price notional.** [D-12]'s machinery already handles
  this correctly — but the reason it matters has changed. Under Bedrock,
  notional cost existed to keep a *cached* re-run making the same decisions as
  the paid run. Under Gemini it is the only cost signal that exists at all: if
  the gate read actual spend it would read zero on every call, on every run,
  paid or cached, and CAES would degenerate to `MAX_ITERATIONS` on every query.
* **The ledger and `HARD_BUDGET_USD` become vestigial safety rather than active
  constraints.** They are kept. The Bedrock path may return, and a guard that
  only exists when it is needed is a guard that is wrong when it is needed.
* **The paper must report costs as list-price-derived, not billed.** No spend
  was incurred. Saying "cost" without that qualifier would misdescribe what was
  measured. §10 (Construct) states this explicitly.

**The binding constraint moved.** It is no longer a credit balance; it is
free-tier rate limiting. `llm.py` gains a client-side RPM pacer
(`GEMINI_MAX_RPM`) and retries 429 `RESOURCE_EXHAUSTED` honouring the
server-supplied `retryDelay`. See §11.

**Models substituted.** The migration plan named `gemini-2.0-flash` and
`text-embedding-004`. Both were retired before this landed: 2.0 Flash is listed
under "Previous models / Shut down", and `text-embedding-004` is gone from the
model list entirely. The nearest live equivalents in the same tier are
`gemini-2.5-flash` and `gemini-embedding-001`, confirmed 2026-08-31.

**Two Gemini specifics that are cost-correctness issues, not config:**

* **Thinking is ON by default on `gemini-2.5-flash`, and thinking tokens bill as
  output.** Left on they would inflate ΔC and, being variable in length, would
  undermine the determinism [D-19] relies on for cache correctness. Disabled via
  `thinkingConfig.thinkingBudget = 0`. `thoughtsTokenCount` is nonetheless
  folded into the output count, so an ignored budget surfaces as measured cost
  rather than as a silently understated ΔC.
* **`EMBED_DIM` changes from 1024 to 768.** `gemini-embedding-001` returns 3072
  by default and supports Matryoshka truncation. 768 keeps a flat index over
  ~20k chunks near 60 MB rather than 250 MB, which matters on the intended
  t3.micro host. Truncated vectors are **not** unit-norm as returned, so
  `llm.embed` re-normalises — without that, `IndexFlatIP` stops meaning cosine.
  `Retriever` now refuses to load an index whose dimension disagrees with
  `config.EMBED_DIM`, and warns when `meta.json` names a different embedding
  model: a stale index is otherwise a *silent* retrieval failure, not an error.

**Rejected.** Waiting on the support case (unbounded, and the project cannot
wait). Abandoning AWS (S3, EC2 and CloudWatch are unaffected and stay in the
deployment — see the README; Bedrock was only the model-serving layer). Deleting
the Bedrock path (throws away the robustness result and the two-provider tests).
Using the `google-generativeai` SDK (a second entry point to keep honest, and
the cache key is the request body, which raw REST keeps under our control —
same reasoning as [D-13]).

**Consequences.** Two providers to keep passing, which the test matrix pays for.
Every price in `config.py` now carries a provider prefix, and the neutral names
`PRICE_LLM_*` / `MODEL_*` / `EMBED_DIM` are resolved once by
`config.provider_settings()`. λ tuned on one provider does not transfer to the
other: ΔC differs by roughly 3× on input and 2× on output, and the sensitive
band moves with it.

---

### [D-23] Gemini's embedding endpoint returns no token count, so the count is measured separately

**Context.** Invariant 4 says token counts are read from the response, never
estimated. Gemini's `:embedContent` returns `{"embedding": {"values": [...]}}`
and nothing else — no token count of any kind. Bedrock's Titan response carries
`inputTextTokenCount`; Gemini has no equivalent field.

Three options, none free of cost:

1. **Raise, as [D-2] does for LLM calls.** Consistent, and makes the system
   unbuildable — every embedding call fails. Rejected: an invariant that
   forbids the only available implementation is not an invariant, it is a stop.
2. **Estimate from character count.** One line, and it puts a fabricated number
   into ΔC with no visible signal. This is precisely the failure [D-2] exists to
   prevent.
3. **Measure with a separate `:countTokens` call.** Preserves "measured, never
   estimated" at the price of one extra request per *uncached* text.

**Decision.** Option 3, controlled by `config.EMBED_TOKENS_MODE`
(`"measured"`, the default). `"estimated"` is available, logs a warning naming
the invariant it breaks on every affected call, and must be opted into
knowingly.

**Refined 2026-08-31, once the real quota was known (see [D-24]).** Option 3 as
originally written doubled a 4,500-request ingest against a 1,000/day embedding
quota — nine days instead of four and a half. The refinement rests on a
distinction the original framing missed:

> **Corpus embeddings are not on the ΔC path. Query embeddings are.**

`retrieval.py` embeds a query on every iteration, and that cost *is* part of
measured ΔC — so it stays measured per call, no change. Corpus embeddings are a
one-time ingest cost whose **per-chunk** attribution reaches no reported number
anywhere; only the aggregate is ever printed or stored. So ingest now:

* measures the aggregate **exactly**, with one batched `:countTokens` per batch
  of 50 chunks — 90 requests instead of 4,500, and the reported ingest cost is
  a measured number, not an estimate;
* attributes per-chunk tokens with the estimator, and stamps
  `per_chunk_tokens_estimated: true` into `data/meta.json` so the choice is
  auditable from the artifact.

Invariant 4 is preserved everywhere it applies. This is not the invariant being
relaxed — it is the invariant being applied to the quantity it was written
about.

**Why measured is worth the request.** `countTokens` is free and its result is
written into the same cache entry as the embedding, so it is paid once per
unique text and never again — a re-run or a resumed ingest costs nothing extra.
The query embedding is on the gate's ΔC path (`retrieval.py` embeds a query
every iteration), so this is not merely an ingest-time concern.

**The cost is rate limit, not money.** Free-tier limits are per-account and no
longer published per model, and embedding the corpus doubles the request count
against them. That is the real risk in this decision, and it is why
`EMBED_TOKENS_MODE` exists as a switch rather than as a constant: if ingest
proves infeasible against the account's actual RPD, flipping to `"estimated"`
is a *research* trade-off — a stated deviation from invariant 4 affecting a
term worth well under 1% of ΔC — and should be recorded as such, not made
silently. Measure the real limits before deciding.

**Consequences.** Cold ingest issues one embedding request per chunk plus one
batched count per batch (~1.02 requests per chunk, not 2). Warm ingest issues
none. A `countTokens` response without `totalTokens` is fatal, on the same
reasoning as [D-2]. Guarded by
`tests/test_provider.py::test_gemini_embed_token_count_is_measured_not_estimated`
and `::test_gemini_countTokens_without_a_total_is_fatal`.

---

### [D-24] Free-tier daily quotas set the experiment's scale, and one of them cost us corpus size

**Context.** [D-22] moved model serving to Gemini because Bedrock invocation is
blocked. That assumed the free tier's binding constraint was rate — requests per
*minute* — which pacing handles. The account's actual quotas, read off
`aistudio.google.com/rate-limit` on 2026-08-31, are:

| Model | RPM | TPM | **RPD** |
|---|---|---|---|
| `gemini-2.5-flash` | 5 | 250K | **20** |
| `gemini-3.5-flash-lite` | 15 | 250K | **500** |
| `gemini-embedding-001` | 100 | 30K | **1,000** |

**Requests per day is the binding constraint, and it is not close.** At 20 RPD
the full experiment — roughly 7,200 LLM calls across baselines, the λ sweep and
the test runs — needs **360 days**. Pacing cannot help: the limit is not how
fast you ask, it is how many times.

**Decision, in three parts.**

**1. The LLM becomes `gemini-3.5-flash-lite`.** It lists at the same
$0.30/$2.50 per 1M as `gemini-2.5-flash` and allows **25× the daily requests**
(500 against 20). Same ΔC economics, 25× the throughput; on the free tier this
is not a trade. The experiment drops from 360 days to ~15.

**2. `CORPUS_SAMPLE_SIZE` drops from 2000 to 500.** Ingest embeds the whole
corpus up front against a separate 1,000/day embedding quota. 2000 questions is
~18k chunks — and with [D-23]'s `countTokens` call, ~36k requests, so **36 days
of ingest for a 15-day experiment**. 500 questions is ~4.5k chunks and ~9 days.

**This one is a research change and is not free.** What shrinks is the
*distractor pool*, not the question count: the evaluation split is still 50 tune
+ 150 test, because those need only 200 of the sampled questions. But HotpotQA's
difficulty comes substantially from its distractors, so a thinner pool makes
retrieval easier and **inflates F1**.

Why it is nonetheless acceptable: it inflates F1 for **all three arms
identically**. The claim is a *relative* one — CAES's cost against fixed-depth
at indistinguishable F1 — and relative comparisons survive a uniformly easier
task. What does not survive is external validity: the absolute F1 numbers are
not comparable to published HotpotQA results, and the cost reduction is measured
on a smaller retrieval problem than the one the literature uses. Recorded in
METHODOLOGY §10 (External).

**3. Daily-quota exhaustion becomes a first-class outcome, not a crash.**
`llm.QuotaExhausted` is raised on a 429 whose `QuotaFailure.quotaId` names a
per-day limit (or whose `retryDelay` exceeds five minutes — no per-minute limit
asks you to wait an hour). It is deliberately **not retried**: a per-minute
limit clears in seconds, a per-day limit does not clear until tomorrow, and
burning five backoff attempts on it wastes time and teaches nothing.

`ingest.py`, `experiments/run.py` and `calibrate_verifier.py` each catch it and
exit cleanly with resume instructions. Nothing is lost, because the disk cache
already makes completed work free to replay — a resumed ingest re-reads its
finished chunks from disk and spends quota only on new ones. This is the cache
earning its keep for a reason [D-12] never anticipated.

**Why the pacer is now per model.** `GEMINI_MAX_RPM` was a single shared 15 —
three times `gemini-2.5-flash`'s real 5 RPM and a sixth of the embedding
model's 100. The two models draw on **separate** quota buckets, so one shared
pacer necessarily either overran one limit or throttled the other for nothing.
Split into `GEMINI_LLM_RPM` / `GEMINI_EMBED_RPM`, tracked in separate buckets.

**Why thinking control is now keyed by model.** The 2.5 family disables thinking
with `generationConfig.thinkingConfig.thinkingBudget`; the 3.x family moved to
`thinkingLevel`. Sending the wrong field is a 400 on **every** call. 3.x Lite
models have thinking off by default, so `GEMINI_THINKING_CONFIG` maps them to
`None` — send no thinking field at all. That is an assumption about a default
rather than an instruction, so it is verified rather than trusted: the preflight
reports output tokens for a one-word reply, and `_parse_llm` folds any non-zero
`thoughtsTokenCount` into the output count, so unexpected thinking surfaces as
measured cost instead of an understated ΔC.

**Rejected.** Enabling billing (~$5 for the whole project, and it would have
made the ledger real and the costs *billed* rather than notional — declined in
favour of staying free). Keeping the 2000-question corpus (36-day ingest).
Falling back to `EMBED_TOKENS_MODE="estimated"` to halve ingest (breaks
invariant 4 to save four days; the corpus reduction buys more for less
principle). Shrinking the *evaluation* split instead of the corpus (destroys the
paired test and the bootstrap CI — the corpus is the right thing to cut because
it degrades external validity rather than statistical validity).

**Consequences.**

* Absolute F1 will read high relative to published HotpotQA baselines. Say so
  when reporting; the comparison is internal.
* **`gemini-3.5-flash-lite` is a weaker verifier, and the verifier is the
  instrument (METHODOLOGY §6).** Task 5 calibration is exactly the test of
  whether it discriminates coverage into usable bands. If it fails there, this
  decision is the first thing to revisit — and per invariant 9, changing the
  rubric to compensate is a research change that invalidates any λ tuned before
  it. Calibration is only ~30 LLM calls, so it is cheap to re-run.
* Every phase now has a wall-clock cost measured in days, and any phase can be
  interrupted by quota and resumed. `experiments/run.py` prints its projected
  request count and day count before starting.
* The quotas are per model **and** per account. They do not travel with the
  code, and `tests/test_provider.py` refuses to run a model whose RPM has not
  been read off the dashboard and recorded.

---

### [D-25] `TOP_K` is 2 because at 5 there was nothing for the gate to decide

**Context.** The first real verifier calibration failed its gate: 80% of
coverage scores landed in the single bin 0.9–1.0, against a ceiling of 60%.

The obvious reading is that the rubric does not discriminate, and the obvious
fix is to sharpen `VERIFIER_PROMPT`. **Both are wrong**, and acting on them
would have destroyed a working instrument.

**What the diagnosis actually found.** Measuring gold-passage recall against
HotpotQA's `supporting_facts` titles — reconstructible for free, because the
query embeddings were already cached:

| | |
|---|---:|
| Gold recall at iteration 1, k=5 | **100%** |
| Median rank of the *harder* gold passage | **2** |
| Questions where coverage = 1.00 *and* all gold retrieved | 24 / 24 |

**The verifier was right every time.** When both supporting passages are in the
evidence, coverage genuinely is 1.0, and JSON parsed cleanly 30/30. The
instrument was reading a task that was actually trivial.

**Why the task is trivial.** `build_corpus` indexes exactly the paragraphs
belonging to the sampled questions. For question Q, its two gold paragraphs are
the ones HotpotQA selected *because they contain the answer*, and nothing else
in the index concerns that topic except Q's own eight distractors. Dense
retrieval returns them at rank 1–2 essentially always.

Two hypotheses were tested and rejected:

* **Question type.** Bridge questions, which need a genuine second hop,
  saturate as badly as comparison questions (82% vs 75% at coverage 1.00), with
  the same median gold rank of 2. Filtering to bridge-only would not help.
* **Corpus size.** Scaling back to 2000 questions would cost ~12 days of ingest
  and probably change nothing: adding *unrelated* passages does not hide a
  question's topical gold, and the hard distractors that matter are already
  indexed.

**Decision.** `TOP_K` 5 → 2. At k=2, recall of both gold passages is 67%
overall and 59% on bridge questions, so roughly 35% of queries genuinely need a
second retrieval.

**Why this is regime selection, not result tuning.** The distinction matters,
because [D-21] rejects exactly this move when applied to λ.

* k applies **identically to all three arms**. It cannot advantage CAES.
* The gate never reads k, and no policy branches on it.
* The direction was chosen from a *measured property of the retrieval task* —
  recall was saturated — not from looking at which setting made CAES win. No
  outcome was observed at k=2 before committing to it.
* At k=5 the experiment is not merely unflattering, it is **unable to test its
  own hypothesis**: if one retrieval always suffices, no stopping rule can
  differ from any other, and CAES, Fixed(1), and a coin flip all produce
  identical evidence sets.

Contrast with tuning λ toward a better-looking iteration histogram, which
[D-21] forbids: that optimises a property *of the figure* after seeing results.
This chooses the operating point at which the question is answerable at all,
before any result exists.

**What it costs.** Absolute F1 falls for every policy, because two chunks is
thinner evidence than five. The comparison is unaffected — all arms lose the
same evidence — but absolute numbers are further from published HotpotQA
baselines than [D-24]'s corpus reduction already put them. Recorded in
METHODOLOGY §10 (External).

**Consequences.**

* Calibration must be re-run. It is ~30 LLM calls, and the cached query
  embeddings replay free, so only the verifier calls are new.
* Any λ tuned before this change is invalid — but none exists yet, which is
  precisely why calibration runs before tuning.
* If calibration still fails at k=2, the rubric becomes the next suspect, and
  *then* editing `VERIFIER_PROMPT` is the right move under invariant 9.

---

### [D-26] Gold-passage recall is recorded per iteration

**Context.** DECISIONS §8 open question 3 has been open since the first build:
a low F1 is ambiguous between *the gate stopped too early* and *retrieval never
surfaced the supporting passage*. Those call for opposite fixes — a lower λ
versus a better retriever — and nothing in the results distinguished them.

[D-25] is the proof that this matters. The single most consequential finding in
the project so far was recoverable only by reconstructing recall after the fact
from cached embeddings. Had that reconstruction been impossible, the obvious
move would have been to "sharpen the rubric" — damaging a correct instrument to
compensate for a trivial retrieval task.

**Decision.** `graph.gold_recall` computes the fraction of a question's
`supporting_facts` titles present in the evidence, and `node_retrieve` records
it **every iteration** into `gold_recall_history`. `state_summary` exports the
series and its final value; `calibrate_verifier.py` reports it alongside the
coverage distribution.

**The ground-truth firewall.** Supporting-fact titles are the answer key. A
policy that could see them would be choosing its depth from the answer, which
would invalidate every result in the paper. So:

* `gold_titles` enters through `run_query` and lives in state, but is read by
  exactly one function, which returns a number nothing else consumes;
* `evaluate_gate`, `route_from_state` and the verify node never reference it,
  pinned by `test_gold_titles_are_invisible_to_the_gate`, which greps their
  source;
* an unlabelled run records `-1.0`, so "no ground truth available" is
  distinguishable from "retrieved nothing" rather than silently reading as a
  failed retrieval.

**Consequences.** Every per-query record gains a recall series. This turns the
retrieval-versus-gate question from an argument into a column, and it is the
highest-value addition to the results section — open question 3 said so before
this was built; [D-25] demonstrated it.

---

### [D-27] Retrieval searches deep enough to return TOP_K *unseen* chunks

**Context.** Calibration phase B measured coverage trajectories for the first
time and found every one of them frozen:

```
0.40 -> 0.63 -> 0.63 -> 0.63 -> 0.63
0.20 -> 0.62 -> 0.62 -> 0.62 -> 0.62
0.40 -> 0.40 -> 0.40 -> 0.40 -> 0.40
```

Coverage moved at most once, between iterations 1 and 2, then never again.
Three measurements identified the cause and ruled out the verifier:

| Signal | Reading |
|---|---|
| Gold recall per iteration | **flat for 15/15 queries** — 0 ever improved |
| Cost per iteration | 0.000333, 0.000377, 0.000378, 0.000378, 0.000378 |
| Verifier spend | trivial after iteration 2 — the calls were cache hits |

A growing evidence set makes the verifier prompt grow, so cost must rise with
depth. It did not. Identical prompts hit the cache and returned identical
coverage. **The evidence set was not growing.**

**The defect.** `node_retrieve` searched exactly `TOP_K` deep and then filtered
out chunks already in evidence:

```python
hits  = search(query, k=config.TOP_K)
fresh = [c for c in hits if c.chunk_id not in seen]   # can be EMPTY
```

When a later iteration's query ranks the same chunks first — which is the
normal case, since the planner rewrites a query about the same question over
the same index — every hit is filtered and `fresh` is empty. The iteration
still pays for a planner call, a query embedding and a verifier call, and adds
nothing.

**Decision.** Search `TOP_K + len(seen)` deep and take the first `TOP_K` unseen
hits. Every iteration then adds `TOP_K` new chunks until the corpus is
exhausted.

**Why this is a correctness defect and not a tuning knob.** A dead iteration
makes ΔQ **structurally zero**, not merely small:

* coverage cannot move, because the verifier sees byte-identical evidence;
* so ΔQ ≈ 0 from the first dead iteration onward, for every query;
* so the gate's margin has a constant sign and CAES degenerates to a
  fixed-depth policy — the exact failure **[D-21]** describes, arriving through
  retrieval instead of through λ;
* and **the baselines are corrupted too**: `FixedPolicy(n=3)` pays for three
  iterations while holding iteration 2's evidence, so the headline "CAES costs
  less at equal F1" would have been trivially true and completely uninformative.

It would not have been visible in the results. F1 and cost would both be real
numbers, the figures would render, and the only symptom is a cost reduction
that measures nothing. This is the class of failure **[D-21]** and §6 warn
about, and it was caught only because the trajectory measurement added in the
same session made the evidence set's behaviour observable.

**Why TOP_K=2 exposed it.** The defect predates [D-25] but was masked at
`TOP_K=5`: a deeper slice is more likely to contain something unseen, so
evidence limped along instead of stopping dead. Lowering k to 2 turned an
intermittent stall into a total one. The bug was always wrong; it was only ever
partly hidden.

**Instrumentation added.** `new_chunks_history` records chunks added per
iteration and `dead_iterations` counts the zeros, both in every per-query
record. A dead iteration is now a number in the results rather than an
inference from flat cost. Guarded by
`tests/test_graph.py::test_each_iteration_adds_new_evidence`, which pins that
search depth grows with what has been seen.

**Rejected.** Varying the planner prompt to force query diversity (treats the
symptom — the retriever should return new evidence regardless of how similar
two queries are); dropping the seen-filter and allowing duplicate evidence
(inflates the evidence set without adding information, and doubles the
verifier's prompt cost for nothing).

**Consequences.** Every iteration now genuinely deepens the evidence set, so
per-iteration cost grows with depth as METHODOLOGY §3.2 assumes. Calibration
must be re-run; the cached calls replay free, and only the newly reachable
chunks cost quota. Any measurement taken before this fix — including both
earlier calibration runs — describes a system whose iterations after the first
did nothing.

---

## 6. Traps

Things that will cost you time, in rough order of likelihood.

| Trap | Symptom | Fix |
|---|---|---|
| `GEMINI_API_KEY` not exported | `RuntimeError` naming the variable, before any network call | `export GEMINI_API_KEY=...` (never commit it) |
| Index built by the other provider | `ValueError` naming both dimensions on `Retriever` construction | `rm -rf data/` and re-run `python ingest.py` — see **[D-22]** |
| Free-tier rate limit (per minute) | HTTP 429, retried with the server's `retryDelay` | Lower `GEMINI_LLM_RPM` / `GEMINI_EMBED_RPM` |
| Free-tier quota spent (per day) | `QuotaExhausted`, not retried; the run exits with resume instructions | Re-run tomorrow. Cached calls replay free, so no quota is re-spent. See **[D-24]** |
| Every call 400s right after a model change | wrong thinking field for the model family | Add the model to `GEMINI_THINKING_CONFIG`; 2.5 takes `thinkingBudget`, 3.x takes `thinkingLevel` |
| Bedrock model access blocked | `ValidationException: Operation not allowed` | Account-wide block; use `CAES_PROVIDER=gemini` — see **[D-22]** |
| Synthetic data still in `data/` | `ingest.py` exits 2 mentioning devdata | `rm -rf data/` |
| `CAESPolicy` raises `ValueError` | λ never tuned | Run `tune_lambda.py`, write the value into `config.py` |
| `experiments/run.py` exits 2 immediately | output file exists and `--resume` not passed | Add `--resume`, or delete `results/{policy}_raw.jsonl` |
| Nothing happens, exits 0 | pre-flight confirmation | Add `--yes` |
| *Notional* cost is `$0.00` | `DRY_RUN=1` is still exported | `unset DRY_RUN` |
| *Actual* cost is `$0.00` | not a trap on Gemini — the free tier bills nothing, always. The gate reads notional; see **[D-22]** | nothing to fix |
| Every CAES query stops at the same iteration | λ is far outside the sensitive region | Check `caes_decisions.jsonl`: if `lambda_times_delta_c` dwarfs `delta_q` everywhere, λ is too high |
| λ sweep says "DEGENERATE" | F1 flat — usually still in `DRY_RUN` | Run with real responses |
| λ sweep warns "degenerate on spread" | recommended λ puts >90% of queries in one iteration bucket | Compare against the best-spread λ it names; see **[D-21]** |
| Coverage all lands in one band | rubric not discriminating **or the retrieval task is trivial** | Check `gold_recall` FIRST. At 100% recall the verifier is right and the corpus is too easy — sharpening the rubric would damage a working instrument. See **[D-25]** |
| Coverage frozen after iteration 2 | iterations are adding no evidence | Check `new_chunks_history` for zeros and whether per-iteration cost stops growing. See **[D-27]** |
| Ledger seems stuck | it is cumulative and persistent, by design | `python -c "from costs import TRACKER; print(TRACKER.summary())"` |

### The one that is hardest to notice

If you add a model call that bypasses `llm.py`, everything keeps working. Tests
pass. Figures render. The only symptom is that ΔC is understated, so the gate
retrieves more than it should, and the paper's cost numbers are quietly wrong.

No *runtime* test can catch this, which is why invariant 1 exists. There is now
a static one — `test_llm_is_the_only_module_that_calls_a_provider` greps every
module outside `llm.py` and `config.py` for a provider endpoint — but it
recognises the two endpoints we know about, so it narrows the hole rather than
closing it.

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
3. ~~**Retrieval failure and gate failure are currently indistinguishable.**~~
   **RESOLVED 2026-08-31 — see [D-26].** Gold-passage recall against
   `supporting_facts` is now recorded every iteration in `gold_recall_history`.
   It was worth more than anticipated: reconstructing it after the fact is what
   revealed that [D-25]'s calibration failure was a trivial retrieval task
   rather than a blunt rubric.
4. **One-shot's complexity score is hand-built.** The spec permitted a small
   Haiku call instead. The current heuristic is free and deterministic, but if
   B2 routes nearly every question to the same depth on real data, the baseline
   is weak and a reviewer will say so — check the depth distribution.
5. **`FixedPolicy(n=3)` is the headline baseline.** Whether N=3 is the fair
   comparison, or whether the N that matches CAES's mean iteration count is
   fairer, is a framing decision worth making explicitly before writing up.

6. **UNRESOLVED — the ledger does not record what [D-22] claims it records.**
   Deliberately left open on 2026-08-31; decide before writing up.

   **[D-22] says "actual spend is now always $0.00". The code does not do
   that.** `CostTracker.record_llm` / `record_embed` compute cost from the price
   table with no knowledge of the free tier, so the ledger accrues list price
   for every network call. The first real preflight
   (`python -m llm --check`) returned
   `notional / actual usd: $0.00000430 / $0.00000430` — identical — and moved
   the ledger to `$0.0000049` on calls Google billed nothing for.

   The two counters are still meaningfully different, just not in the way
   [D-22] describes:

   | | Records | On a paid tier | On the free tier |
   |---|---|---|---|
   | ledger ("actual") | list price of calls that **hit the network** | money billed | money that *would* have been billed |
   | notional | list price of **all** calls, cache hits included | — | — |

   That distinction is the one [D-12] actually depends on (cache hits accrue
   notional, never touch the ledger), so **[D-12] is unaffected and the gate is
   correct**. What is wrong is [D-22]'s description, and its conclusion that
   `HARD_BUDGET_USD` is "vestigial" — on this reading the ceiling is a live cap
   on list-price exposure, which is a useful thing to keep.

   Two ways to close it, not yet chosen:

   * **Correct the documentation.** Leave the code alone; restate the ledger as
     list-price exposure in [D-22], METHODOLOGY §3.3 and §10. Cheapest, and the
     budget guard stays live.
   * **Make actual genuinely $0.00.** Add a free-tier flag so the ledger records
     zero. Makes [D-22] literally true, but budget guards read actual, so the
     guard goes permanently inert — and silently so if billing is ever attached
     to the Google account.

   **Whichever is chosen, the paper's wording is already safe**: §10 (Construct)
   says costs are list-price-derived and not billed, which is true under both
   readings. The risk is confined to anyone reading the ledger as an invoice.
