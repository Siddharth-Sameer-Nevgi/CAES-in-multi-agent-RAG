# METHODOLOGY.md — research design

How the claim is constructed, measured, and defended. This file covers *what is
being tested and why the test is valid*. For *how the code works*, see
[IMPLEMENTATION.md](IMPLEMENTATION.md); for *why individual engineering choices
were made*, see [DECISIONS.md](DECISIONS.md).

---

## 1. Problem statement

Iterative RAG systems retrieve, verify, and retrieve again until some stopping
condition is met. Almost all of them stop on a rule that ignores money:

* **fixed depth** — always N iterations, whatever the evidence looks like;
* **one-shot routing** — pick a depth up front from a complexity heuristic, then
  commit to it regardless of what the retrieved evidence turns out to contain;
* **quality thresholds** — stop when a verifier's confidence passes a bar, with
  no cost term at all.

Every one of these treats the marginal retrieval iteration as free. It is not.
On a hosted LLM, each additional iteration costs a planner call, a verifier
call, a query embedding, and a growing prompt that carries the accumulated
evidence — while evidence quality shows **diminishing returns**: the first
iteration usually resolves most of what a question needs, and the fifth rarely
resolves anything.

The gap: no widely used gate compares the *expected* quality gain of the next
iteration against its *measured* cost.

## 2. Hypothesis

> A stopping gate that weighs estimated marginal evidence gain against measured
> marginal cost achieves substantially lower cost per query than a fixed-depth
> policy, at answer quality that is statistically indistinguishable from it.

Two halves, and both must hold. A cost reduction bought with an F1 drop is not
the claim; neither is F1 parity at the same cost.

## 3. The decision rule

At the end of every iteration, continue retrieving only while

```
ΔQ − λ·ΔC > 0
```

| Term | Meaning | Source |
|---|---|---|
| `ΔQ` | estimated evidence-quality gain of the **next** iteration | extrapolated from the verifier's coverage trajectory |
| `ΔC` | expected cost of the **next** iteration, in USD | mean of the per-iteration costs actually measured for this query |
| `λ` | exchange rate between quality points and dollars | tuned once, on held-out data |

λ is the only free parameter. It carries units of quality-per-dollar: a large λ
prices a dollar expensively and stops early; a small λ prices quality
expensively and retrieves deeper. Because the two terms live on incomparable
scales (coverage is in [0,1]; cost is in fractions of a cent), the usable λ band
is numerically large — order 10¹–10³ — which is why the tuning grid is
log-spaced rather than linear.

### 3.1 Estimating ΔQ

The verifier returns a `coverage` score in [0,1] each iteration, so a query
produces a coverage trajectory. ΔQ extrapolates the next step from the last
observed step, damped by a decay factor (`DECAY_FACTOR = 0.6`) that encodes the
diminishing-returns premise:

```
ΔQ = max(0, (coverage[t] − coverage[t−1]) · DECAY_FACTOR)
```

With fewer than two observations there is no trajectory to differentiate, so ΔQ
is set to 1.0 — "unknown, allow one more iteration". Combined with
`MIN_ITERATIONS = 1`, no query is ever answered on zero retrievals.

**Smoothing.** The series is differentiated as a *running max*, not raw.
Coverage genuinely dips: a newly retrieved document introduces a second
plausible entity and the verifier correctly becomes less certain. Differencing
the raw series would read that transient dip as negative gain and force a
premature stop on a query that was in fact making progress. Both the raw and
smoothed series are recorded for every decision, so the smoothing is auditable
and its effect measurable.

### 3.2 Estimating ΔC

ΔC is the mean of the per-iteration costs already observed **for this query** —
not a constant, and not a forecast from a model. Per-iteration cost is metered
by diffing process-wide token-cost counters across the span of one iteration, so
it reflects the real prompt sizes that iteration used, including the growth in
evidence carried into the verifier.

Cost is accrued at list price from *measured token counts read out of the API
response*, never from a character-count estimate. Estimates are used only for
the pre-flight budget check, never for a number that reaches the results. The
one endpoint that returns no token count — Gemini's embedding endpoint — is
measured with a separate token-counting call rather than estimated.

### 3.3 Cache neutrality

Re-running an experiment replays LLM responses from a disk cache, which means
the run spends no money. If ΔC were taken from money actually spent, a warm
cache would show a free next iteration, the gate would run to the iteration cap,
and a cached re-run would produce *different decisions from the paid run that
produced the cache*.

The system therefore separates two ideas of cost: the **ledger** (money actually
spent, for budget enforcement) and **notional cost** (list price of the measured
tokens, cached or not). The gate reads notional cost. Consequence: results are
identical whether the run paid or replayed — which is what makes the experiment
reproducible from the artifact.

On the current provider's free tier this separation is not merely convenient but
necessary: nothing is ever billed, so actual spend is zero on every call of every
run. Were the gate reading actual cost, λ·ΔC would be identically zero and CAES
would retrieve to the iteration cap on every query. Notional cost is the only
cost signal that exists. See [DECISIONS.md](DECISIONS.md) **[D-22]**.

## 4. Experimental design

A three-way comparison over a common question set, with each policy differing
**only** in the stopping rule. Same corpus, same index, same retriever, same
top-k, same planner, same verifier, same generator, same prompts, same
temperature (0.0), same iteration cap.

| Arm | Policy | Stopping rule |
|---|---|---|
| **B1** | `fixed` | exactly N iterations (N = 3 by default) |
| **B2** | `oneshot` | depth chosen before iteration 1 from a complexity score, then committed |
| **Treatment** | `caes` | `ΔQ − λ·ΔC > 0`, re-evaluated every iteration |
| *(fallback)* | `threshold` | stop when `coverage_delta < 0.05 AND coverage > 0.7` |

**Baseline fidelity is what makes the comparison credible**, so the baselines
are implemented seriously rather than as strawmen:

* B2's complexity score is deterministic and derived from the question text
  alone — length, capitalised-token count as an entity proxy, multi-hop marker
  phrases, clause count — mapped to a depth of 1–4. It deliberately never
  consults the verifier or any retrieved evidence, because the defining property
  of one-shot routing is that depth is chosen *without feedback*. The committed
  depth is cached per query id and never revisited.
* B1 is the honest fixed-depth system: it ignores the verifier entirely, even
  when coverage is already 1.0.

The `threshold` arm exists as insurance. If the ΔQ estimator proved unusable, it
would still yield a working system with real cost data — but it is a weaker
contribution (closer to prior quality-only gating, with no cost term), so it is
kept behind a flag and is not part of the headline comparison.

### 4.1 Controlled confounds

* **The confidence short-circuit is disabled for all experimental arms.** The
  pipeline supports stopping early when the verifier reports both high coverage
  and confidence. Enabling it would give CAES a second, orthogonal stopping rule
  the baselines do not have, and any cost reduction could then be attributed to
  either mechanism. It is off for the three-way comparison and on only in the
  serving API, which optimises latency and makes no cross-policy claim.
* **The iteration cap is enforced by the orchestration graph, before the policy
  is consulted at all.** No arm can exceed `MAX_ITERATIONS = 5`, so a gate bug
  cannot manufacture a cost difference. This is tested with a deliberately
  broken gate that always answers "retrieve".
* **Determinism.** Temperature 0.0 on every LLM call; a fixed split seed; a
  content-addressed cache keyed on the exact request payload. The same question
  under the same policy produces the same trajectory.
* **Identical evidence accumulation.** Every arm accumulates evidence across
  iterations and de-duplicates by chunk id in the same way; arms differ in *how
  many* iterations they run, never in what an iteration does.

## 5. Data

**HotpotQA (distractor, validation split).** Chosen because its questions are
explicitly multi-hop: a single retrieval usually cannot answer them, so
retrieval depth genuinely matters and the diminishing-returns curve has room to
show itself. A single-hop dataset would flatten every policy to one iteration
and the experiment would measure nothing.

* 2,000 questions sampled with a fixed seed; their gold paragraphs form the
  corpus, deduplicated by title.
* Passages chunked at ~200 tokens with ~30 tokens of overlap, with the title
  prepended to each chunk — titles carry real retrieval signal in HotpotQA.
* Embedded with Titan Text Embeddings V2 at 1024 dimensions, L2-normalised, in a
  flat inner-product FAISS index, so inner product is cosine similarity and
  retrieval is exact rather than approximate. At this corpus size there is no
  reason to accept ANN recall error.

### 5.1 Splits

Two disjoint splits are derived deterministically (seed `20240917`) from the
question set — sorted by id, then shuffled — rather than stored as files:

| Split | Size | Used for |
|---|---|---|
| **tune** | 50 | verifier calibration, λ sweep |
| **test** | 150 | the reported three-way comparison |

Disjointness is asserted every time the splits are loaded, and the tuning script
never imports the test split. **Tuning λ on the evaluation data would invalidate
the headline number**, so this is enforced structurally rather than by
discipline. Deriving the splits rather than storing them means they cannot drift
out of sync with the question file.

## 6. The verifier is the instrument

The entire method rests on coverage being an informative signal. If coverage is
noise, ΔQ is noise and the gate is a random number generator with a budget.

The verifier is therefore treated as a measuring instrument that must be
**calibrated before use**, and calibration is a hard blocker in a separate
acceptance script run over the tuning split. Two criteria:

1. **Parse validity** — every sampled response must yield valid JSON.
2. **Spread** — coverage must actually use the range:

   | Criterion | Threshold |
   |---|---|
   | standard deviation | ≥ 0.12 |
   | max − min | ≥ 0.40 |
   | share in any single 0.1-wide bin | ≤ 0.60 |

Failing any of these blocks progress to the experiment phase and directs the
researcher to sharpen the verifier rubric. The thresholds are written down
explicitly so the gate is arguable rather than a judgement call made after
seeing the data.

The rubric itself is engineered for spread: four labelled bands with concrete
descriptions, an instruction to judge only what the evidence states (not what
the model knows), a rule capping multi-hop questions at 0.6 when a hop is
unsupported, and two worked examples at opposite ends of the range.

**Parse failures are handled conservatively.** After one repair attempt, an
unparseable response holds coverage flat at the previous value rather than
guessing. Flat coverage means zero measured gain, so the gate stops. A parse
failure can therefore end a query early but can never be mistaken for progress,
and the failure count is recorded per query so its rate is reportable.

## 7. Tuning λ

Run on the 50-question tuning split only, in two passes:

1. **Coarse:** λ ∈ {1, 3, 10, 30, 100, 300, 1000} — roughly half-decade steps.
   The gate's sensitive band, where `λ·ΔC` is comparable to ΔQ and iteration
   counts actually vary, is narrow; a decade-spaced grid steps clean over it and
   returns a flat sweep.
2. **Refine:** log-spaced multipliers {0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0} about
   the knee. The coarse grid is log-spaced, so the knee is located to within a
   *multiplicative* factor; refining linearly would oversample above the centre
   and undersample below it.

The knee is the λ with the best F1-per-dollar. That ratio is a **search
heuristic for where to spend the refinement budget**, not a claim that
F1-per-dollar is the objective.

The sweep reports two failure modes rather than silently returning a number:

* **Flat F1** (range < 0.01 across the whole sweep) — there is no tradeoff to
  tune, and F1-per-dollar degenerates to "pick the cheapest λ", which would
  happily recommend one that stops at a single iteration every time. The script
  refuses to recommend, and names the likely causes: running under dry-run mode,
  an undiscriminating verifier, or too small a tuning split.
* **Degenerate spread** (≥ 90% of queries stopping at the same iteration) — a
  warning, not a blocker. Such a λ may be genuinely optimal on F1-per-dollar,
  but its iteration histogram is a single bar, which makes CAES visually
  indistinguishable from a fixed policy on the figure that is supposed to
  demonstrate per-iteration granularity. The best-spread alternative in the
  sweep is printed alongside it; choosing between them is a judgement about what
  the experiment should demonstrate, so it is left to the researcher.

The chosen value is written into configuration by hand and never re-tuned. The
CAES policy refuses to construct while λ is unset — guessing it would make the
headline number meaningless, so the failure is loud rather than silent.

## 8. Metrics

Recorded per query, for every arm:

| Metric | Definition |
|---|---|
| `total_usd` | notional cost of the whole query (all iterations + generation) |
| `total_latency_ms` | summed model latency across the query |
| `iterations_used` | retrieval iterations actually run |
| `exact_match` | SQuAD-style normalised exact match |
| `f1` | SQuAD-style token F1 |
| `final_coverage` | verifier coverage at the stopping iteration |
| `abstained` | 1 if the generator declined for want of evidence |
| `stop_reason` | which rule ended the query (`caes`, `fixed`, `max_iter`, …) |
| `parse_failures` | verifier parse failures in this query |

Two scoring conventions matter:

* **Abstentions score zero on both EM and F1**, rather than being excluded. A
  policy that stops too early and honestly says "insufficient evidence" must be
  penalised for it — otherwise the cost saving would look free, which is exactly
  the failure mode the experiment exists to detect.
* **Yes/no questions require an exact match** for F1. Token overlap on a
  one-token gold answer is meaningless and would inflate F1 for any answer
  containing the word.

## 9. Statistical analysis

The headline compares CAES against Fixed **paired by query id** — the same
question, both policies — which removes per-question difficulty as a variance
source.

* **Quality:** paired t-test on per-query F1 differences, plus a 10,000-sample
  bootstrap 95% CI on the mean difference. The parity claim is made only when
  that CI contains zero. If it excludes zero, the analysis says so explicitly
  and instructs that the cost reduction be reported *alongside* the quality
  change rather than as parity.
* **Cost:** mean cost reduction versus Fixed, with a bootstrap 95% CI on the
  per-query relative difference.

The claim is therefore falsifiable in a specific way: it fails if the F1 CI
excludes zero, or if the cost-reduction CI includes zero.

Supporting evidence beyond the headline:

| Output | What it establishes |
|---|---|
| iteration histogram | CAES spreads across depths 1–5 where Fixed is a single bar — the granularity claim |
| coverage-vs-iteration curve | diminishing returns are real in this data, i.e. the premise holds |
| λ sweep curve | the cost/quality tradeoff is continuous and λ controls it |
| stop-reason breakdown | the gate is actually firing, rather than the iteration cap doing the work |
| per-decision log | ΔQ, ΔC, λ·ΔC, margin, and outcome for every iteration of every query |

The per-decision log is the mechanism-level evidence: it shows *why* each stop
happened, not merely that costs differed.

## 10. Threats to validity

**Internal.**

* The ΔQ estimator is a one-step linear extrapolation with a fixed decay factor;
  that factor is set a priori and not tuned — conservative, but arbitrary.
* Running-max smoothing biases ΔQ upward on noisy trajectories, making the gate
  slightly more willing to continue. Both series are logged so the effect can be
  quantified.
* The verifier judges evidence for the same question it will later be used to
  answer, using the same model as the generator. Correlated errors between
  verifier and generator would understate the quality cost of stopping early.

**External.**

* Single dataset (HotpotQA), single retrieval architecture (dense, flat index,
  fixed top-k), single model (`gemini-2.5-flash`, with `gemini-embedding-001`
  for retrieval), single price point. The direction of the result should
  generalise; the magnitude is specific to this cost structure.
* **The provider was substituted mid-project, and not by choice.** The system
  was built against AWS Bedrock (Claude Haiku 4.5 + Titan Text Embeddings V2);
  model invocation is blocked account-wide on the available AWS account —
  `InvokeModel` returns `ValidationException: Operation not allowed` for every
  model and also through the console, so it is neither model- nor SDK-specific.
  The Bedrock code path is retained and tested, so a two-provider replication is
  available if that block clears; it has not been run. Since λ is an exchange
  rate between coverage points and dollars, a different price table moves the
  band in which the gate is sensitive: Gemini input tokens are ~3× cheaper and
  output tokens ~2× cheaper than the Bedrock configuration, so λ tuned on one
  provider does not transfer to the other.
* The result depends on the price ratio between the gate's own overhead and a
  retrieval iteration. On a cheaper verifier or a more expensive retriever, the
  optimal λ changes and so does the size of the win.

**Construct.**

* Coverage is a proxy for evidence quality, not evidence quality itself. The
  calibration gate establishes that it is *informative*, not that it is correct.
* Cost is list-price notional cost, excluding infrastructure, storage, and
  engineering time. It is the cost the gate can actually act on. **No spend was
  billed for any reported number**: the evaluation ran entirely on the
  provider's free tier, so every cost figure in this work is derived from
  published per-token list prices applied to measured token counts, and none of
  it was charged. Costs must be reported as list-price-derived, not as billed.

## 11. Reproducibility

* Every parameter — prices, model ids, budgets, chunking, top-k, λ, seeds — lives
  in one configuration module; nothing else hardcodes any of them.
* Splits are derived from a fixed seed and asserted disjoint on every load.
* All LLM calls are temperature 0.0 and content-addressed in a disk cache, so a
  re-run replays the exact responses that produced the reported numbers.
* Because the gate reads notional rather than actual cost, a fully cached re-run
  reproduces the results exactly while spending nothing.
* Per-query results are checkpointed to disk as they complete, so a partial run
  is a valid partial artifact rather than a loss.
* A dry-run mode exercises the entire pipeline — including the gate, against a
  synthesised diminishing-returns coverage curve — with no network access and no
  spend, so the wiring can be verified independently of the measurement.

**Resource constraints are part of the method, not an afterthought.** The
project was built to run inside a fixed credit budget, with a hard ceiling
enforced pre-flight from an estimate, before any billable call is made. That
constraint is why model selection is fixed at the cheapest capable tier, why the
verifier's evidence is truncated, and why the cache exists — all three are
design decisions the experiment then has to work within, each documented in
[DECISIONS.md](DECISIONS.md).

The binding constraint has since changed shape rather than disappeared. The
evaluation runs on the provider's free tier, so **no call is billed and the
monetary ceiling never binds**; what binds instead is **rate limiting**. Free
tier limits are per-account and no longer published per model, so the system
paces itself client-side and retries `429 RESOURCE_EXHAUSTED` honouring the
server-supplied delay, rather than assuming a documented ceiling.

Two consequences for reproducibility:

* **The three design decisions above are unchanged, but their justification
  moved.** Truncating the verifier's evidence and caching aggressively now buy
  requests-per-day rather than dollars. Both remain necessary; the cheapest
  capable tier is still the right model choice, for latency and throughput
  reasons rather than price.
* **Replication cost is not zero for someone else.** These runs cost nothing on
  a free-tier account, but the reported numbers are list-price notional (§10,
  Construct). A reader replicating on a paid account should expect to be billed
  approximately the costs this work reports.

The spend ledger and the `HARD_BUDGET_USD` ceiling are retained despite being
inert on the free tier, because the Bedrock path is retained too — see
[DECISIONS.md](DECISIONS.md) **[D-22]**.
