# Changelog

Progress log for CAES-RAG. Entries are grouped by the implementation phase they
belong to, in the order the phases were built. Each entry records what landed,
and — where it matters — what was found broken and fixed along the way.

This file is the history. Keep appending here alongside commits; commit
messages are not a substitute, because most of the interesting content is
*why*, not *what*.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Rationale for anything marked **[D-n]** lives in [DECISIONS.md](DECISIONS.md).

---

## [0.1.0] — 2026-08-16

First complete build. All six phases implemented and verified end to end under
`DRY_RUN=1`. **No paid Bedrock call has been made yet** — the system has never
been run against real HotpotQA data or real model responses.

### Status at this tag

| Phase | State | Verified how |
|---|---|---|
| 0 — Cost safety | Complete | 16 unit tests + `$0.00` smoke run |
| 1 — Corpus & index | Complete, unrun | Exercised via synthetic corpus |
| 2 — Agents | Complete, uncalibrated | 13 parse/metric tests; calibration is a paid step |
| 3 — Graph & baselines | Complete | 17 tests incl. runaway-gate containment |
| 4 — CAES gate | Complete, λ unset | 16 tests; λ sweep runs but needs real F1 |
| 5 — Experiments | Complete | Full dry-run: 3 policies × 40 questions + 4 figures |
| 6 — Packaging | Complete | API smoke-tested via `TestClient` |

**Blocking next step:** verify pricing constants in `config.py`, then
`python ingest.py`. `LAMBDA` is deliberately `None` and `CAESPolicy` refuses to
construct until it is set by tuning.

---

### Phase 0 — Cost safety infrastructure

Built first; nothing else may call Bedrock until this exists.

**Added**
- `config.py` — single home for prices, budgets, model ids, loop bounds, paths,
  and split parameters. No price or model id is hardcoded anywhere else.
- `costs.py` — `CostTracker` with a persistent JSON ledger, itemised
  `CostRecord`s (timestamp, model, call type, in/out tokens, latency, USD,
  query id, iteration, policy), `to_dataframe()` / `to_csv()` export, and a
  `run_budget(max_usd)` context manager.
- `cache.py` — sha256-keyed disk cache, sharded by key prefix, with hit/miss/
  write counters and `log_stats()`.
- `bedrock.py` — the sole Bedrock entry point. Cache lookup → pre-flight budget
  check → call → read real token counts → record → cache → return.
- `smoke.py` — `DRY_RUN=1 python -m smoke`, full graph for `$0.00`, asserts the
  ledger did not move.
- `tests/test_costs.py` (9), `tests/test_cache.py` (7).

**Design points**
- Budget checks are **pre-flight**, from an estimate, and raise before the
  network call. A test asserts the mocked client's call count stays at zero
  when the ceiling is blown. **[D-1]**
- The ledger is written with a temp-file-then-atomic-replace so a crash
  mid-write cannot corrupt it. A test writes, drops the object, reloads, and
  asserts the total survives.
- A response with no `usage` block is a **fatal error**, not a fallback to
  estimation — measured ΔC is a core claim. **[D-2]**
- Cache hits never touch `CostTracker`; a replayed call records `usd=0.0`.
- An unreadable ledger raises rather than silently starting from zero.

---

### Phase 1 — Corpus and index

**Added**
- `ingest.py` — HotpotQA distractor loading, title-deduplicated passage
  extraction, word-based chunking (~200 tokens, 30 overlap), batched embedding
  with a progress bar, `faiss.IndexFlatIP` over L2-normalised vectors, and
  persistence to `data/`. Optional `--upload-s3`.
- `retrieval.py` — `Retriever.search()` and a process-wide singleton.
- `splits.py` — deterministic, seed-stable tune/test splits with a disjointness
  assertion.

**Design points**
- Titan V2 has no batch endpoint, so `bedrock.embed()` loops internally;
  `EMBED_BATCH` paces progress reporting and rate limits, not a batch API call.
- Titles are prepended to chunk text — they carry real signal in HotpotQA.
- Re-running `ingest.py` refuses to re-embed unless `--force`.
- `Retriever` fails loudly on an index/chunk count mismatch rather than serving
  silently misaligned results.

---

### Phase 2 — Agents

**Added**
- `agents/prompts.py` — every prompt as a named constant.
- `agents/planner.py` — iteration 1 returns the question verbatim (no LLM call);
  later iterations request a focused sub-query, capped at 100 output tokens.
- `agents/verifier.py` — the ΔQ signal. Strict-JSON rubric with four explicit
  coverage bands and one worked low-coverage plus one worked high-coverage
  example. Evidence truncated to ~150 tokens per chunk.
- `agents/generator.py` — grounded answer, abstains with
  `"insufficient evidence"` rather than speculating.
- `metrics.py` — SQuAD-style EM and token F1, with yes/no handling.
- `calibrate_verifier.py` — the Phase 2 acceptance gate, runnable.
- `tests/test_verifier.py` (13).

**Design points**
- Verifier parse ladder: strip fences → `json.loads` → regex-extract the
  outermost object → one repair retry → **hold the previous coverage**. Holding
  flat means a parse failure reads as zero gain and stops the loop, rather than
  being mistaken for progress. **[D-3]**
- `calibrate_verifier.py` fails the build on imperfect JSON parsing *or* on
  clustered coverage, with explicit thresholds (stdev ≥ 0.12, range ≥ 0.40, no
  single 0.1-wide bin over 60%) so the gate is arguable rather than a judgement
  call.
- Abstentions score zero rather than being excluded from the mean. **[D-4]**

---

### Phase 3 — Graph and baselines

**Added**
- `graph.py` — LangGraph `StateGraph`: `plan → retrieve → verify → [gate] →
  generate`, plus an equivalent manual executor used only if LangGraph is
  missing.
- `policies.py` — `FixedPolicy` (B1), `OneShotPolicy` (B2) with a deterministic
  complexity score, a `Policy` protocol, and a `build_policy()` factory.
- `tests/test_graph.py` (7), `tests/test_policies.py` (10).

**Design points**
- `MAX_ITERATIONS` is checked **inside the graph, before the policy is consulted
  at all**. `test_max_iterations_is_enforced_against_a_broken_gate` proves it
  with a policy that always says "retrieve". **[D-5]**
- `OneShotPolicy` commits its depth before iteration 1 from information
  available with no retrieval and no verifier feedback, then never revisits it —
  a test feeds it a contradictory coverage signal mid-run and asserts it does
  not change its mind. Baseline fidelity is what makes the comparison credible.
- Evidence is deduplicated by `chunk_id` across iterations.

**Fixed during the build**
- **`stop_reason` was always `max_iter`.** The router mutated the state dict
  handed to the conditional edge, and LangGraph does not merge writes made
  inside an edge back into graph state, so every reason was lost and a fallback
  filled in `max_iter`. The gate now runs at the end of `verify` and writes
  `_route` / `stop_reason` into state; the edge is a pure read. **[D-6]**

---

### Phase 4 — The CAES gate

**Added**
- `caes.py` — `estimate_delta_q()` (decayed extrapolation of the last coverage
  delta), `estimate_delta_c()` (mean observed per-iteration cost), running-max
  coverage smoothing, `CAESPolicy`, `ThresholdPolicy` (the insurance fallback),
  and a JSONL `DecisionLogger`.
- `tune_lambda.py` — coarse sweep over `[0.1, 1, 10, 100, 1000]`, refinement
  around the knee, `results/lambda_sweep.csv`.
- `tests/test_caes.py` (16).

**Design points**
- Every gate decision is logged with `coverage_raw`, `coverage_smoothed`,
  `delta_q`, `delta_c`, `lambda_value`, `lambda_times_delta_c`, `margin`,
  `outcome`, and `reason` — this file is the source of the paper's central
  figure.
- Coverage is smoothed with a running max before differencing, so a transient
  dip (new documents introducing ambiguity) does not force a premature stop.
  Both series are logged. **[D-7]**
- `CAESPolicy` raises on construction while `config.LAMBDA is None`. Guessing λ
  would make the headline number meaningless.

**Fixed during the build**
- **`tune_lambda.py` would silently recommend a degenerate λ.** When F1 is flat
  across the whole sweep, the F1-per-dollar knee heuristic reduces to "pick the
  cheapest", which happily recommends a λ that stops at one iteration every
  time. Added `f1_is_flat()`; the script now exits non-zero, refuses to
  recommend, and names the likely causes. **[D-8]**

---

### Phase 5 — Experiments

**Added**
- `experiments/run.py` — pre-flight cost projection requiring `--yes`,
  checkpoint-after-every-query to `results/{policy}_raw.jsonl`, `--resume`,
  run-budget enforcement, and graceful `BudgetExceeded` / `KeyboardInterrupt`
  handling that preserves partial results.
- `experiments/analyze.py` — main table, four figures, stop-reason breakdown,
  and the headline number with a paired t-test plus bootstrap CIs.
- `devdata.py` — synthetic corpus generator so the whole pipeline can be
  exercised for `$0.00`.

**Design points**
- Checkpoints are flushed per query; a torn final line from a crash is skipped
  on reload rather than aborting.
- The headline number is computed on **paired** query ids across policies, with
  a bootstrap CI on the F1 difference and on the relative cost change. If the CI
  on ΔF1 excludes zero, the script explicitly refuses to claim parity.
- Figures follow a CVD-validated three-colour categorical palette assigned to
  **policies, never to rank**, so re-ordering never repaints a series. Direct
  labels are mandatory — one slot sits below 3:1 contrast on the light surface,
  and labels are the required relief. **[D-9]**

**Fixed during the build**
- Direct labels overflowed the right axis edge in fig1 → labels now flip inward
  past the midpoint, with added margins.
- Direct labels stacked on top of each other in fig3 when policies stop at the
  same depth → staggered vertically by draw order.
- Legend crowded the 100% bar in fig2 → explicit y headroom and a horizontal
  legend.

---

### Phase 6 — Packaging

**Added**
- `api.py` — FastAPI `POST /query` and `GET /health`. No auth, no rate limiting,
  no deployment tooling, by design.
- `README.md` — setup, model-access prerequisites, the
  ingest → calibrate → tune → run → analyze workflow, the cost table, and the
  forbidden-services warning.
- `.gitignore` — excludes the ledger, cache, `data/`, and `results/`.

**Design points**
- Query handling is serialised behind a lock: per-iteration cost is metered by
  diffing process-wide counters, so concurrent runs would mis-attribute cost to
  each other. Documented as a demonstration-endpoint limitation. **[D-10]**
- The API enables `honor_confidence=True`; the experiments do not. **[D-11]**

---

### Cross-cutting work

**Added**
- **Notional cost accounting** in `bedrock.py` (`totals()`, `_accrue()`,
  `notional_llm_usd()`), plus `notional_usd` on `LLMResponse` and persisted
  token counts on cached embeddings.

  Not in the original spec, and the spec does not work without it: ΔC is
  measured spend, but a cache hit spends nothing. On a warm cache ΔC would
  collapse to zero, the gate would see an infinitely cheap next iteration and
  run to `MAX_ITERATIONS`, and a cached re-run would make **different decisions**
  than the paid run. Notional cost is derived from the real measured token
  counts the cache preserves, so gate behaviour is identical cached or not.
  The ledger still tracks actual money. **[D-12]**

- **Synthetic-data guard.** `devdata.py` stamps `data/meta.json` with
  `"synthetic": true`, and `ingest.py` detects it and exits non-zero with a
  specific message rather than the generic "index already exists". Without this,
  a developer who dry-ran the pipeline would find the real ingest silently
  refusing to build.

**Verified**
- 62 tests, no network access and no AWS credentials required.
- `DRY_RUN=1 python -m smoke` on a clean checkout: ledger delta `$0.000000`.
- Full dry pipeline: `devdata` → 3 policies × 40 questions → λ sweep → analysis
  → 4 figures.
- `--resume` correctly reports "4 to run, 6 already done" and lands on 10 total.
- Every field the spec requires per query is present in the raw records.

---

## [0.1.1] — 2026-08-16

Pre-flight corrections before the first paid run. Both tasks free; no Bedrock
call made. 75 tests pass (13 new).

### Task A — λ grid resolution

**Problem.** A `DRY_RUN` sweep produced uniform iteration counts at almost every
λ on the old decade-spaced grid. Only λ≈60 showed spread. At a degenerate λ the
paper's central figure — CAES spread across 1–5 versus Fixed flat at N —
collapses to a single bar, which looks identical to a fixed policy.

**Changed**
- `COARSE_GRID` → `[1, 3, 10, 30, 100, 300, 1000]` (half-decade steps).
- `refine_grid()` now refines **log-spaced**,
  `centre × [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]`, replacing linear
  `× [0.25, 0.5, 2.0, 4.0]`.

**Added**
- `iteration_distribution()`, `format_distribution()`, `max_bucket_share()`,
  `spread_is_degenerate()`, `best_spread_row()`, `format_row()` in
  `tune_lambda.py`.
- `iteration_dist` and `max_bucket_share` columns in
  `results/lambda_sweep.csv`.
- Per-λ progress lines now show the spread inline and flag single-bucket λ
  while the sweep runs, rather than leaving it to be discovered in the CSV.
- A spread warning at the recommended λ (≥90% in one bucket) that **names the
  best-spread alternative** with its F1 and cost. Warning only — never exits
  non-zero, because the λ may be legitimately optimal and real data may spread
  where synthetic data does not.
- `tests/test_tune_lambda.py` (13 tests) pinning grid spacing, log-scaling of
  refinement, distribution accounting, and both degeneracy guards.
- Decision record **[D-21]** in `DECISIONS.md`, plus a traps-table row.

**Measured on the synthetic corpus (40 questions).** The sensitive band is
λ∈[40,70], best spread at λ=50 (55% largest bucket):

| λ | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|
| largest bucket | 100% | 90% | 75% | **55%** | 72% | 92% | 95% | 100% |

The coarse grid brackets the band without landing in it — but `refine_grid(30)`
reaches 42 and 60, and `refine_grid(100)` reaches 50 and 70. **Log refinement
from either bracketing coarse point covers the band**; linear refinement from
λ=30 would have topped out at 120 and missed the lower half entirely. This is
the concrete justification for the log-spacing change.

**Known limitation, not fixed.** The refinement centre is chosen by
F1-per-dollar alone. If that heuristic is uninformative — as under `DRY_RUN`,
where F1 is identically zero and the knee lands at λ=1000 — refinement explores
the wrong region entirely. The existing `f1_is_flat()` guard catches exactly
this case and blocks before the recommendation, so the failure is loud rather
than silent. On real data with a responsive F1 curve the knee should land in a
sensible region. Worth re-checking after Task E.

### Task B — pricing constants verified

`PRICE_HAIKU_INPUT_PER_1K` and `PRICE_HAIKU_OUTPUT_PER_1K` **confirmed correct**
at $1.00 / $5.00 per 1M tokens.

**Fixed — `PRICE_TITAN_EMBED_PER_1K` was 5.5× too high.** It read `0.00011`
($0.11/1M). The actual rate for `amazon.titan-embed-text-v2:0` is **$0.02/1M**
(`0.00002`). The old value is approximately the *previous generation's* price —
Titan Embeddings G1 / v1 bill at $0.10/1M — so this looks like a v1 figure
carried into a v2 config. Corroborated across three independent trackers.

**Impact.**
- Phase 1 ingest: ~$0.21 → **~$0.04** for ~1.9M tokens.
- Per-query ΔC: **negligible.** Embeddings are ~0.03% of a three-iteration
  query; the verifier LLM call dominates at ~$0.00145 each. ΔC is in practice
  almost entirely verifier and planner calls, which is why
  `VERIFIER_CHUNK_CHARS` is the real lever on gate overhead.

No previously-recorded spend is affected — the ledger is empty and no paid call
has ever been made.

**Also updated.** `config.py` carries the verification date, the source note,
and an explicit warning about the v1/v2 confusion. README cost table, Phase 1
estimate, prerequisites, and the λ-grid description brought in line.

---

## Unreleased / next

Nothing is committed to these; they are the open items at the time of writing.

- **Verify the three price constants** in `config.py` against current Bedrock
  pricing. They drive every number in the paper.
- **Run `ingest.py`** against real HotpotQA (~$0.50).
- **Run `calibrate_verifier.py --n 30`** (~$1). This is a blocker: if coverage
  does not spread, sharpen `VERIFIER_PROMPT` before going further.
- **Tune λ** and write the value into `config.py`. Dry-run behaviour suggests
  the live region is roughly λ ∈ [15, 80] given per-iteration costs around
  $0.0018 and typical ΔQ of 0.03–0.15 — the coarse grid brackets it, but the
  refinement step will do the real work.
- **Run the three-way comparison** and regenerate the figures.
- Consider a per-request cost meter so `api.py` no longer needs its global lock.
- Consider recording retrieval-precision-at-k against HotpotQA supporting titles,
  to separate "the gate stopped too early" from "retrieval never found it".

---

## [0.2.0] — 2026-08-31

**The evaluation provider changes from AWS Bedrock to Google Gemini.** Not a
preference: Bedrock model invocation is blocked account-wide on this AWS
account. Rationale, consequences and what was rejected are in
[DECISIONS.md](DECISIONS.md) **[D-22]** and **[D-23]**.

Still no paid call has been made. Ingest, calibration, λ tuning and the
experiments all remain ahead.

### The block

`ListFoundationModels` succeeds and returns 120 models. `InvokeModel` fails with
`ValidationException: Operation not allowed` for `amazon.titan-embed-text-v2:0`,
`amazon.nova-lite-v1:0`, **and** through the AWS console playground — so it is
not model-specific, not SDK-related, and not fixable from our side. A support
case is open and unresolved. The project cannot wait on it.

**AWS is not abandoned.** Bedrock was only the model-serving layer. S3, EC2 and
CloudWatch remain in the deployment and are still free-tier.

### Added

- **`config.PROVIDER`** (`"bedrock" | "gemini"`, default `"gemini"`,
  overridable with `CAES_PROVIDER`). `config.provider_settings()` resolves every
  name that varies with it, so the switch has exactly one definition.
- **Gemini price constants**, verified 2026-08-31 against
  `ai.google.dev/gemini-api/docs/pricing` (paid tier, Standard row), in the same
  style as the Titan correction in `[0.1.1]`:
  `gemini-2.5-flash` $0.30/1M in, $2.50/1M out; `gemini-embedding-001` $0.15/1M.
  Both models bill **nothing** on the free tier; these are list prices used for
  notional accounting only.
- **`python -m llm --check`** — a preflight making two tiny real calls, reporting
  the embedding dimension actually returned, whether `temperature=0.0` was
  accepted, and the measured token counts. Run before any ingest.
- **`GEMINI_MAX_RPM`** client-side pacer and `429 RESOURCE_EXHAUSTED` retry
  honouring the server-supplied `retryDelay`. Free-tier rate limiting is the
  binding constraint now, not a credit balance.
- **`EMBED_TOKENS_MODE`** — Gemini's `:embedContent` returns no token count, so
  the count is measured with a separate `:countTokens` call rather than
  estimated. `"estimated"` exists, warns loudly, and must be opted into. **[D-23]**
- **`tests/conftest.py`** with fake transports for both providers, and
  **`tests/test_provider.py`**. Transport-level tests now run once per provider;
  the provider not under test is booby-trapped so a mis-routed call fails loudly.
- **Index/model mismatch guard** in `Retriever`: an index whose dimension
  disagrees with `config.EMBED_DIM` is refused with both numbers named, and a
  `meta.json` naming a different embedding model warns. Previously this would
  have been a *silent* retrieval failure.

### Changed

- **`bedrock.py` → `llm.py`** (`git mv`, history preserved). Every import
  updated: `graph.py`, `ingest.py`, `retrieval.py`, `devdata.py`, `api.py`,
  `smoke.py`, `agents/*`, and the tests. The Bedrock path is kept fully working
  and fully tested — the support case may resolve, and a two-provider robustness
  result would strengthen the paper.
- **Price constants renamed** `PRICE_HAIKU_*` / `PRICE_TITAN_EMBED_PER_1K` →
  `PRICE_LLM_INPUT_PER_1K` / `PRICE_LLM_OUTPUT_PER_1K` / `PRICE_EMBED_PER_1K`,
  resolved from provider-prefixed constants. The published Bedrock figures are
  unchanged and now pinned by test.
- **`EMBED_DIM` 1024 → 768** on the Gemini path. `gemini-embedding-001` returns
  3072 by default; 768 is a supported Matryoshka truncation that keeps a flat
  index over ~20k chunks near 60 MB rather than 250 MB, which matters on the
  t3.micro host. **Truncated vectors are not unit-norm as returned**, so
  `llm.embed` now re-normalises unconditionally — without that, `IndexFlatIP`
  stops meaning cosine.
- **Cost-guard message** no longer names Bedrock.
- Docs: `DECISIONS.md` §3, §4, §6 and records **[D-2]**, **[D-12]**, **[D-13]**,
  **[D-19]**; `METHODOLOGY.md` §3.2, §3.3, §10 (External and Construct) and §11;
  `IMPLEMENTATION.md` §1, §2, §5 and §10.

### Fixed

- `_estimate_tokens` is no longer reachable as a billing path on any provider.
  Gemini LLM responses missing `usageMetadata` counts raise, matching **[D-2]**;
  `countTokens` responses missing `totalTokens` raise for the same reason.

### Models substituted

The migration plan named `gemini-2.0-flash` and `text-embedding-004`. **Both
were retired before this landed** — 2.0 Flash is listed under "Previous models /
Shut down" and `text-embedding-004` is gone from the model list entirely
(checked 2026-08-31). Substituted with their nearest live equivalents in the
same tier, `gemini-2.5-flash` and `gemini-embedding-001`.

### Two Gemini specifics that are cost-correctness issues

- **Thinking is ON by default on `gemini-2.5-flash` and thinking tokens bill as
  output.** Disabled with `thinkingConfig.thinkingBudget = 0`. Left on it would
  inflate ΔC and, being variable in length, undermine the determinism **[D-19]**
  relies on for cache correctness. `thoughtsTokenCount` is folded into the output
  count regardless, so an ignored budget surfaces as measured cost rather than as
  a silently understated ΔC.
- **`temperature` moves to `generationConfig.temperature`.** Pinned for both
  providers by `test_temperature_zero_is_sent_on_both_providers`. Whether the
  service honours 0.0 in practice is an empirical question `python -m llm --check`
  answers; it has not been run, because it needs a key.

### Consequences for the results

- **Actual spend is now structurally $0.00.** ΔC is entirely list-price notional.
  The ledger and `HARD_BUDGET_USD` become vestigial safety rather than active
  constraints — kept deliberately, because the Bedrock path may return and a
  guard that only exists when needed is a guard that is wrong when it is needed.
  The paper must report costs as **list-price-derived, not billed**.
- **λ does not transfer.** Gemini input tokens are ~3× cheaper and output ~2×
  cheaper than the Bedrock configuration, so the sensitive band moves. The
  synthetic λ∈[40,70] finding from **[D-21]** does not carry over.
- **Any prior expectation about the verifier rubric is invalid.** A different
  model may use the four coverage bands differently, so `calibrate_verifier.py`
  matters more than before, not less.

### Verification

- 113 tests pass (was 75), on `gemini` and on `CAES_PROVIDER=bedrock`.
- `DRY_RUN=1 python -m smoke` reports `$0.00` with the ledger unmoved, on both
  providers.
- `CAES_PROVIDER=bedrock` still constructs its `bedrock-runtime` client.
- No `GEMINI_API_KEY` value in the repo or its history; pinned by
  `test_no_api_key_value_is_committed`.

---

## [0.2.1] — 2026-08-31

AWS stays in the deployment. Bedrock was only the model-serving layer; four of
the six layers still run on AWS at zero cost, and now demonstrably do.

### Added

- **`observability.py` — optional CloudWatch publishing.** `--cloudwatch` on
  `experiments/run.py` publishes per-iteration cost, latency, coverage and
  iteration count to the `CAES-RAG` namespace. **Off by default**, so
  experiments stay runnable offline and with no AWS credentials.

  Not decoration: METHODOLOGY §3.2 defines ΔC as *measured* marginal cost,
  metered per iteration, and this makes it observable per iteration **in the
  deployment** rather than only inside the process.

  A publish failure is logged, counted in the run summary, and swallowed — an
  observability backend must never be able to fail an experiment that is
  otherwise producing valid data. Verified by deliberately failing publisher.
- **`--cloudwatch-no-dimensions`.** Cardinality is four metric names × a
  `Policy` dimension = four metrics per policy, so a three-policy run creates
  twelve against a free allowance of ten. This flag collapses it to four.
  `Iteration` is deliberately *not* a dimension: it would multiply cardinality
  by `MAX_ITERATIONS` for a by-index breakdown, while each iteration already
  emits its own datapoint.
- **`tests/test_aws.py`** — 8 tests. The S3 upload is validated against the real
  service model with botocore's `Stubber`, so a malformed `put_object` fails in
  the suite rather than at ingest time; CloudWatch batching, dimensioning, the
  1000-datum `PutMetricData` limit and the never-fail-the-run guarantee are
  covered against a recording double. Neither reaches AWS.

### Changed

- `graph.state_summary` now carries `latency_history` alongside `cost_history`
  and `coverage_history`. Additive; needed to publish latency per iteration, and
  useful for the same reason the other two series are recorded.
- README documents the four AWS layers that remain (S3 corpus, EC2 t3.micro host
  for FAISS/LangGraph/FastAPI, CloudWatch metrics, all free tier) and why the
  768-dimension embedding choice is partly a t3.micro memory decision.

### Verified

- 121 tests pass (was 113) on both providers; `DRY_RUN=1 python -m smoke` still
  reports `$0.00` with the ledger unmoved.
- `--cloudwatch` exercised end to end under `DRY_RUN=1`: 30 datums buffered
  across 3 queries, the publish refused, **the run completed normally and the
  results were unaffected** — which is the designed behaviour.

### Two AWS findings that need the researcher's action

- **`cloudwatch:PutMetricData` is denied** for `iam::099868052312:user/sid_nevgi`
  — *"no identity-based policy allows the cloudwatch:PutMetricData action"*.
  `ListMetrics` is denied too. The code path is verified; the permission is not
  granted. Attach a policy allowing `cloudwatch:PutMetricData` before relying on
  `--cloudwatch` for a real run.
- **No S3 bucket exists** in the account (`ListBuckets` succeeds and returns
  none). `ingest.py --upload-s3 BUCKET` is verified against the S3 service model
  but has not been run live, because that needs a bucket to be created — a
  persistent resource in the account, and not something to create unasked.

---

## [0.3.0] — 2026-08-31

**Free-tier daily quotas turned out to be the binding constraint, and they set
the experiment's scale.** Rationale in [DECISIONS.md](DECISIONS.md) **[D-24]**.

Still no paid call. Ingest, calibration, λ tuning and the experiments remain
ahead — but they are now feasible, which they were not this morning.

### The finding

Quotas read off `aistudio.google.com/rate-limit`:

| Model | RPM | TPM | RPD |
|---|---|---|---|
| `gemini-2.5-flash` | 5 | 250K | **20** |
| `gemini-3.5-flash-lite` | 15 | 250K | **500** |
| `gemini-embedding-001` | 100 | 30K | **1,000** |

**Requests per day, not per minute, is the wall.** At 20 RPD the ~7,200-call
experiment needs 360 days. Pacing cannot help — the limit is how many times you
ask, not how fast.

### Changed

- **LLM: `gemini-2.5-flash` → `gemini-3.5-flash-lite`.** Same list price
  ($0.30/$2.50 per 1M), **25× the daily requests**. Experiment drops 360 days →
  ~15. Chosen for quota, not cost; the cost of the choice is that "Lite" is a
  weaker verifier, and the verifier is the instrument.
- **`CORPUS_SAMPLE_SIZE` 2000 → 500.** Ingest embeds the corpus up front against
  a separate 1,000/day quota: 18k chunks (~36k requests with [D-23]'s
  `countTokens`) is 36 days; 4.5k chunks is ~9.

  **This is a research change.** The evaluation split is untouched — still 50
  tune + 150 test — but the *distractor pool* is four times thinner, so
  retrieval is easier and **absolute F1 will read high against published
  HotpotQA baselines**. It inflates F1 for all three arms identically, so the
  relative claim survives; external validity does not. Recorded in
  METHODOLOGY §10 (External).
- **Pacer split per model.** `GEMINI_MAX_RPM` was one shared 15 — 3× the LLM's
  real limit and a sixth of the embedding model's. The two draw on separate
  quota buckets, so a shared pacer necessarily overran one or throttled the
  other. Now `GEMINI_LLM_RPM` / `GEMINI_EMBED_RPM`, tracked separately.
- **Thinking control keyed by model.** 2.5 disables thinking with
  `thinkingConfig.thinkingBudget`; 3.x moved to `thinkingLevel`. Sending the
  wrong field is a 400 on *every* call. `GEMINI_THINKING_CONFIG` maps 3.x Lite
  models to `None` (send nothing — thinking is off by default there). That is an
  assumption about a default, so it is verified rather than trusted: the
  preflight now reports output tokens for a one-word reply, and any non-zero
  `thoughtsTokenCount` is still folded into output so unexpected thinking shows
  up as measured cost rather than an understated ΔC.

### Added

- **`llm.QuotaExhausted`** — raised on a 429 whose `QuotaFailure.quotaId` names a
  per-day limit, or whose `retryDelay` exceeds five minutes. **Deliberately not
  retried:** a per-minute limit clears in seconds, a per-day limit does not clear
  until tomorrow, and spending five backoff attempts on it teaches nothing.
- **Clean resume paths.** `ingest.py`, `experiments/run.py` and
  `calibrate_verifier.py` each catch it and exit with resume instructions
  instead of a stack trace. Nothing is lost: the disk cache already makes
  completed work free to replay, so a resumed run spends quota only on new work
  — the cache earning its keep for a reason [D-12] never anticipated.
  `calibrate_verifier` explicitly refuses to *judge* the instrument on a
  truncated sample, since its pass criteria are distributional.
- **Pre-flight feasibility warning** in `experiments/run.py`: projected request
  count and day count against the daily cap, printed before the run starts.
- Six tests: per-model pacing buckets, per-day vs per-minute 429
  discrimination, `QuotaExhausted` not being retried, thinking-config matching
  the model family, every configured model having an explicit thinking decision,
  and a refusal to run a model whose RPM has not been read off the dashboard.

### Verified

- 127 tests pass (was 123) on both providers; `DRY_RUN=1 python -m smoke` still
  reports `$0.00` with the ledger unmoved.

### Corrections to the previous estimate

`calibrate_verifier` runs **one** verify per question, not one per iteration, so
Task 5 is ~30 LLM calls rather than the 300 previously projected — comfortably
inside a single day's quota.

### Still open

- **Whether `gemini-3.5-flash-lite` accepts the request as shaped.** No real
  call has been made against it. `python -m llm --check` settles it.
- **Whether it discriminates coverage.** Task 5 is the go/no-go, and it matters
  more after a model change, not less.
- The ledger/[D-22] discrepancy, still open question 6.

---

## [0.3.1] — 2026-08-31

Pre-ingest hardening. Three defects found by rehearsing Task 4 rather than
starting it — each would have surfaced partway into a multi-day run.

### Fixed

- **`load_dataset("hotpot_qa", ...)` no longer works.** `datasets` 3.0 removed
  script-based dataset loading and 5.x rejects bare repo ids outright
  (*"Repository id must be 'namespace/name'"*). Installed here is 5.0.1, so
  ingest would have failed on its first call, before a single embedding.
  Corrected to the namespaced `hotpotqa/hotpot_qa`, which works on old and new
  versions alike. Schema verified unchanged against 5.0.1:
  `context.{title,sentences}`, `supporting_facts.{title,sent_id}`.
- **A resumed ingest re-issued its batched `countTokens` requests.** Ingest
  spans ~6 days of free-tier quota and is therefore restarted repeatedly; the
  embeddings themselves replayed from cache correctly, but the per-batch token
  counts did not, burning ~112 requests of each day's allowance on finished
  work. Batch counts are now cached by batch contents like any other provider
  call. Verified: a second pass issues **zero** new requests, returns identical
  vectors, and reports an identical token total.
- **`count_tokens` ignored `DRY_RUN`.** Every other network path is exercisable
  for $0.00; this one raised on the missing API key, which meant a multi-day
  ingest could not be rehearsed before being committed to. That is how the
  resume bug above stayed hidden.

### Measured, replacing estimates

Corpus built for real (free — no API key needed, dataset processing only):

| | Estimated | **Actual** |
|---|---:|---:|
| Chunks (500 questions) | 4,500 | **5,552** |
| Chunks per question | 9.0 | **11.1** |
| Embedding requests | 4,590 | **5,664** |
| Ingest days at 1,000/day | 4.6 | **5.7** |
| Ingest cost (notional) | — | **~$0.12** |

### Added

- Two regression tests: a resumed ingest issues no new provider requests, and
  `count_tokens` works under `DRY_RUN`.
- The preflight verifies batched `countTokens` before ingest depends on it.

### Verified live

`python -m llm --check` against `gemini-3.5-flash-lite`:

- `temperature=0.0` accepted; **1 output token** for a one-word reply, so
  thinking is confirmed off and omitting the field was correct.
- Embedding dimension **768**, L2 norm `1.000000` after re-normalisation.
- Batched `countTokens` **exact**: 12 for three texts, `[4, 4, 4]` individually.
- **[D-12] confirmed against the live API**: a repeated call returned
  `notional $0.00000430 / actual $0.00000000` with the ledger unmoved — a cache
  hit accruing notional cost while recording nothing to the ledger, which is
  the behaviour the gate's cache-neutrality depends on.

---

## [0.3.2] — 2026-08-31

**First real ingest run. 1,517 of 5,552 chunks embedded before the daily
embedding quota tripped, then a clean stop.** No index yet — `data/` is empty
by design, since `ingest.py` writes the FAISS index only after every chunk is
embedded.

### The quota machinery worked

`QuotaExhausted` fired on Google's per-day `embed_content_free_tier_requests`
violation, was **not** retried, and exited with a progress figure and resume
instructions rather than a stack trace. Exactly the behaviour [D-24] specifies.

Worth noting: Google's error suggested `retryDelay: 45.4s`, which would be
useless advice for a cap that resets tomorrow. The per-day detection correctly
ignored it and keyed on the quota id instead.

### Resume verified before relying on it

Rebuilt the corpus from scratch and checked cache membership without issuing a
single request:

| | |
|---|---:|
| Chunks rebuilt | 5,552 (identical to the run) |
| Cached | **1,517**, a contiguous prefix, first miss at index 1517 |
| Batch token counts cached | 31 of 112 |
| Requests remaining | **4,116** |

Corpus construction is deterministic — fixed split seed, insertion-ordered
dedup — so tomorrow's run replays the finished prefix from disk and continues
at chunk 1,518.

### Measured

| | |
|---|---:|
| Requests before the cap | **~1,549** (1,518 embeddings + 31 counts) |
| Wall time | **22 min** for a full day's quota |
| Throughput | ~69 requests/min against a 100 RPM pacer |
| Ledger after run | $0.0331 list-price, **$0.00 billed** |

**The dashboard's 1,000 RPD was not the effective limit** — the cap tripped at
~1,549. `GEMINI_EMBED_RPD` stays at the documented 1,000 anyway: it only feeds
a projected day count, and over-estimating days is the safe direction. At the
observed rate ingest completes in **~2.7 more days**, not the 4.1 the
conservative figure implies.

### Not a defect

The `multiprocess.ResourceTracker` `AttributeError` at interpreter shutdown
comes from a `datasets` dependency on Windows, fires after the run has
finished, and affects nothing.

---

## [0.4.0] — 2026-08-31

**Phase 1 complete. The corpus is ingested and the index is built.** First
phase of the project to run against real data and real model responses.

### Result

| | |
|---|---:|
| Questions | 500 |
| Deduplicated passages | 4,965 |
| **Chunks embedded** | **5,552** |
| Index | `IndexFlatIP`, 5,552 × 768, 17.1 MB |
| Input tokens (exact) | **684,090** |
| **Notional cost** | **$0.1026** |
| **Actual billed** | **$0.00** |
| Elapsed | 4 sessions across the daily embedding quota |

`meta.json` carries **no `synthetic` stamp** and `dry_run: false` — this is the
real HotpotQA corpus, not the devdata scaffold.

### Verified after the fact

* Index `ntotal` (5,552) equals `chunks.jsonl` length; dimension 768 matches
  `config.EMBED_DIM`.
* Splits: 50 tune, 150 test, **0 overlap**.
* **Self-retrieval 5/5 at score exactly `1.0000`** — probing the index with a
  chunk's own cached vector ranks that chunk first at unit cosine. This
  confirms the re-normalisation in `llm.embed` works and that `IndexFlatIP`
  really is cosine here, which was the risk [D-22] flagged when `EMBED_DIM`
  moved to a Matryoshka-truncated 768.

### The character-ratio estimator overestimates by 18.7%

The ledger accrued **$0.1218** from per-chunk estimates while the batched
`countTokens` measurement gives **$0.1026**. Actual is 4.99 chars/token against
the estimator's assumed 3.6.

This is the [D-23] split behaving exactly as designed — the *reported* corpus
cost is the measured one, and the estimate never reaches a result. It is also a
useful calibration of **[D-1]**: the pre-flight budget guard uses the same
ratio, so it over-projects spend by roughly a fifth and therefore trips early.
That is the direction [D-1] deliberately chose.

### Retry logic exercised in anger

Two `503 Service Unavailable` responses during the final batches were retried
with backoff and recovered — the run completed without intervention.

### Fixed

- **The pre-flight projected the whole corpus, not the outstanding work.** A run
  98% finished still announced "5,664 requests → 5.7 days". It now counts only
  uncached chunks and prints resume progress: *"Resuming: 5,520 of 5,552 chunks
  already embedded (99% done)."*

### Note

No `HF_TOKEN` is needed. `hotpotqa/hotpot_qa` is public, and the 914 MB local
cache means subsequent loads take ~13 s from disk. The library's warning is
generic advice, not a limit this project hit.

---
