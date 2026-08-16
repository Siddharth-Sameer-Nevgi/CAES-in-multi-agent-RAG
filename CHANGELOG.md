# Changelog

Progress log for CAES-RAG. Entries are grouped by the implementation phase they
belong to, in the order the phases were built. Each entry records what landed,
and — where it matters — what was found broken and fixed along the way.

This project is not under version control yet, so this file is the history.
When git is initialised, keep appending here; commit messages are not a
substitute, because most of the interesting content is *why*, not *what*.

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
