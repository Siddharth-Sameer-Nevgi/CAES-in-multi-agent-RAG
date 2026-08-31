"""Phase 5: the experiment driver.

    python -m experiments.run --policy {fixed,oneshot,caes} --n 150 --max-usd 5 --yes
    python -m experiments.run --policy caes --resume
    python -m experiments.run --policy caes --n 150 --yes --cloudwatch

Checkpoints after every query, so a crash at query 130 does not lose the first
129. --resume skips query ids already present in the output file.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import config

log = logging.getLogger("caes.run")

# Empirical per-iteration cost used only for the pre-flight projection:
# planner + verifier LLM calls plus one query embedding.
EST_USD_PER_ITERATION = 0.0015
EST_USD_PER_GENERATION = 0.0010


def raw_path(policy: str) -> Path:
    return config.RESULTS_DIR / f"{policy}_raw.jsonl"


def load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write can leave one torn final line. Drop it.
                log.warning("Skipping malformed checkpoint line in %s", path)
                continue
            done[rec["query_id"]] = rec
    return done


def expected_iterations(policy_name: str, question: str) -> float:
    """Rough per-query depth, for the pre-flight cost projection only."""
    if policy_name == "fixed":
        return 3.0
    if policy_name == "oneshot":
        from policies import complexity_score, score_to_depth
        return float(score_to_depth(complexity_score(question)))
    return 2.5   # caes: assume it lands between the baselines


def project_cost(policy_name: str, questions: list[dict]) -> float:
    iters = sum(expected_iterations(policy_name, q["question"]) for q in questions)
    return iters * EST_USD_PER_ITERATION + len(questions) * EST_USD_PER_GENERATION


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Run one policy over the test set.")
    ap.add_argument("--policy", required=True,
                    choices=["fixed", "oneshot", "caes", "threshold"])
    ap.add_argument("--n", type=int, default=config.N_TEST)
    ap.add_argument("--max-usd", type=float, default=config.SINGLE_RUN_MAX_USD)
    ap.add_argument("--fixed-n", type=int, default=3,
                    help="iteration count for --policy fixed")
    ap.add_argument("--lam", type=float, default=None,
                    help="override config.LAMBDA (for diagnostics only)")
    ap.add_argument("--resume", action="store_true",
                    help="skip query ids already in the output file")
    ap.add_argument("--yes", action="store_true",
                    help="proceed past the cost confirmation")
    ap.add_argument("--out", type=Path, default=None)
    # Off by default so experiments stay runnable offline and without AWS
    # credentials. Publishing per-iteration cost is what makes dC observable in
    # the deployment rather than merely computed; see observability.py.
    ap.add_argument("--cloudwatch", action="store_true",
                    help="publish per-iteration cost/latency/coverage/depth to "
                         "the CAES-RAG CloudWatch namespace (off by default)")
    ap.add_argument("--cloudwatch-no-dimensions", action="store_true",
                    help="drop the Policy dimension: 4 unique metrics total "
                         "instead of 4 per policy, to stay inside the free tier")
    args = ap.parse_args(argv)

    from costs import TRACKER
    from graph import run_query, state_summary
    from metrics import score
    from policies import build_policy
    from splits import test_set

    questions = test_set()[:args.n]
    out_path = args.out or raw_path(args.policy)

    completed = load_completed(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        print(f"{out_path} already exists. Pass --resume to continue it, or "
              f"delete it to start over.", file=sys.stderr)
        return 2
    todo = [q for q in questions if q["id"] not in completed]

    # ---- pre-flight ----
    projected = project_cost(args.policy, todo)
    print("=" * 66)
    print(f"policy          : {args.policy}"
          + (f" (n={args.fixed_n})" if args.policy == "fixed" else ""))
    print(f"questions       : {len(todo)} to run"
          + (f", {len(completed)} already done" if completed else ""))
    print(f"projected cost  : ${projected:.2f}")
    print(f"run allowance   : ${args.max_usd:.2f}")
    print(f"cumulative spend: ${TRACKER.cumulative():.2f} of "
          f"${config.HARD_BUDGET_USD:.2f} "
          f"(${TRACKER.remaining():.2f} left)")
    print(f"provider        : {config.PROVIDER} ({config.MODEL_LLM})")
    if args.cloudwatch:
        print(f"cloudwatch      : on, namespace {config.CLOUDWATCH_NAMESPACE}"
              + (" (no dimensions)" if args.cloudwatch_no_dimensions else ""))
    print("=" * 66)
    # Free-tier daily caps, not money, are what actually stop a run. Say so
    # before it starts rather than after it dies partway.
    if config.PROVIDER == "gemini" and config.GEMINI_LLM_RPD > 0 and todo:
        est_calls = sum(2 * expected_iterations(args.policy, q["question"]) + 1
                        for q in todo)
        days = est_calls / config.GEMINI_LLM_RPD
        print(f"est. LLM requests: {est_calls:.0f} against a "
              f"{config.GEMINI_LLM_RPD}/day cap -> {days:.1f} day(s)")
        if days > 1.0:
            print(f"NOTE: this run needs more than one day of quota. It will "
                  f"stop when the cap is hit; re-run with --resume tomorrow. "
                  f"Cached calls replay free, so no quota is re-spent.")
        print("=" * 66)

    if projected > args.max_usd:
        print(f"WARNING: projection exceeds the run allowance. The run-budget "
              f"guard will stop it partway; use --resume afterwards.")
    if not args.yes:
        print("\nRe-run with --yes to proceed.")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    policy = build_policy(args.policy, n=args.fixed_n, lam=args.lam)

    publisher = None
    if args.cloudwatch:
        from observability import CloudWatchPublisher
        publisher = CloudWatchPublisher(
            policy=args.policy,
            dimensions=not args.cloudwatch_no_dimensions)

    # ---- run with checkpointing ----
    t_start = time.perf_counter()
    n_done = 0
    stopped_early = False
    import llm as llm_mod
    from costs import BudgetExceeded

    with out_path.open("a", encoding="utf-8") as fh:
        try:
            with TRACKER.run_budget(args.max_usd, name=f"run-{args.policy}"):
                for i, q in enumerate(todo, 1):
                    final = run_query(q["question"], policy,
                                     query_id=q["id"],
                                     gold_titles=q.get("supporting_titles"))
                    rec = state_summary(final)
                    rec["gold_answer"] = q["answer"]
                    rec.update(score(rec["answer"], q["answer"]))
                    rec["level"] = q.get("level", "")
                    rec["type"] = q.get("type", "")

                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()          # checkpoint after EVERY query
                    n_done += 1

                    if publisher is not None:
                        # Never allowed to fail the run; see observability.py.
                        publisher.record_query(rec)

                    print(f"[{i:>3}/{len(todo)}] {q['id'][:12]} "
                          f"it={rec['iterations_used']} "
                          f"stop={rec['stop_reason']:<9} "
                          f"cov={rec['final_coverage']:.2f} "
                          f"f1={rec['f1']:.2f} "
                          f"${rec['total_usd']:.5f}")
        except llm_mod.QuotaExhausted as exc:
            stopped_early = True
            print(f"{chr(10)}DAILY QUOTA SPENT: {exc}")
            print(f"{n_done} queries checkpointed to {out_path}. Re-run with "
                  f"--resume tomorrow; cached calls replay for free, so the "
                  f"quota is spent only on new work.")
        except BudgetExceeded as exc:
            stopped_early = True
            print(f"\nSTOPPED BY BUDGET GUARD: {exc}")
            print(f"{n_done} queries were checkpointed to {out_path}. "
                  f"Re-run with --resume after raising the allowance.")
        except KeyboardInterrupt:
            stopped_early = True
            print(f"\nInterrupted. {n_done} queries checkpointed to {out_path}.")

    # ---- summary ----
    all_recs = load_completed(out_path)
    recs = list(all_recs.values())
    elapsed = time.perf_counter() - t_start
    print("\n" + "-" * 66)
    print(f"{args.policy}: {len(recs)} total results in {out_path}")
    if recs:
        print(f"  mean iterations : {statistics.mean(r['iterations_used'] for r in recs):.2f}")
        print(f"  mean cost       : ${statistics.mean(r['total_usd'] for r in recs):.5f}")
        print(f"  mean latency    : {statistics.mean(r['total_latency_ms'] for r in recs):.0f} ms")
        print(f"  exact match     : {statistics.mean(r['exact_match'] for r in recs):.3f}")
        print(f"  F1              : {statistics.mean(r['f1'] for r in recs):.3f}")
        print(f"  abstention rate : {statistics.mean(r['abstained'] for r in recs):.3f}")
        stops: dict[str, int] = {}
        for r in recs:
            stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
        print(f"  stop reasons    : {stops}")
    print(f"  wall time       : {elapsed:.0f}s")
    print(f"  cumulative spend: ${TRACKER.cumulative():.4f}")

    if publisher is not None:
        publisher.flush()
        print(f"  {publisher.summary()}")

    from cache import CACHE
    CACHE.log_stats("run cache")
    TRACKER.to_csv(config.RESULTS_DIR / "cost_records.csv")
    return 1 if stopped_early else 0


if __name__ == "__main__":
    sys.exit(main())
