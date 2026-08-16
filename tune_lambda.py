"""Phase 4: choose lambda on held-out data.

    python tune_lambda.py --max-usd 5 --yes

Sweeps lambda over a coarse grid, then refines around the knee of the
cost/quality curve. Emits results/lambda_sweep.csv.

Runs on the 50-question tuning split, which is disjoint from the 150-question
test split by construction (see splits.py). Tuning on test data would invalidate
the headline result, so the test split is never loaded here.

The chosen lambda must be written into config.py by hand and never re-tuned.
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys

import config

log = logging.getLogger("caes.tune")

COARSE_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def evaluate_lambda(lam: float, questions: list[dict],
                    honor_confidence: bool = False) -> dict:
    """Run the CAES policy at one lambda over the tuning set."""
    from caes import CAESPolicy
    from graph import run_query, state_summary
    from metrics import score

    policy = CAESPolicy(lam=lam, record=False)
    per_query = []
    for q in questions:
        final = run_query(q["question"], policy,
                          query_id=f"tune-{lam}-{q['id']}",
                          honor_confidence=honor_confidence)
        s = state_summary(final)
        s.update(score(s["answer"], q["answer"]))
        s["gold_answer"] = q["answer"]
        per_query.append(s)

    return {
        "lambda": lam,
        "n": len(per_query),
        "mean_usd": statistics.mean(r["total_usd"] for r in per_query),
        "mean_latency_ms": statistics.mean(
            r["total_latency_ms"] for r in per_query),
        "mean_iterations": statistics.mean(
            r["iterations_used"] for r in per_query),
        "mean_coverage": statistics.mean(r["final_coverage"] for r in per_query),
        "exact_match": statistics.mean(r["exact_match"] for r in per_query),
        "f1": statistics.mean(r["f1"] for r in per_query),
        "abstention_rate": statistics.mean(r["abstained"] for r in per_query),
        "_per_query": per_query,
    }


FLAT_F1_TOLERANCE = 0.01


def f1_is_flat(rows: list[dict]) -> bool:
    """True when quality does not respond to lambda at all.

    If F1 is identical across the whole sweep there is no tradeoff to trade off,
    and F1-per-dollar degenerates into "pick the cheapest" — which would happily
    recommend a lambda that stops at one iteration every time. That is a
    degenerate curve, not a tuned parameter, so it must be surfaced rather than
    silently returned.
    """
    f1s = [r["f1"] for r in rows]
    return bool(f1s) and (max(f1s) - min(f1s)) < FLAT_F1_TOLERANCE


def find_knee(rows: list[dict]) -> float:
    """Lambda with the best F1-per-dollar, used as the centre for refinement.

    Not a claim that F1/$ is the objective — it is a search heuristic for where
    to spend the refinement budget on the curve.
    """
    scored = [(r["f1"] / r["mean_usd"] if r["mean_usd"] > 0 else 0.0, r["lambda"])
              for r in rows]
    return max(scored)[1]


def refine_grid(centre: float) -> list[float]:
    return [round(centre * m, 4) for m in (0.25, 0.5, 2.0, 4.0)]


def write_csv(rows: list[dict], path) -> None:
    fields = ["lambda", "n", "mean_usd", "mean_latency_ms", "mean_iterations",
              "mean_coverage", "exact_match", "f1", "abstention_rate", "stage"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["lambda"]):
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=config.SINGLE_RUN_MAX_USD)
    ap.add_argument("--n", type=int, default=config.N_TUNE)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="skip the cost confirmation prompt")
    args = ap.parse_args(argv)

    from costs import TRACKER
    from splits import tune_set

    questions = tune_set()[:args.n]
    grid = list(COARSE_GRID)

    # Pre-flight: coarse grid, worst case every query runs to MAX_ITERATIONS.
    # ~3 LLM calls + 1 embed per iteration, plus one generate per query.
    per_iter = 0.0015
    projected = len(grid) * len(questions) * config.MAX_ITERATIONS * per_iter
    if not args.no_refine:
        projected *= 1.8
    print(f"Tuning lambda over {grid} on {len(questions)} held-out questions.")
    print(f"Projected worst-case cost: ${projected:.2f} "
          f"(cache hits reduce this substantially on re-runs)")
    print(f"Cumulative spend so far:   ${TRACKER.cumulative():.2f} of "
          f"${config.HARD_BUDGET_USD:.2f}")
    if not args.yes:
        print("\nRe-run with --yes to proceed.")
        return 0

    rows: list[dict] = []
    with TRACKER.run_budget(args.max_usd, name="tune_lambda"):
        for lam in grid:
            r = evaluate_lambda(lam, questions)
            r["stage"] = "coarse"
            rows.append(r)
            print(f"  lambda={lam:<9g} iters={r['mean_iterations']:.2f}  "
                  f"cost=${r['mean_usd']:.5f}  F1={r['f1']:.3f}  "
                  f"EM={r['exact_match']:.3f}")

        if not args.no_refine:
            centre = find_knee(rows)
            print(f"\nKnee near lambda={centre:g}; refining around it.")
            for lam in refine_grid(centre):
                if any(abs(lam - r["lambda"]) < 1e-9 for r in rows):
                    continue
                r = evaluate_lambda(lam, questions)
                r["stage"] = "refine"
                rows.append(r)
                print(f"  lambda={lam:<9g} iters={r['mean_iterations']:.2f}  "
                      f"cost=${r['mean_usd']:.5f}  F1={r['f1']:.3f}  "
                      f"EM={r['exact_match']:.3f}")

    out = config.RESULTS_DIR / "lambda_sweep.csv"
    write_csv(rows, out)
    print(f"\nWrote {out}")

    best = find_knee(rows)
    best_row = next(r for r in rows if r["lambda"] == best)
    print("\n" + "=" * 66)

    if f1_is_flat(rows):
        print("DEGENERATE SWEEP: F1 is effectively constant across every "
              "lambda\n(spread "
              f"{max(r['f1'] for r in rows) - min(r['f1'] for r in rows):.4f} "
              f"< {FLAT_F1_TOLERANCE}).")
        print("\nThere is no cost/quality tradeoff to tune here, so F1-per-dollar\n"
              "degenerates to 'pick the cheapest lambda'. Do NOT write the value\n"
              "below into config.py. Likely causes:")
        print("  - running under DRY_RUN (answers are canned; F1 is meaningless)")
        print("  - the verifier is not discriminating (re-run "
              "calibrate_verifier.py)")
        print("  - the tuning split is too small to separate the policies")
        print("=" * 66)
        return 1

    print(f"Suggested lambda = {best:g}")
    print(f"  mean iterations {best_row['mean_iterations']:.2f}   "
          f"mean cost ${best_row['mean_usd']:.5f}   "
          f"F1 {best_row['f1']:.3f}")
    print("\nACTION REQUIRED: write this value into config.py as")
    print(f"    LAMBDA = {best:g}")
    print("and do not re-tune. Tuning on the test set invalidates the result.")
    print("=" * 66)
    print(f"Cumulative spend: ${TRACKER.cumulative():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
