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

# Roughly half-decade steps. The gate's sensitive band -- where lambda*dC is
# comparable to dQ and iteration counts actually vary -- is narrow, and a
# decade-spaced grid steps clean over it. See [D-21] in DECISIONS.md.
# EXTENDED 2026-08-31 from a 1000 ceiling. Simulating the gate on measured
# calibration trajectories showed the histogram identical for every lambda from
# 1 to 300, degenerate at 1000 (100% of queries at depth 2), and only splitting
# at 3000. The sensitive band is ~1000-10000, so a grid stopping at 1000 steps
# clean over it -- exactly the failure [D-21] was written about, one decade up.
#
# The band moved because dC shrank: Gemini input tokens are ~3x cheaper than
# the Bedrock configuration the synthetic lambda-in-[40,70] finding came from,
# and lambda must grow to keep lambda*dC comparable to dQ. Half-decade spacing
# is preserved. See DECISIONS [D-28].
COARSE_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

# Multipliers for the refinement pass, log-spaced about the knee. Linear
# refinement around a centre found on a log grid oversamples one side.
REFINE_MULTIPLIERS = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]

# A lambda is "degenerate on spread" when this share of queries or more land in
# a single iteration bucket. Warning only, never a blocker.
SPREAD_WARN_SHARE = 0.90


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

    dist = iteration_distribution(per_query)
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
        "iteration_dist": format_distribution(dist),
        "max_bucket_share": max_bucket_share(dist),
        "_per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Iteration spread
# ---------------------------------------------------------------------------

def iteration_distribution(per_query: list[dict]) -> dict[int, int]:
    """Count queries per iteration bucket, 1..MAX_ITERATIONS."""
    dist = {i: 0 for i in range(1, config.MAX_ITERATIONS + 1)}
    for r in per_query:
        used = r["iterations_used"]
        dist[used] = dist.get(used, 0) + 1
    return dist


def format_distribution(dist: dict[int, int]) -> str:
    """Compact, CSV-safe encoding: '1:0|2:31|3:9|4:0|5:0'."""
    return "|".join(f"{k}:{dist[k]}" for k in sorted(dist))


def max_bucket_share(dist: dict[int, int]) -> float:
    total = sum(dist.values())
    return (max(dist.values()) / total) if total else 0.0


def format_row(r: dict) -> str:
    """One progress line per lambda, including the iteration spread.

    Spread is shown inline because it is the property the paper's central
    figure depends on: a lambda with good F1-per-dollar and a single-bucket
    histogram is not usable, and that has to be visible while the sweep runs
    rather than discovered afterwards in the CSV.
    """
    flag = "  <- single bucket" if spread_is_degenerate(r) else ""
    return (f"  lambda={r['lambda']:<9g} iters={r['mean_iterations']:.2f}  "
            f"cost=${r['mean_usd']:.5f}  F1={r['f1']:.3f}  "
            f"EM={r['exact_match']:.3f}  spread=[{r['iteration_dist']}]{flag}")


def best_spread_row(rows: list[dict]) -> dict | None:
    """The swept lambda whose iteration histogram is least concentrated.

    Reported alongside a degenerate recommendation so the warning is actionable.
    Deliberately not returned as *the* answer: trading F1-per-dollar for a nicer
    histogram is a judgement about what the experiment should demonstrate, and
    that belongs to the researcher, not to this script.
    """
    scored = [r for r in rows if "max_bucket_share" in r]
    return min(scored, key=lambda r: r["max_bucket_share"]) if scored else None


def spread_is_degenerate(row: dict) -> bool:
    """True when nearly every query stops at the same iteration.

    A lambda can be optimal on F1-per-dollar and still be useless for the paper:
    if the iteration histogram is a single bar, CAES is indistinguishable from a
    fixed policy on the figure that is supposed to demonstrate per-iteration
    granularity. See [D-21].
    """
    return row.get("max_bucket_share", 0.0) >= SPREAD_WARN_SHARE


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
    """Log-spaced refinement about the knee.

    The coarse grid is log-spaced, so the knee is located to within a
    multiplicative factor, not an additive one. Refining linearly would
    oversample above the centre and undersample below it.
    """
    return [round(centre * m, 4) for m in REFINE_MULTIPLIERS]


def write_csv(rows: list[dict], path) -> None:
    fields = ["lambda", "n", "mean_usd", "mean_latency_ms", "mean_iterations",
              "mean_coverage", "exact_match", "f1", "abstention_rate",
              "iteration_dist", "max_bucket_share", "stage"]
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
            print(format_row(r))

        if not args.no_refine:
            centre = find_knee(rows)
            print(f"\nKnee near lambda={centre:g}; refining log-spaced "
                  f"around it.")
            for lam in refine_grid(centre):
                if any(abs(lam - r["lambda"]) < 1e-9 for r in rows):
                    continue
                r = evaluate_lambda(lam, questions)
                r["stage"] = "refine"
                rows.append(r)
                print(format_row(r))

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
    print(f"  iteration spread [{best_row['iteration_dist']}]  "
          f"(largest bucket {best_row['max_bucket_share']:.0%})")

    if spread_is_degenerate(best_row):
        print(f"\n  WARNING: degenerate on spread. {best_row['max_bucket_share']:.0%} "
              f"of queries stop at the same\n"
              f"  iteration, so the iteration histogram will be a single bar and "
              f"CAES will\n"
              f"  look indistinguishable from a fixed policy on the paper's "
              f"central figure.\n"
              f"  This lambda may still be optimal on F1-per-dollar -- the "
              f"warning is about\n"
              f"  what it demonstrates, not what it costs.")
        alt = best_spread_row(rows)
        if alt is not None and alt["lambda"] != best_row["lambda"]:
            print(f"\n  Best spread in this sweep is lambda={alt['lambda']:g}: "
                  f"[{alt['iteration_dist']}]\n"
                  f"  (largest bucket {alt['max_bucket_share']:.0%}, "
                  f"F1 {alt['f1']:.3f}, cost ${alt['mean_usd']:.5f}).\n"
                  f"  Compare the two before choosing; this is a judgement call "
                  f"about what the\n"
                  f"  experiment demonstrates, so it is yours to make, not the "
                  f"script's.")

    print("\nACTION REQUIRED: write this value into config.py as")
    print(f"    LAMBDA = {best:g}")
    print("and do not re-tune. Tuning on the test set invalidates the result.")
    print("=" * 66)
    print(f"Cumulative spend: ${TRACKER.cumulative():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
