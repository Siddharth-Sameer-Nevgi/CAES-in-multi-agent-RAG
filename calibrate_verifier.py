"""Phase 2 acceptance gate: is the verifier a usable dQ signal?

    python calibrate_verifier.py --n 30

Two phases:

  A. Iteration-1 marginal -- JSON validity, and the spread of coverage over a
     single retrieval. Cheap, and it catches a verifier that emits one number.

  B. Coverage TRAJECTORIES over MAX_ITERATIONS. This is the phase that matters.
     dQ is a *difference across iterations*, so a marginal distribution cannot
     establish it: a verifier that says 1.0 on easy questions at iteration 1
     and 0.2 on hard ones, then climbs to 1.0 on the hard ones by iteration 3,
     is exactly the signal the gate needs -- and phase A alone reads that as
     "top-heavy and therefore flat". Phase B measures whether coverage RISES
     and FLATTENS, which is the diminishing-returns premise the method rests on
     (METHODOLOGY 3.1).

**Phase B gates; phase A warns.** Phase A's spread criteria are a proxy for
"is the verifier informative?", and phase B measures that directly on the
series the gate actually differentiates. A failing marginal alongside a healthy
trajectory means the TASK is easy, not that the instrument is blunt -- so it is
reported loudly and does not block. With `--no-trajectories` there is no direct
measurement, and phase A blocks again. See DECISIONS [D-28].

Costs roughly $1 at n=30. Runs against the tuning split so the test set stays
untouched.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys

import config

log = logging.getLogger("caes.calibrate")

# Spread criteria. Deliberately explicit so the gate is arguable, not vibes.
MIN_STDEV = 0.12
MIN_RANGE = 0.40
MAX_SINGLE_BIN_SHARE = 0.60

# Trajectory criteria (phase B). dQ needs coverage to actually move.
#
# Measured on the RUNNING-MAX SMOOTHED series, because that is the series the
# gate consumes: CAESPolicy.decide calls smooth_coverage() and feeds the result
# to estimate_delta_q (METHODOLOGY 3.1). Judging the raw series would fail a
# verifier whose smoothed signal is perfectly usable -- coverage genuinely dips
# when a newly retrieved document introduces a second plausible entity, which
# is the whole reason smoothing exists. The raw series is still reported, and
# its volatility is a real finding worth watching (see the verdict text).
MIN_TOTAL_RISE = 0.05      # smoothed mean coverage must climb this much
MIN_MOVING_SHARE = 0.25    # fraction of queries whose coverage rises at all


def histogram(values: list[float], width: float = 0.1) -> dict[str, int]:
    bins: dict[str, int] = {}
    for lo in [round(i * width, 1) for i in range(int(1 / width))]:
        bins[f"{lo:.1f}-{lo + width:.1f}"] = 0
    for v in values:
        idx = min(int(v / width), int(1 / width) - 1)
        lo = round(idx * width, 1)
        bins[f"{lo:.1f}-{lo + width:.1f}"] += 1
    return bins


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--traj-n", type=int, default=15,
                    help="questions to run full trajectories for (phase B)")
    ap.add_argument("--no-trajectories", dest="trajectories",
                    action="store_false",
                    help="skip phase B (phase A alone cannot establish dQ)")
    args = ap.parse_args(argv)

    from agents.verifier import verify
    from graph import gold_recall
    from costs import TRACKER
    from retrieval import get_retriever
    from splits import tune_set

    questions = tune_set()[:args.n]
    if not questions:
        print("No questions available. Run `python ingest.py` first.",
              file=sys.stderr)
        return 2

    import llm as llm_mod

    retriever = get_retriever()
    results = []
    quota_spent = False

    try:
        with TRACKER.run_budget(args.max_usd, name="calibrate"):
            for i, q in enumerate(questions, 1):
                chunks = retriever.search(q["question"], k=args.k,
                                          query_id=q["id"], iteration=1,
                                          policy="calibrate")
                v = verify(q["question"], chunks, query_id=q["id"], iteration=1,
                           policy="calibrate")
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "gold": q["answer"],
                    # Separates "gate stopped early" from "retrieval missed it".
                    "gold_recall": gold_recall(chunks, q.get("supporting_titles")),
                    "coverage": v.coverage,
                    "missing": v.missing,
                    "confident": v.confident,
                    "parse_failed": v.parse_failed,
                })
                print(f"[{i:>3}/{len(questions)}] cov={v.coverage:.2f} "
                      f"{'PARSE-FAIL ' if v.parse_failed else ''}"
                      f"{q['question'][:70]}")

    except llm_mod.QuotaExhausted as exc:
        quota_spent = True
        print(f"{chr(10)}DAILY QUOTA SPENT after {len(results)}/"
              f"{len(questions)} questions: {exc}", file=sys.stderr)

    # ---------------- phase B: trajectories ----------------
    trajectories = []
    if not quota_spent and args.trajectories:
        from graph import run_query, state_summary
        from policies import FixedPolicy

        print()
        print(f"--- phase B: coverage trajectories over "
              f"{config.MAX_ITERATIONS} iterations ---")
        # FixedPolicy runs to a fixed depth regardless of the gate, which is
        # what we want: the trajectory must be observed to its end, not cut
        # short by the very rule being calibrated.
        pol = FixedPolicy(n=config.MAX_ITERATIONS)
        try:
            with TRACKER.run_budget(args.max_usd, name="calibrate-traj"):
                for i, q in enumerate(questions[:args.traj_n], 1):
                    final = run_query(q["question"], pol, query_id=q["id"],
                                      gold_titles=q.get("supporting_titles"))
                    summ = state_summary(final)
                    trajectories.append({
                        "id": q["id"],
                        "question": q["question"],
                        "coverage_history": summ["coverage_history"],
                        "gold_recall_history": summ["gold_recall_history"],
                        "new_chunks_history": summ.get("new_chunks_history", []),
                        "cost_history": summ["cost_history"],
                    })
                    cov = summ["coverage_history"]
                    print(f"[{i:>3}/{min(args.traj_n, len(questions))}] "
                          + " -> ".join(f"{c:.2f}" for c in cov)
                          + f"   {q['question'][:46]}")
        except llm_mod.QuotaExhausted as exc:
            quota_spent = True
            print(file=sys.stderr)
            print(f"DAILY QUOTA SPENT during trajectories after "
                  f"{len(trajectories)} questions: {exc}", file=sys.stderr)

    if trajectories:
        traj_out = config.RESULTS_DIR / "verifier_trajectories.jsonl"
        with traj_out.open("w", encoding="utf-8") as fh:
            for t in trajectories:
                fh.write(json.dumps(t) + chr(10))
        print(f"trajectories -> {traj_out}")

    out = config.RESULTS_DIR / "verifier_calibration.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    if quota_spent:
        # Judging the instrument on a truncated sample would be worse than not
        # judging it: the pass criteria are distributional, and a short sample
        # can pass or fail for reasons that have nothing to do with the rubric.
        print(f"Partial results saved to {out}. Calibration is NOT decided -- "
              f"re-run tomorrow to finish the sample.", file=sys.stderr)
        return 3

    coverages = [r["coverage"] for r in results]
    n_ok = sum(1 for r in results if not r["parse_failed"])
    stdev = statistics.pstdev(coverages) if len(coverages) > 1 else 0.0
    spread = max(coverages) - min(coverages)
    bins = histogram(coverages)
    top_bin, top_count = max(bins.items(), key=lambda kv: kv[1])
    top_share = top_count / len(coverages)

    print("\n" + "=" * 62)
    print(f"JSON validity : {n_ok}/{len(results)}")
    print(f"coverage      : mean {statistics.mean(coverages):.3f}  "
          f"stdev {stdev:.3f}  min {min(coverages):.2f}  max {max(coverages):.2f}")
    print("distribution  :")
    for label, count in bins.items():
        print(f"    {label}  {'#' * count}{'' if count else '.'} ({count})")
    print(f"largest bin   : {top_bin} holds {top_share:.0%}")
    recalls = [r["gold_recall"] for r in results if r["gold_recall"] >= 0]
    if recalls:
        full = sum(1 for r in recalls if r >= 1.0)
        print(f"gold recall   : mean {statistics.mean(recalls):.2f}  "
              f"all-gold-retrieved {full}/{len(recalls)} "
              f"({full / len(recalls):.0%} of questions need no 2nd retrieval)")
    print(f"spend         : ${TRACKER.cumulative():.4f} cumulative")
    print("=" * 62)

    # ---------------- phase B report ----------------
    traj_rise = None
    traj_moving = None
    if trajectories:
        depth = max(len(t["coverage_history"]) for t in trajectories)
        print()
        print("=" * 62)
        print(f"TRAJECTORIES  (n={len(trajectories)}, depth={depth})")
        print()
        def running_max(seq):
            out, best = [], float("-inf")
            for c in seq:
                best = max(best, c)
                out.append(best)
            return out

        print("mean coverage per iteration -- the diminishing-returns premise.")
        print("SMOOTHED is the series the gate differentiates; raw is shown "
              "for audit.")
        raw_means, sm_means = [], []
        for it in range(depth):
            rv = [t["coverage_history"][it] for t in trajectories
                  if len(t["coverage_history"]) > it]
            sv = [running_max(t["coverage_history"])[it] for t in trajectories
                  if len(t["coverage_history"]) > it]
            raw_means.append(statistics.mean(rv))
            sm_means.append(statistics.mean(sv))
            delta = ("" if it == 0
                     else f"   dQ_obs {sm_means[it] - sm_means[it - 1]:+.3f}")
            bar = "#" * int(round(sm_means[it] * 40))
            print(f"  it{it + 1}  smooth {sm_means[it]:.3f}  "
                  f"raw {raw_means[it]:.3f}  {bar}{delta}")
        traj_rise = sm_means[-1] - sm_means[0]
        raw_rise = raw_means[-1] - raw_means[0]
        print()
        print(f"  total rise it1 -> it{depth}:  smoothed {traj_rise:+.3f}   "
              f"raw {raw_rise:+.3f}")

        gains = [max(t["coverage_history"]) - t["coverage_history"][0]
                 for t in trajectories]
        n_moving = sum(1 for g in gains if g > 0.01)
        traj_moving = n_moving / len(gains)
        print(f"  queries whose coverage moves at all: "
              f"{n_moving}/{len(gains)} ({traj_moving:.0%})")

        # Raw volatility. Smoothing makes dQ usable, but big dips mean the
        # evidence set is being diluted, and the GENERATOR sees the raw diluted
        # set. METHODOLOGY 10 flags that smoothing biases dQ upward; this
        # quantifies it.
        steps = [b - a for t in trajectories
                 for a, b in zip(t["coverage_history"],
                                 t["coverage_history"][1:])]
        if steps:
            drops = [d for d in steps if d < -0.01]
            print(f"  raw volatility: mean |step| "
                  f"{statistics.mean(abs(d) for d in steps):.3f}, "
                  f"{len(drops)}/{len(steps)} steps drop, worst "
                  f"{min(steps):+.2f}")

        # Did the extra retrieval actually surface missing gold evidence?
        gr0 = [t["gold_recall_history"][0] for t in trajectories
               if t.get("gold_recall_history")]
        grN = [t["gold_recall_history"][-1] for t in trajectories
               if t.get("gold_recall_history")]
        if gr0 and gr0[0] >= 0:
            improved = sum(1 for t in trajectories
                           if max(t["gold_recall_history"])
                           > t["gold_recall_history"][0])
            print(f"  gold recall: {statistics.mean(gr0):.3f} -> "
                  f"{statistics.mean(grN):.3f}  "
                  f"({improved}/{len(trajectories)} queries improved)")

        dead = sum(1 for t in trajectories
                   for n in t.get("new_chunks_history", []) if n == 0)
        total_it = sum(len(t.get("new_chunks_history", [])) for t in trajectories)
        if total_it:
            print(f"  dead iterations (added no evidence): {dead}/{total_it}"
                  + ("   <-- see [D-27]" if dead else ""))

        print()
        print("sample trajectories (raw, then running-max smoothed):")
        for t in trajectories[:5]:
            raw = t["coverage_history"]
            sm, run = [], 0.0
            for c in raw:
                run = max(run, c)
                sm.append(run)
            print("  raw    " + " ".join(f"{c:.2f}" for c in raw)
                  + f"   {t['question'][:40]}")
            print("  smooth " + " ".join(f"{c:.2f}" for c in sm))

        # Gate overhead: what fraction of a query's spend is the verifier.
        by_type = TRACKER.summary()["by_call_type"]
        verify_usd = by_type.get("verify", {}).get("usd", 0.0)
        total_usd = sum(v["usd"] for v in by_type.values())
        if total_usd:
            print()
            print(f"gate overhead: verifier is {verify_usd / total_usd:.0%} of "
                  f"all spend (${verify_usd:.4f} of ${total_usd:.4f})")
        print("=" * 62)

    failures = []
    if n_ok < len(results):
        failures.append(f"{len(results) - n_ok} response(s) failed to parse")
    if stdev < MIN_STDEV:
        failures.append(f"stdev {stdev:.3f} < {MIN_STDEV}")
    if spread < MIN_RANGE:
        failures.append(f"range {spread:.2f} < {MIN_RANGE}")
    if top_share > MAX_SINGLE_BIN_SHARE:
        failures.append(f"{top_share:.0%} of scores in the single bin {top_bin}")

    traj_failures = []
    if trajectories:
        if traj_rise is not None and traj_rise < MIN_TOTAL_RISE:
            traj_failures.append(
                f"SMOOTHED mean coverage rises only {traj_rise:+.3f} across "
                f"iterations (need {MIN_TOTAL_RISE:+.2f}) -- dQ has nothing to "
                f"extrapolate")
        if traj_moving is not None and traj_moving < MIN_MOVING_SHARE:
            traj_failures.append(
                f"only {traj_moving:.0%} of queries move at all "
                f"(need {MIN_MOVING_SHARE:.0%})")

    print()
    if not failures and not traj_failures:
        print("PASS. Coverage is spread, it rises with retrieval, and parsing "
              "is clean; proceed to Phase 3.")
        return 0

    if traj_failures:
        print("BLOCKED on TRAJECTORIES. Coverage does not respond to further "
              "retrieval:")
        for f in traj_failures:
            print(f"  - {f}")
        print()
        print("This is the fatal one. If more retrieval does not raise "
              "coverage, there")
        print("is no marginal quality to trade against cost, and the gate has "
              "nothing")
        print("to decide. Check the gold_recall line first: if retrieval is "
              "already")
        print("saturated at iteration 1, the corpus is too easy and the rubric "
              "is NOT")
        print("the problem -- see DECISIONS [D-25].")
        return 1

    # The marginal is a WARNING when trajectories were measured, and a BLOCKER
    # when they were not. It is a proxy for "is the verifier informative?"; the
    # trajectory is the direct measurement of the same thing, on the series the
    # gate actually differentiates. A proxy that contradicts a direct
    # measurement is not evidence -- but with no direct measurement available,
    # the proxy is all there is. See DECISIONS [D-28].
    if not trajectories:
        print("BLOCKED on the ITERATION-1 MARGINAL:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Trajectories were NOT measured, so this verdict rests on a "
              "marginal")
        print("distribution alone, which cannot establish dQ. Re-run without "
              "--no-trajectories")
        print("before acting on it.")
        return 1

    print("PASS (with warnings). The verifier is a usable dQ signal.")
    print()
    print("Trajectory evidence -- the direct test, on the series the gate "
          "differentiates:")
    print(f"  smoothed coverage rises {traj_rise:+.3f} across iterations "
          f"(need {MIN_TOTAL_RISE:+.2f})")
    print(f"  {traj_moving:.0%} of queries move (need "
          f"{MIN_MOVING_SHARE:.0%})")
    print()
    print("WARNINGS on the iteration-1 marginal:")
    for f in failures:
        print(f"  - {f}")
    print()
    print("This measures how hard the TASK is, not how blunt the verifier is.")
    print("Check the gold_recall line: questions whose supporting passages are")
    print("all retrieved on the first try are correctly scored 1.0, and")
    print("sharpening a rubric that is already right damages a working")
    print("instrument (DECISIONS [D-25], [D-28]).")
    print()
    print("Carry forward to Phase 3: a top-heavy marginal predicts a")
    print("depth-heavy ITERATION HISTOGRAM. If the recommended lambda puts most")
    print("queries in one bucket, that is [D-21]'s degenerate spread and it")
    print("threatens the paper's central figure -- check it there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
