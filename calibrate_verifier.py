"""Phase 2 acceptance gate: is the verifier a usable dQ signal?

    python calibrate_verifier.py --n 30

Two blockers, both checked here:
  1. Valid JSON on every sampled question (target 30/30).
  2. Coverage scores spread across the range. If everything lands in a narrow
     band the verifier carries no information, dQ is noise, and the whole
     method fails. Sharpen the rubric in agents/prompts.py before proceeding.

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

    failures = []
    if n_ok < len(results):
        failures.append(f"{len(results) - n_ok} response(s) failed to parse")
    if stdev < MIN_STDEV:
        failures.append(f"stdev {stdev:.3f} < {MIN_STDEV}")
    if spread < MIN_RANGE:
        failures.append(f"range {spread:.2f} < {MIN_RANGE}")
    if top_share > MAX_SINGLE_BIN_SHARE:
        failures.append(f"{top_share:.0%} of scores in the single bin {top_bin}")

    if failures:
        print("\nBLOCKED. The verifier is not yet a usable dQ signal:")
        for f in failures:
            print(f"  - {f}")
        print("\nSharpen the rubric in agents/prompts.py (VERIFIER_PROMPT) and "
              "re-run.\nDo not proceed to Phase 3 on a flat coverage signal.")
        return 1

    print("\nPASS. Coverage is spread and parsing is clean; proceed to Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
