"""Free end-to-end smoke test of the full graph.

    DRY_RUN=1 python -m smoke

Catches wiring bugs for $0.00 before any billable call is made. Builds a tiny
synthetic index if the real one is absent, so it works on a clean checkout and
never touches data/index.faiss.

Asserts on exit that cumulative ledger spend did not move.
"""
from __future__ import annotations

import json
import logging
import sys

import numpy as np

import config

SMOKE_INDEX = config.DATA_DIR / "smoke_index.faiss"
SMOKE_CHUNKS = config.DATA_DIR / "smoke_chunks.jsonl"

_SYNTHETIC = [
    ("Speed (1994 film)", "Speed is a 1994 American action thriller film "
     "directed by Jan de Bont in his directorial debut. It stars Keanu Reeves "
     "and Sandra Bullock."),
    ("Jan de Bont", "Jan de Bont is a Dutch filmmaker and cinematographer. He "
     "directed Speed in 1994 and the disaster film Twister in 1996."),
    ("Twister (1996 film)", "Twister is a 1996 American disaster film directed "
     "by Jan de Bont and starring Helen Hunt and Bill Paxton."),
    ("Inception", "Inception is a 2010 science fiction action film written and "
     "directed by Christopher Nolan."),
    ("Christopher Nolan", "Christopher Nolan is a British-American filmmaker "
     "known for Memento, The Dark Knight, Inception and Interstellar."),
    ("Memento (film)", "Memento is a 2000 psychological thriller film directed "
     "by Christopher Nolan, based on a story by his brother Jonathan Nolan."),
    ("Helen Hunt", "Helen Hunt is an American actress and director. She won an "
     "Academy Award for As Good as It Gets in 1997."),
    ("Keanu Reeves", "Keanu Reeves is a Canadian actor known for The Matrix, "
     "John Wick and Speed."),
]

QUESTIONS = [
    ("smoke-1", "The director of the 1994 film Speed also directed which 1996 "
                "disaster film?"),
    ("smoke-2", "Who directed Inception?"),
    ("smoke-3", "Which actress starred in the 1996 film directed by Jan de "
                "Bont, and what award did she win?"),
]


def build_synthetic_index() -> None:
    import faiss

    import llm

    chunks = [{"chunk_id": str(i), "title": t, "part": 0, "text": f"{t}: {b}"}
              for i, (t, b) in enumerate(_SYNTHETIC)]
    vecs = llm.embed([c["text"] for c in chunks], policy="smoke")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))
    faiss.write_index(index, str(SMOKE_INDEX))
    with SMOKE_CHUNKS.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    import llm
    import retrieval
    from cache import CACHE
    from costs import TRACKER
    from policies import CAESPolicy, FixedPolicy, OneShotPolicy

    if not llm.DRY_RUN:
        print("Refusing to run: smoke is a $0.00 check. Set DRY_RUN=1.",
              file=sys.stderr)
        return 2

    spend_before = TRACKER.cumulative()

    real = None
    if config.INDEX_PATH.exists():
        try:
            real = retrieval.Retriever()
            print(f"Using the real index at {config.INDEX_PATH}")
        except (ValueError, FileNotFoundError) as exc:
            # e.g. an index built by the previous provider's embedding model.
            print(f"Ignoring the index at {config.INDEX_PATH}: {exc}")
    if real is not None:
        retrieval._RETRIEVER = real
    else:
        print(f"Building a tiny synthetic index at {config.EMBED_DIM} dims "
              f"(free).")
        build_synthetic_index()
        retrieval._RETRIEVER = retrieval.Retriever(SMOKE_INDEX, SMOKE_CHUNKS)

    from graph import run_query, state_summary

    policies = [
        ("fixed(n=3)", FixedPolicy(n=3)),
        ("oneshot", OneShotPolicy()),
        # LAMBDA is unset before tuning, so pin an illustrative value here.
        # This is a wiring check, not a measurement.
        ("caes(lam=200)", CAESPolicy(lam=200.0, record=False)),
    ]

    print()
    header = f"{'policy':<16} {'query':<9} {'iters':>5} {'stop':<10} " \
             f"{'cov':>5} {'notional$':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, policy in policies:
        for qid, question in QUESTIONS:
            final = run_query(question, policy, query_id=f"{policy.name}-{qid}")
            s = state_summary(final)
            rows.append((label, s))
            print(f"{label:<16} {qid:<9} {s['iterations_used']:>5} "
                  f"{s['stop_reason']:<10} {s['final_coverage']:>5.2f} "
                  f"{s['total_usd']:>10.5f}")

    print()
    for label, s in rows[:1]:
        print(f"sample answer ({label}): {s['answer'][:100]}")

    iters = {label: sorted({s["iterations_used"] for lbl, s in rows if lbl == label})
             for label, _ in policies}
    print()
    for label, seen in iters.items():
        print(f"iteration counts for {label}: {seen}")

    CACHE.log_stats("smoke cache")
    t = llm.totals()
    print(f"llm calls={t['calls']} cache_hits={t['cache_hits']} "
          f"notional=${t['notional_usd']:.5f} actual=${t['actual_usd']:.5f}")

    spend_after = TRACKER.cumulative()
    delta = spend_after - spend_before
    print(f"\nledger spend delta: ${delta:.6f}")
    if delta != 0.0:
        print("FAIL: DRY_RUN recorded real spend.", file=sys.stderr)
        return 1
    print("OK: full graph exercised for $0.00")
    return 0


if __name__ == "__main__":
    sys.exit(main())
