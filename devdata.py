"""Generate a synthetic corpus + question set so the whole pipeline can be
exercised end to end for $0.00.

    DRY_RUN=1 python devdata.py --n 260

This is a development scaffold, NOT the experiment. It writes the same files
`ingest.py` writes, and stamps meta.json with "synthetic": true so downstream
tools can tell the difference. Delete data/ and run `python ingest.py` for the
real thing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np

import config

SUBJECTS = ["Aldridge", "Brookfield", "Calloway", "Danforth", "Ellsworth",
            "Fairbairn", "Grimsby", "Harcourt", "Illingworth", "Jessop",
            "Kirkbride", "Langmere", "Moresby", "Netherby", "Oakhurst",
            "Penhaligon", "Quillon", "Ravensworth", "Stanbury", "Thorncroft"]
ROLES = ["novelist", "architect", "cartographer", "botanist", "composer",
         "geologist", "lexicographer", "shipwright"]
PLACES = ["Ardwick", "Belhaven", "Corbridge", "Dunmore", "Eastleigh",
          "Fenwick", "Glenmore", "Harewood"]


def build(n_questions: int, seed: int):
    rng = random.Random(seed)
    passages, questions = [], []

    for i in range(n_questions):
        a, b = rng.sample(SUBJECTS, 2)
        role = rng.choice(ROLES)
        place = rng.choice(PLACES)
        year = rng.randint(1840, 1990)

        passages.append({
            "title": f"{a} ({i})",
            "text": (f"{a} was a {role} active in the nineteenth century. "
                     f"{a} collaborated extensively with {b} and is chiefly "
                     f"remembered for work carried out at {place}."),
        })
        passages.append({
            "title": f"{b} ({i})",
            "text": (f"{b} was born in {place} in {year}. {b} later moved away "
                     f"and worked alongside several contemporaries."),
        })
        questions.append({
            "id": f"syn-{i:05d}",
            "question": (f"The {role} who collaborated with {b} is remembered "
                         f"for work at which place, and in what year was {b} "
                         f"born?"),
            "answer": f"{place} {year}",
            "type": "bridge",
            "level": rng.choice(["easy", "medium", "hard"]),
            "supporting_titles": [f"{a} ({i})", f"{b} ({i})"],
        })

    # Distractors, so retrieval has something to get wrong.
    for j in range(n_questions):
        passages.append({
            "title": f"Distractor {j}",
            "text": (f"{rng.choice(SUBJECTS)} published a minor treatise on "
                     f"{rng.choice(ROLES)} practice in {rng.choice(PLACES)}."),
        })
    return questions, passages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_TUNE + config.N_TEST + 60)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import llm
    if not llm.DRY_RUN:
        print("Refusing to run outside DRY_RUN: this would spend real money on "
              "fake data. Set DRY_RUN=1.", file=sys.stderr)
        return 2

    if config.INDEX_PATH.exists() and not args.force:
        print(f"{config.INDEX_PATH} exists. Pass --force to overwrite "
              f"(this destroys a real index if you built one).", file=sys.stderr)
        return 2

    import faiss

    from ingest import build_chunks

    questions, passages = build(args.n, config.SPLIT_SEED)
    chunks = build_chunks(passages)
    vecs = llm.embed([c["text"] for c in chunks], policy="devdata")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))

    faiss.write_index(index, str(config.INDEX_PATH))
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    with (config.DATA_DIR / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q) + "\n")
    config.META_PATH.write_text(json.dumps({
        "synthetic": True,
        "n_questions": len(questions),
        "n_passages": len(passages),
        "n_chunks": len(chunks),
        "embed_dim": config.EMBED_DIM,
        "note": "SYNTHETIC development scaffold. Not the HotpotQA experiment.",
    }, indent=2), encoding="utf-8")

    print(f"SYNTHETIC dataset written: {len(questions)} questions, "
          f"{len(chunks)} chunks. meta.json is stamped synthetic:true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
