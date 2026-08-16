"""Deterministic, disjoint question splits.

Tuning lambda on the test set would invalidate the result, so the split is
computed once from a fixed seed and asserted disjoint every time it is loaded.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config

QUESTIONS_PATH = config.DATA_DIR / "questions.jsonl"


def load_questions(path: Path | str = QUESTIONS_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python ingest.py` first."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_splits(
    questions: list[dict] | None = None,
    n_tune: int = config.N_TUNE,
    n_test: int = config.N_TEST,
    seed: int = config.SPLIT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Return (tune, test). Guaranteed disjoint and stable across runs."""
    if questions is None:
        questions = load_questions()
    ordered = sorted(questions, key=lambda q: q["id"])
    rng = random.Random(seed)
    rng.shuffle(ordered)

    if len(ordered) < n_tune + n_test:
        raise ValueError(
            f"Need at least {n_tune + n_test} questions for disjoint splits, "
            f"have {len(ordered)}."
        )

    tune = ordered[:n_tune]
    test = ordered[n_tune:n_tune + n_test]

    tune_ids, test_ids = {q["id"] for q in tune}, {q["id"] for q in test}
    assert not (tune_ids & test_ids), "tune/test splits overlap"
    return tune, test


def tune_set() -> list[dict]:
    return make_splits()[0]


def test_set() -> list[dict]:
    return make_splits()[1]


if __name__ == "__main__":
    tune, test = make_splits()
    print(f"tune: {len(tune)} questions  (first id {tune[0]['id']})")
    print(f"test: {len(test)} questions  (first id {test[0]['id']})")
    print("disjoint: True")
