"""HotpotQA answer metrics: SQuAD-style exact match and token F1."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # yes/no questions: token overlap is meaningless, require an exact match.
    if gold_tokens in (["yes"], ["no"]) or pred_tokens in (["yes"], ["no"]):
        return float(pred_tokens == gold_tokens)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


ABSTENTION = "insufficient evidence"


def is_abstention(prediction: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ABSTENTION)


def score(prediction: str, ground_truth: str) -> dict[str, float]:
    """EM and F1, with abstentions scored as zero rather than dropped.

    Scoring an abstention as zero is deliberate: a policy that stops too early
    and honestly says so should be penalised for it, otherwise the cost saving
    would look free.
    """
    if is_abstention(prediction):
        return {"exact_match": 0.0, "f1": 0.0, "abstained": 1.0}
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "f1": f1(prediction, ground_truth),
        "abstained": 0.0,
    }
