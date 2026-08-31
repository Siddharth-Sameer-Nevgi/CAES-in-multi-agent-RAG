"""Grounded answer generation.

Instructed to answer from evidence only and to say "insufficient evidence"
rather than speculate, so that a stopped-too-early gate shows up as an honest
abstention in the results rather than as a confident hallucination.
"""
from __future__ import annotations

import logging

from agents.prompts import GENERATOR_PROMPT, GENERATOR_SYSTEM

log = logging.getLogger("caes.generator")

MAX_ANSWER_TOKENS = 256

# The generator sees more of each chunk than the verifier does; it has to
# produce the answer, not just judge whether one is derivable.
GENERATOR_CHUNK_CHARS = 1200


def format_evidence(chunks: list, max_chars: int = GENERATOR_CHUNK_CHARS) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = getattr(c, "text", None) or c.get("text", "")
        text = text.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        parts.append(f"[{i}] {text}")
    return "\n".join(parts) if parts else "(no evidence retrieved)"


def generate(
    question: str,
    evidence: list,
    *,
    query_id: str = "",
    iteration: int = 0,
    policy: str = "",
) -> str:
    import llm

    prompt = GENERATOR_PROMPT.format(
        question=question, evidence=format_evidence(evidence))
    resp = llm.invoke_llm(
        prompt, call_type="generate", query_id=query_id, iteration=iteration,
        max_tokens=MAX_ANSWER_TOKENS, system=GENERATOR_SYSTEM, policy=policy,
    )
    answer = resp.text.strip()
    if not answer:
        log.warning("[%s] generator returned empty text", query_id)
        return "insufficient evidence"
    return answer
