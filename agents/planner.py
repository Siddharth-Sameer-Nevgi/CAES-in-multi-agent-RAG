"""Query planner.

Iteration 1 returns the question unchanged — spending an LLM call to rewrite a
question we have not yet searched is pure overhead. Later iterations ask Haiku
for a focused sub-query targeting what the verifier said was missing.
"""
from __future__ import annotations

import logging

from agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM

log = logging.getLogger("caes.planner")

MAX_PLAN_TOKENS = 100


def plan(
    question: str,
    prior_evidence: list | None = None,
    missing: str | None = None,
    *,
    query_id: str = "",
    iteration: int = 1,
    policy: str = "",
) -> str:
    """Return the retrieval query for this iteration."""
    prior_evidence = prior_evidence or []

    # Iteration 1 (or nothing to go on): use the question verbatim, no LLM call.
    if iteration <= 1 or not prior_evidence or not missing \
            or missing.strip().lower() in ("", "nothing", "none"):
        return question

    import llm

    titles = ", ".join(sorted({
        getattr(c, "title", None) or c.get("title", "")
        for c in prior_evidence
    })) or "(none)"

    prompt = PLANNER_PROMPT.format(
        question=question, titles=titles[:1000], missing=missing)
    resp = llm.invoke_llm(
        prompt, call_type="plan", query_id=query_id, iteration=iteration,
        max_tokens=MAX_PLAN_TOKENS, system=PLANNER_SYSTEM, policy=policy,
    )
    sub_query = resp.text.strip().strip('"').strip()
    if not sub_query:
        log.warning("[%s] planner returned empty text; falling back to the "
                    "missing-fact phrase", query_id)
        return f"{question} {missing}".strip()
    return sub_query
