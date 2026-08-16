"""All prompts live here as named constants so they can be tuned without
touching orchestration logic.

The verifier prompt is the most important text in this repository: its coverage
scores are the dQ signal that the CAES gate differentiates. An underspecified
rubric produces noise and the whole method fails.
"""

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = (
    "You rewrite multi-hop questions into a single focused retrieval query. "
    "You output only the query text, with no preamble, quotes, or explanation."
)

PLANNER_PROMPT = """\
Original question:
{question}

Evidence already retrieved (titles only):
{titles}

What is still missing:
{missing}

Write ONE search query, at most 20 words, that would retrieve the missing
information. Target the specific entity, relation, or attribute named in the
"missing" field. Do not restate the whole original question — that has already
been searched and returned the evidence above.

Query:"""


# ---------------------------------------------------------------------------
# Verifier  --  CRITICAL COMPONENT
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM = (
    "You are a strict evidence-sufficiency judge. You respond with a single "
    "JSON object and nothing else: no prose, no explanation, no markdown code "
    "fences."
)

VERIFIER_PROMPT = """\
Judge whether the EVIDENCE below is sufficient to answer the QUESTION.

QUESTION:
{question}

EVIDENCE:
{evidence}

Score `coverage` on this rubric. Use the whole range — most real evidence sets
are partial, and scores clustered in a narrow band are useless.

  0.0 - 0.3  Evidence barely touches the question. Key entities named in the
             question are absent from the evidence, or appear only in passing
             with none of the required facts attached.
  0.4 - 0.6  Partial. One hop is answered but another is unaddressed; or the
             right entities are present but the specific relation, date,
             number, or attribute the question asks for is missing.
  0.7 - 0.8  Most required facts are present. A minor gap, an ambiguity between
             two candidates, or one unstated link remains.
  0.9 - 1.0  Fully answerable from the evidence alone. Every entity, relation,
             and value the question needs is explicitly stated.

Rules:
  - Judge only what is written in the evidence. Do not use your own knowledge
    of the answer. If you know the answer but the evidence does not state it,
    coverage is LOW.
  - A multi-hop question needs every hop supported. If hop two is missing,
    coverage cannot exceed 0.6 no matter how good hop one is.
  - `missing` must name the specific fact still needed, in under 20 words, so
    it can be used as a search query. If nothing is missing, write "nothing".
  - `confident` is true only when you would stake the answer on this evidence.

WORKED EXAMPLE A (low coverage)
Question: "The director of the 1994 film Speed also directed which 1996 film?"
Evidence: "Speed is a 1994 American action thriller film. It stars Keanu Reeves
and Dennis Hopper. The film was a commercial success, grossing $350 million."
Correct output:
{{"coverage": 0.2, "missing": "the name of the director of Speed (1994)", "confident": false}}
Reasoning (do not output this): the evidence never names the director, so hop
one fails and hop two cannot even be attempted.

WORKED EXAMPLE B (high coverage)
Question: "The director of the 1994 film Speed also directed which 1996 film?"
Evidence: "Speed is a 1994 American action thriller film directed by Jan de
Bont. | Jan de Bont is a Dutch filmmaker. He directed Speed (1994) and the
disaster film Twister (1996)."
Correct output:
{{"coverage": 0.95, "missing": "nothing", "confident": true}}
Reasoning (do not output this): both hops are explicitly stated in the evidence.

Now judge the QUESTION and EVIDENCE above. Respond with exactly this JSON shape
and nothing else:
{{"coverage": <float 0.0-1.0>, "missing": "<short phrase>", "confident": <true|false>}}"""


VERIFIER_REPAIR_PROMPT = """\
Your previous response was not valid JSON. You returned:

{bad_output}

Return ONLY the JSON object, with no code fence and no other text:
{{"coverage": <float 0.0-1.0>, "missing": "<short phrase>", "confident": <true|false>}}"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = (
    "You answer questions strictly from provided evidence. You never speculate "
    "and you never draw on outside knowledge."
)

GENERATOR_PROMPT = """\
Answer the question using ONLY the evidence below.

QUESTION:
{question}

EVIDENCE:
{evidence}

Rules:
  - Answer from the evidence alone. Do not use outside knowledge.
  - If the evidence does not contain the answer, reply exactly:
    insufficient evidence
  - Be direct. Give the shortest complete answer — usually a name, date, number,
    or short phrase. Do not restate the question and do not explain your
    reasoning.

Answer:"""
