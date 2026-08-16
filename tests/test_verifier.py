"""Phase 2: the verifier's defensive parse ladder, and answer metrics.

One unparsed verifier response is one corrupted point on the coverage curve,
which is one corrupted dQ. These paths matter.
"""
from __future__ import annotations

import pytest

import bedrock
from agents import verifier
from agents.verifier import Verification, truncate_evidence, verify
from metrics import exact_match, f1, is_abstention, score


class FakeLLM:
    """Returns a scripted sequence of texts, recording how many calls happened."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        text = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        return bedrock.LLMResponse(text=text, input_tokens=10, output_tokens=5,
                                   latency_ms=1.0, cached=False, usd=0.0,
                                   notional_usd=0.0)


@pytest.fixture()
def llm(monkeypatch):
    def install(*texts):
        fake = FakeLLM(*texts)
        monkeypatch.setattr(bedrock, "invoke_llm", fake)
        return fake
    return install


# --- parsing --------------------------------------------------------------

def test_clean_json_parses(llm):
    llm('{"coverage": 0.42, "missing": "the director", "confident": false}')
    v = verify("q", [], query_id="q1")
    assert v.coverage == pytest.approx(0.42)
    assert v.missing == "the director"
    assert v.confident is False
    assert v.parse_failed is False


def test_markdown_fences_are_stripped(llm):
    llm('```json\n{"coverage": 0.8, "missing": "nothing", "confident": true}\n```')
    v = verify("q", [], query_id="q1")
    assert v.coverage == pytest.approx(0.8)
    assert v.parse_failed is False


def test_json_embedded_in_prose_is_recovered(llm):
    llm('Here is my assessment:\n'
        '{"coverage": 0.6, "missing": "the year", "confident": false}\n'
        'Hope that helps!')
    v = verify("q", [], query_id="q1")
    assert v.coverage == pytest.approx(0.6)


def test_repair_retry_happens_once_and_can_succeed(llm):
    fake = llm("total nonsense, no json here",
               '{"coverage": 0.55, "missing": "x", "confident": false}')
    v = verify("q", [], query_id="q1")
    assert fake.calls == 2, "expected exactly one repair attempt"
    assert v.coverage == pytest.approx(0.55)
    assert v.parse_failed is False


def test_double_failure_holds_previous_coverage(llm):
    """A parse failure must not read as progress."""
    fake = llm("nonsense", "still nonsense")
    v = verify("q", [], query_id="q1", previous_coverage=0.63)
    assert fake.calls == 2
    assert v.coverage == pytest.approx(0.63)
    assert v.parse_failed is True


def test_coverage_is_clamped_to_the_unit_interval(llm):
    llm('{"coverage": 1.7, "missing": "", "confident": true}')
    assert verify("q", [], query_id="q1").coverage == 1.0
    llm('{"coverage": -0.4, "missing": "", "confident": false}')
    assert verify("q", [], query_id="q1").coverage == 0.0


def test_non_numeric_coverage_triggers_repair(llm):
    fake = llm('{"coverage": "high", "missing": "x", "confident": true}',
               '{"coverage": 0.9, "missing": "nothing", "confident": true}')
    v = verify("q", [], query_id="q1")
    assert fake.calls == 2
    assert v.coverage == pytest.approx(0.9)


# --- evidence truncation --------------------------------------------------

def test_evidence_is_truncated_per_chunk():
    long_chunk = {"text": "word " * 500}
    out = truncate_evidence([long_chunk], max_chars=100)
    assert len(out) < 200
    assert out.endswith("...")


def test_empty_evidence_is_explicit():
    assert "no evidence" in truncate_evidence([])


# --- metrics --------------------------------------------------------------

def test_exact_match_normalizes_articles_case_and_punctuation():
    assert exact_match("The Beatles.", "beatles") == 1.0
    assert exact_match("Beatles", "Rolling Stones") == 0.0


def test_f1_partial_credit():
    assert f1("Jan de Bont", "Jan de Bont") == pytest.approx(1.0)
    assert 0.0 < f1("Jan de Bont", "de Bont") < 1.0
    assert f1("completely wrong", "Jan de Bont") == 0.0


def test_yes_no_questions_require_exact_agreement():
    assert f1("yes", "yes") == 1.0
    assert f1("no", "yes") == 0.0
    assert f1("yes it is", "yes") == 0.0


def test_abstention_scores_zero_and_is_flagged():
    assert is_abstention("insufficient evidence")
    s = score("insufficient evidence", "Jan de Bont")
    assert s["exact_match"] == 0.0 and s["f1"] == 0.0 and s["abstained"] == 1.0
