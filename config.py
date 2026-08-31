"""Central configuration for CAES-RAG.

Every tunable lives here. Nothing else in the codebase should hardcode a
price, a model id, or a loop bound.

Two model providers are supported. `PROVIDER` selects one; everything
downstream reads the provider-neutral names (`MODEL_LLM`, `MODEL_EMBED`,
`EMBED_DIM`, `PRICE_LLM_*`, `PRICE_EMBED_PER_1K`) resolved at the bottom of
this file. Only request/response shaping and price constants differ between
providers -- see DECISIONS [D-22].
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
# "gemini"  -- Google Generative Language API (default).
# "bedrock" -- AWS Bedrock. Kept fully working: model invocation is blocked
#              account-wide on the project's AWS account (a support case is
#              open), but if it clears, a two-provider robustness result is
#              worth having. See DECISIONS [D-22].
PROVIDER = os.environ.get("CAES_PROVIDER", "gemini").strip().lower()
if PROVIDER not in ("gemini", "bedrock"):
    raise ValueError(
        f"CAES_PROVIDER={PROVIDER!r} is not a known provider. "
        f"Use 'gemini' or 'bedrock'."
    )

# ---------------------------------------------------------------------------
# Pricing (USD per 1K tokens)
# ---------------------------------------------------------------------------
# Every dC number in the paper derives from these floats. Each carries its
# source and the date it was confirmed. Notional cost is computed from list
# price whether or not the call was billed -- on the Gemini free tier nothing
# is billed at all, which is precisely why the notional machinery matters here.
# See DECISIONS [D-12] and [D-22].

# --- Bedrock, us-east-1 on-demand. VERIFIED 2026-08-16.
#     The AWS pricing page renders its tables in JS, so confirm in the Bedrock
#     console (Model access -> pricing) rather than by scraping it. ---
BEDROCK_PRICE_LLM_INPUT_PER_1K  = 0.001    # $1.00 / 1M  Claude Haiku 4.5
BEDROCK_PRICE_LLM_OUTPUT_PER_1K = 0.005    # $5.00 / 1M  Claude Haiku 4.5

# CORRECTED 2026-08-16: was 0.00011 ($0.11/1M), which is roughly the price of
# the PREVIOUS generation -- Titan Embeddings G1 / v1 bill at $0.10/1M. This
# project uses titan-embed-text-v2:0, which is ~80% cheaper at $0.02/1M.
# Corroborated by three independent trackers; see CHANGELOG [0.1.1] Task B.
BEDROCK_PRICE_EMBED_PER_1K      = 0.00002  # $0.02 / 1M  (v2, NOT v1's $0.10)

# --- Gemini. VERIFIED 2026-08-31 against
#     https://ai.google.dev/gemini-api/docs/pricing (paid tier, Standard row).
#     The free tier bills nothing for either model ("Free of charge"), so these
#     are LIST prices, used for notional accounting only; actual spend is
#     $0.00. Batch pricing reads "Not available" on the free tier; unused. ---
GEMINI_PRICE_LLM_INPUT_PER_1K   = 0.0003   # $0.30 / 1M  gemini-2.5-flash (text)
GEMINI_PRICE_LLM_OUTPUT_PER_1K  = 0.0025   # $2.50 / 1M  gemini-2.5-flash
GEMINI_PRICE_EMBED_PER_1K       = 0.00015  # $0.15 / 1M  gemini-embedding-001

# ---------------------------------------------------------------------------
# Model ids and embedding geometry
# ---------------------------------------------------------------------------
BEDROCK_MODEL_LLM   = "anthropic.claude-haiku-4-5"
BEDROCK_MODEL_EMBED = "amazon.titan-embed-text-v2:0"
BEDROCK_EMBED_DIM   = 1024      # titan-embed-text-v2 default output dimension

# gemini-2.0-flash and text-embedding-004 -- named in the original migration
# plan -- were both retired before this migration landed (2.0 Flash is listed
# under "Previous models / Shut down"; text-embedding-004 is gone entirely).
# These are their nearest live equivalents in the same tier. Confirmed against
# https://ai.google.dev/gemini-api/docs/models on 2026-08-31.
GEMINI_MODEL_LLM    = "gemini-2.5-flash"
GEMINI_MODEL_EMBED  = "gemini-embedding-001"

# gemini-embedding-001 returns 3072 dimensions by default and supports
# Matryoshka truncation to 128..3072, with 768 / 1536 / 3072 recommended.
# 768 is chosen deliberately: a flat FAISS index over ~20k chunks costs ~60 MB
# at 768 dims against ~250 MB at 3072, and the intended host is an EC2 t3.micro
# (1 GB RAM). Truncated vectors are NOT unit-norm as returned, so llm.embed
# re-normalises -- required for IndexFlatIP to mean cosine similarity.
GEMINI_EMBED_DIM    = 768

# ---------------------------------------------------------------------------
# Gemini transport
# ---------------------------------------------------------------------------
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# The API key is read from the environment at call time and never stored here.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_TIMEOUT_S = 120

# gemini-2.5-flash has thinking ON by default, and thinking tokens bill as
# output. Left on they would inflate dC, and their variable length undermines
# the determinism [D-19] relies on for cache correctness. Disabled explicitly.
# thoughtsTokenCount is still folded into the output count if it ever comes
# back non-zero, so an ignored budget surfaces as cost rather than as an
# understated dC.
GEMINI_THINKING_BUDGET = 0

# Client-side pacer, per model class. Limits are per-account and visible only
# at aistudio.google.com/rate-limit, so these mirror one account's observed
# quota rather than a documented universal number. 0 disables pacing; 429s are
# retried with the server-supplied retryDelay regardless.
#
# READ FROM THE DASHBOARD 2026-08-31, free tier:
#   gemini-2.5-flash      5 RPM / 250K TPM /    20 RPD
#   gemini-embedding-001  100 RPM / 30K TPM / 1,000 RPD
# The single shared value this replaced was 15 -- three times the LLM's actual
# limit, and a sixth of the embedding model's. Paced separately now.
#
# RPD, not RPM, is the binding constraint on the free tier: 20 LLM requests a
# day is roughly three questions. See DECISIONS [D-24].
GEMINI_LLM_RPM   = 5
GEMINI_EMBED_RPM = 100

# Daily request ceilings, for pre-flight feasibility warnings only. These are
# NOT enforced -- the server enforces them; we surface them so a run that
# cannot possibly finish says so before it starts. 0 means "no known limit".
GEMINI_LLM_RPD   = 20
GEMINI_EMBED_RPD = 1000

# gemini-embedding-001's :embedContent response carries no token count, so the
# count is measured with a separate :countTokens call. See DECISIONS [D-23].
#   "measured"  -- one extra countTokens call per uncached text (invariant 4)
#   "estimated" -- character-ratio estimate; violates invariant 4, so every
#                  affected run is stamped and warned about. Opt in knowingly.
EMBED_TOKENS_MODE = os.environ.get("CAES_EMBED_TOKENS", "measured").strip().lower()
if EMBED_TOKENS_MODE not in ("measured", "estimated"):
    raise ValueError(
        f"CAES_EMBED_TOKENS={EMBED_TOKENS_MODE!r} must be 'measured' or 'estimated'."
    )

AWS_REGION  = "us-east-1"

# --- Hard spend guards ---
HARD_BUDGET_USD        = 40.00   # cumulative ceiling; raises BudgetExceeded
WARN_BUDGET_USD        = 25.00   # logs a loud warning
SINGLE_RUN_MAX_USD     = 5.00    # ceiling for one experiment invocation

# --- Loop guards ---
MAX_ITERATIONS = 5    # absolute hard cap; a gate bug cannot exceed this
MIN_ITERATIONS = 1

# --- CAES parameters (LAMBDA is set by Phase 5 tuning; do not guess) ---
LAMBDA = None          # must be set explicitly before CAES runs
DECAY_FACTOR = 0.6     # dQ extrapolation

# --- Retrieval ---
TOP_K = 5
CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 30
EMBED_BATCH = 50

# --- Verifier budget ---
VERIFIER_CHUNK_CHARS = 600    # ~150 tokens per chunk of truncated evidence

# --- CloudWatch (optional; off unless --cloudwatch is passed) ---
CLOUDWATCH_NAMESPACE = "CAES-RAG"

# --- Dataset splits (deterministic, disjoint) ---
CORPUS_SAMPLE_SIZE = 2000
SPLIT_SEED   = 20240917
N_TUNE       = 50     # held-out lambda-tuning questions
N_TEST       = 150    # evaluation questions, disjoint from tuning

# ---------------------------------------------------------------------------
# Provider-neutral names. Everything outside config.py and llm.py reads these.
# ---------------------------------------------------------------------------

def provider_settings(provider: str) -> dict:
    """The full set of names that vary by provider.

    Exposed as a function so tests can flip providers by applying this dict,
    rather than re-deriving the mapping and drifting from it.
    """
    if provider == "gemini":
        return {
            "MODEL_LLM":               GEMINI_MODEL_LLM,
            "MODEL_EMBED":             GEMINI_MODEL_EMBED,
            "EMBED_DIM":               GEMINI_EMBED_DIM,
            "PRICE_LLM_INPUT_PER_1K":  GEMINI_PRICE_LLM_INPUT_PER_1K,
            "PRICE_LLM_OUTPUT_PER_1K": GEMINI_PRICE_LLM_OUTPUT_PER_1K,
            "PRICE_EMBED_PER_1K":      GEMINI_PRICE_EMBED_PER_1K,
        }
    if provider == "bedrock":
        return {
            "MODEL_LLM":               BEDROCK_MODEL_LLM,
            "MODEL_EMBED":             BEDROCK_MODEL_EMBED,
            "EMBED_DIM":               BEDROCK_EMBED_DIM,
            "PRICE_LLM_INPUT_PER_1K":  BEDROCK_PRICE_LLM_INPUT_PER_1K,
            "PRICE_LLM_OUTPUT_PER_1K": BEDROCK_PRICE_LLM_OUTPUT_PER_1K,
            "PRICE_EMBED_PER_1K":      BEDROCK_PRICE_EMBED_PER_1K,
        }
    raise ValueError(f"unknown provider {provider!r}")


_SETTINGS = provider_settings(PROVIDER)
MODEL_LLM               = _SETTINGS["MODEL_LLM"]
MODEL_EMBED             = _SETTINGS["MODEL_EMBED"]
EMBED_DIM               = _SETTINGS["EMBED_DIM"]
PRICE_LLM_INPUT_PER_1K  = _SETTINGS["PRICE_LLM_INPUT_PER_1K"]
PRICE_LLM_OUTPUT_PER_1K = _SETTINGS["PRICE_LLM_OUTPUT_PER_1K"]
PRICE_EMBED_PER_1K      = _SETTINGS["PRICE_EMBED_PER_1K"]

# --- Paths ---
ROOT        = Path(__file__).parent
CACHE_DIR   = ROOT / ".cache"
LEDGER_PATH = ROOT / ".spend_ledger.json"
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INDEX_PATH  = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
META_PATH   = DATA_DIR / "meta.json"

for _d in (CACHE_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
