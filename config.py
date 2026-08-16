"""Central configuration for CAES-RAG.

Every tunable lives here. Nothing else in the codebase should hardcode a
price, a model id, or a loop bound.
"""
from dataclasses import dataclass
from pathlib import Path

# --- Pricing (USD). VERIFY against https://aws.amazon.com/bedrock/pricing/
#     before final experiment runs; these change. ---
PRICE_HAIKU_INPUT_PER_1K   = 0.001    # $1.00 / 1M tokens
PRICE_HAIKU_OUTPUT_PER_1K  = 0.005    # $5.00 / 1M tokens
PRICE_TITAN_EMBED_PER_1K   = 0.00011  # $0.11 / 1M tokens

# --- Hard spend guards ---
HARD_BUDGET_USD        = 40.00   # cumulative ceiling; raises BudgetExceeded
WARN_BUDGET_USD        = 25.00   # logs a loud warning
SINGLE_RUN_MAX_USD     = 5.00    # ceiling for one experiment invocation

# --- Loop guards ---
MAX_ITERATIONS = 5    # absolute hard cap; a gate bug cannot exceed this
MIN_ITERATIONS = 1

# --- Model IDs ---
MODEL_LLM   = "anthropic.claude-haiku-4-5"
MODEL_EMBED = "amazon.titan-embed-text-v2:0"
AWS_REGION  = "us-east-1"

# --- CAES parameters (LAMBDA is set by Phase 5 tuning; do not guess) ---
LAMBDA = None          # must be set explicitly before CAES runs
DECAY_FACTOR = 0.6     # dQ extrapolation

# --- Retrieval ---
TOP_K = 5
CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 30
EMBED_BATCH = 50
EMBED_DIM = 1024              # titan-embed-text-v2 default output dimension

# --- Verifier budget ---
VERIFIER_CHUNK_CHARS = 600    # ~150 tokens per chunk of truncated evidence

# --- Dataset splits (deterministic, disjoint) ---
CORPUS_SAMPLE_SIZE = 2000
SPLIT_SEED   = 20240917
N_TUNE       = 50     # held-out lambda-tuning questions
N_TEST       = 150    # evaluation questions, disjoint from tuning

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
