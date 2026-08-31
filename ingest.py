"""Phase 1: build the HotpotQA corpus, embed it, and persist a FAISS index.

One paid run, roughly $0.50. Guarded: if data/index.faiss already exists this
script refuses to re-embed.

    python ingest.py                # build
    python ingest.py --upload-s3 my-bucket
    python ingest.py --force        # deliberate rebuild (re-spends)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("caes.ingest")

# HotpotQA paragraphs are short; chunk on whitespace words using a
# words-per-token ratio, so CHUNK_TOKENS stays the unit of configuration.
_WORDS_PER_TOKEN = 0.75
CHUNK_WORDS = int(config.CHUNK_TOKENS * _WORDS_PER_TOKEN)          # ~150
OVERLAP_WORDS = int(config.CHUNK_OVERLAP_TOKENS * _WORDS_PER_TOKEN)  # ~22


def chunk_text(text: str, size: int = CHUNK_WORDS,
               overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


def load_hotpotqa(sample_size: int, seed: int):
    from datasets import load_dataset

    log.info("Loading hotpot_qa/distractor validation split...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return ds.select(idx[:sample_size])


def build_corpus(rows) -> tuple[list[dict], list[dict]]:
    """Return (questions, passages). Passages are deduplicated by title."""
    questions: list[dict] = []
    by_title: dict[str, str] = {}

    for row in rows:
        questions.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "supporting_titles": sorted(set(
                row.get("supporting_facts", {}).get("title", []))),
        })
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            if title in by_title:
                continue
            by_title[title] = " ".join(s.strip() for s in sentences).strip()

    passages = [{"title": t, "text": txt} for t, txt in by_title.items() if txt]
    return questions, passages


def build_chunks(passages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for p in passages:
        for j, piece in enumerate(chunk_text(p["text"])):
            chunks.append({
                "chunk_id": f"{len(chunks)}",
                "title": p["title"],
                "part": j,
                # Titles carry real signal in HotpotQA; prepend for retrieval.
                "text": f"{p['title']}: {piece}",
            })
    return chunks


def embed_chunks(chunks: list[dict], batch: int = config.EMBED_BATCH) -> np.ndarray:
    import llm
    try:
        from tqdm import tqdm
    except ImportError:                                   # pragma: no cover
        def tqdm(x, **kw):
            return x

    vecs = np.zeros((len(chunks), config.EMBED_DIM), dtype="float32")
    for start in tqdm(range(0, len(chunks), batch), desc="embedding",
                      unit="batch"):
        window = chunks[start:start + batch]
        vecs[start:start + len(window)] = llm.embed(
            [c["text"] for c in window], policy="ingest")
    return vecs


def build_index(vectors: np.ndarray):
    import faiss

    # Titan V2 returns L2-normalised vectors, so inner product == cosine.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    return index


def upload_to_s3(bucket: str, passages: list[dict]) -> None:
    """Substantiates the paper's ingestion-layer claim. Not on the query path."""
    import boto3

    s3 = boto3.client("s3", region_name=config.AWS_REGION)
    body = "\n".join(json.dumps(p) for p in passages).encode("utf-8")
    key = "caes-rag/corpus/passages.jsonl"
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    log.info("Uploaded %d passages to s3://%s/%s", len(passages), bucket, key)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the CAES-RAG corpus index.")
    ap.add_argument("--sample", type=int, default=config.CORPUS_SAMPLE_SIZE)
    ap.add_argument("--upload-s3", metavar="BUCKET", default=None)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if an index exists (re-spends money)")
    args = ap.parse_args(argv)

    if config.INDEX_PATH.exists() and not args.force:
        synthetic = False
        if config.META_PATH.exists():
            try:
                synthetic = json.loads(
                    config.META_PATH.read_text(encoding="utf-8")
                ).get("synthetic", False)
            except (json.JSONDecodeError, OSError):
                pass
        if synthetic:
            print(f"{config.INDEX_PATH} holds the SYNTHETIC devdata.py corpus, "
                  f"not HotpotQA.\nDelete data/ (or pass --force) before "
                  f"building the real index.")
            return 2
        print(f"{config.INDEX_PATH} already exists — refusing to re-embed.\n"
              f"Delete it or pass --force if you really mean to spend again.")
        return 0

    import llm
    from costs import TRACKER

    rows = load_hotpotqa(args.sample, config.SPLIT_SEED)
    questions, passages = build_corpus(rows)
    chunks = build_chunks(passages)
    log.info("%d questions, %d deduplicated passages, %d chunks",
             len(questions), len(passages), len(chunks))

    est = sum(len(c["text"]) / 3.6 for c in chunks) / 1000 * \
        config.PRICE_EMBED_PER_1K
    print(f"Estimated embedding cost: ${est:.2f} "
          f"(cumulative so far ${TRACKER.cumulative():.2f} of "
          f"${config.HARD_BUDGET_USD:.2f})")

    vectors = embed_chunks(chunks)
    index = build_index(vectors)

    import faiss
    faiss.write_index(index, str(config.INDEX_PATH))
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    with (config.DATA_DIR / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q) + "\n")
    config.META_PATH.write_text(json.dumps({
        "n_questions": len(questions),
        "n_passages": len(passages),
        "n_chunks": len(chunks),
        "embed_model": config.MODEL_EMBED,
        "embed_dim": config.EMBED_DIM,
        "chunk_tokens": config.CHUNK_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "sample_size": args.sample,
        "split_seed": config.SPLIT_SEED,
        "dry_run": llm.DRY_RUN,
    }, indent=2), encoding="utf-8")

    if args.upload_s3:
        upload_to_s3(args.upload_s3, passages)

    from cache import CACHE
    CACHE.log_stats("ingest cache")
    print(f"Done. Cumulative spend now ${TRACKER.cumulative():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
