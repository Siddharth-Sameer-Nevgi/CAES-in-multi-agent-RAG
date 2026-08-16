"""Dense retrieval over the FAISS index.

The query embedding is a real cost and is a component of measured dC, so it
goes through bedrock.embed like everything else.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("caes.retrieval")


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    def __init__(self,
                 index_path: Path | str = config.INDEX_PATH,
                 chunks_path: Path | str = config.CHUNKS_PATH) -> None:
        import faiss

        index_path, chunks_path = Path(index_path), Path(chunks_path)
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Missing index ({index_path}) or chunks ({chunks_path}). "
                f"Run `python ingest.py` first."
            )
        self.index = faiss.read_index(str(index_path))
        with chunks_path.open(encoding="utf-8") as fh:
            self.chunks = [json.loads(line) for line in fh if line.strip()]
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/chunk mismatch: {self.index.ntotal} vectors vs "
                f"{len(self.chunks)} chunks. Rebuild with `ingest.py --force`."
            )
        log.info("Retriever ready: %d chunks", len(self.chunks))

    def search(self, query: str, k: int = config.TOP_K, *,
               query_id: str = "", iteration: int = 0,
               policy: str = "") -> list[Chunk]:
        import bedrock

        vec = bedrock.embed([query], query_id=query_id, iteration=iteration,
                            policy=policy)
        vec = np.ascontiguousarray(vec.astype("float32"))
        scores, ids = self.index.search(vec, k)

        out: list[Chunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            rec = self.chunks[int(idx)]
            out.append(Chunk(chunk_id=rec["chunk_id"], title=rec["title"],
                             text=rec["text"], score=float(score)))
        return out


_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide singleton; loading FAISS on every query is wasteful."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = get_retriever()
    for c in r.search("Who directed Inception?", k=5):
        print(f"{c.score:.3f}  {c.title}  |  {c.text[:110]}...")
