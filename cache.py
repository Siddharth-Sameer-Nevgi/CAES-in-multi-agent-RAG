"""Disk cache for Bedrock responses.

This is the single most important cost control during development: re-running an
experiment after a bug fix must be free for everything already computed.

Contract:
  * Key = sha256(model + json.dumps(payload, sort_keys=True)).
  * A cache hit MUST NOT touch CostTracker. A replayed call costs nothing.
  * Hit/miss counters are exposed so the developer can confirm it is working.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("caes.cache")


def make_key(model: str, payload: dict[str, Any]) -> str:
    blob = model + json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, cache_dir: Path | str = config.CACHE_DIR,
                 enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.RLock()

    def _path(self, key: str) -> Path:
        # Shard by first two hex chars so no directory holds ~20k files.
        shard = self.cache_dir / key[:2]
        shard.mkdir(parents=True, exist_ok=True)
        return shard / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt entry is a miss, not a crash. Drop it and re-fetch.
            log.warning("Dropping corrupt cache entry %s (%s)", key[:12], exc)
            p.unlink(missing_ok=True)
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(p)
        with self._lock:
            self.writes += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
                "hit_rate": (self.hits / total) if total else 0.0,
            }

    def log_stats(self, prefix: str = "cache") -> None:
        s = self.stats()
        log.info("%s: %d hits / %d misses (%.1f%% hit rate), %d writes",
                 prefix, s["hits"], s["misses"], 100 * s["hit_rate"], s["writes"])


CACHE = DiskCache()
