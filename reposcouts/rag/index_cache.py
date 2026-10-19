"""Create one cache per user request; release it when the request ends."""
import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict

from .config import RagConfig
from .retrieval import HybridIndex


class RequestIndexCache:
    def __init__(self, max_entries: int = 16):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._indexes = OrderedDict()

    def get_or_build(self, chunks, *, embedder=None, config=None):
        config = config or RagConfig()
        chunks = tuple(chunks)
        payload = {
            "index_version": "rag-v1-bm25l-lexical-v1-rrf-v1",
            "config": asdict(config),
            "embedding": embedder.fingerprint if embedder is not None else "bm25-only",
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        key = hashlib.sha256(json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if key in self._indexes:
            self._indexes.move_to_end(key)
            return self._indexes[key], True
        index = HybridIndex(chunks, embedder=embedder, config=config)
        self._indexes[key] = index
        if len(self._indexes) > self._max_entries:
            self._indexes.popitem(last=False)
        return index, False

    def clear(self):
        self._indexes.clear()
