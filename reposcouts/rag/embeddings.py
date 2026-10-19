"""Local E5 adapter. Imports and model downloads happen only on first use."""
from functools import lru_cache

import numpy as np

from .chunking import format_passage
from .config import RagConfig
from .tracing import run_stage


@lru_cache(maxsize=1)
def _load_model(model_name: str, revision: str, local_files_only: bool = True):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name, revision=revision, trust_remote_code=False,
        local_files_only=local_files_only,
    )


class LocalEmbedder:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig()
        self.fingerprint = f"{self.config.embedding_model}@{self.config.embedding_revision}"
        self._model = run_stage(
            "rag.model_load", {"model": self.config.embedding_model, "local_files_only": self.config.local_files_only},
            lambda: _load_model(self.config.embedding_model, self.config.embedding_revision, self.config.local_files_only),
            lambda model: {"loaded": True, "device": str(model.device)},
        )
        self.max_tokens = min(512, int(self._model.max_seq_length))
        if self.config.chunk_tokens > self.max_tokens:
            raise ValueError("chunk_tokens exceeds the embedding model input limit")

    def token_count(self, text: str) -> int:
        encoded = self._model.tokenizer(
            text, add_special_tokens=True, truncation=False,
        )
        return len(encoded["input_ids"])

    def _encode(self, texts: list[str], *, kind: str) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        if any(self.token_count(text) > self.max_tokens for text in texts):
            raise ValueError("embedding input exceeds model limit; no silent truncation")
        return run_stage(
            f"rag.embedding.{kind}", {"text_count": len(texts), "model": self.config.embedding_model},
            lambda: np.asarray(self._model.encode(
                texts, batch_size=self.config.embedding_batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, prompt="",
            ), dtype=np.float32),
            lambda matrix: {"rows": len(matrix), "dimension": int(matrix.shape[1])},
        )

    def embed_documents(self, chunks) -> np.ndarray:
        return self._encode([
            format_passage(chunk.path, chunk.symbol, chunk.text) for chunk in chunks
        ], kind="documents")

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([f"query: {query}"], kind="query")
