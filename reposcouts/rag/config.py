from dataclasses import dataclass

@dataclass(frozen=True)
class RagConfig:
    """
    Configuration for RAG (Retrieval-Augmented Generation) model.
    """
    # local model used for embedding generation
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_batch_size: int = 16
    # Never download model weights as a side effect of a user request.
    local_files_only: bool = True

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    reranker_batch_size: int = 4
    reranker_max_tokens: int = 512
    rerank_candidates: int = 20
    reranker_enabled: bool = True
    cache_enabled: bool = True
    cache_dir: str = ".cache/reposcouts-rag"
    cache_ttl_s: int = 7 * 24 * 3600
    cache_max_bytes: int = 1024 ** 3
    initial_files: int = 40
    initial_repository_bytes: int = 2 * 1024 * 1024
    max_expansion_rounds: int = 2
    expansion_batch_size: int = 20

    max_files: int = 120                        # total across initial and expansion reads
    max_file_bytes: int = 128 * 1024            # one file max size
    max_repository_bytes: int = 8 * 1024 * 1024 # total across all rounds

    chunk_tokens: int = 400     # standard chunk size
    overlap_tokens: int = 40    # overlap for two chunks

    dense_top_k: int = 20       # top-k for dense retrieval(vector search)
    sparse_top_k: int = 20      # top-k for sparse retrieval(BM25)
    evidence_top_k: int = 5     # top-k for evidence selection
    rrf_k: int = 60

    http_timeout_s: float = 15.0
    repository_timeout_s: float = 100.0
    max_evidence_chars: int = 24000
    max_evidence_chunks: int = 20
    max_evaluator_input_chars: int = 48000

    # 后配置函数
    def __post_init__(self) -> None:
        positive = (
            self.max_files, self.max_file_bytes, self.max_repository_bytes,
            self.chunk_tokens, self.dense_top_k, self.sparse_top_k,
            self.evidence_top_k, self.rrf_k,
            self.http_timeout_s, self.repository_timeout_s,
            self.embedding_batch_size, self.max_evidence_chars, self.max_evidence_chunks,
            self.max_evaluator_input_chars,
            self.reranker_batch_size, self.reranker_max_tokens, self.rerank_candidates,
            self.cache_ttl_s, self.cache_max_bytes, self.initial_files,
            self.initial_repository_bytes, self.expansion_batch_size,
        )

        if any(value <= 0 for value in positive):
            raise ValueError("RAG budgets and timeouts must be positive")

        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("overlap_tokens must be in [0, chunk_tokens)")
        if self.max_expansion_rounds < 0:
            raise ValueError("max_expansion_rounds must not be negative")
        if self.rerank_candidates < self.evidence_top_k:
            raise ValueError("rerank_candidates must cover evidence_top_k")
        if self.cache_enabled and self.cache_max_bytes < 32768:
            raise ValueError("cache_max_bytes must accommodate SQLite metadata (at least 32768)")
