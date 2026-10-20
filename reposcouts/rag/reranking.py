"""Local, pinned Cross-Encoder. Missing weights never trigger a download."""
from functools import lru_cache

import numpy as np

from .tracing import run_stage


@lru_cache(maxsize=1)
def _load_reranker(model_name, revision, max_tokens, local_files_only=True):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, revision=revision, max_length=max_tokens,
                        local_files_only=local_files_only, trust_remote_code=False)


class LocalReranker:
    def __init__(self, config):
        self.config = config
        self.fingerprint = config.reranker_model + "@" + config.reranker_revision
        self.model = run_stage(
            "rag.reranker_load", {"model": config.reranker_model, "local_files_only": config.local_files_only},
            lambda: _load_reranker(config.reranker_model, config.reranker_revision,
                                   config.reranker_max_tokens, config.local_files_only),
            lambda model: {"loaded": True},
        )
        self.truncated_pairs = 0

    def score(self, query, hits):
        pairs = [(query, f"{h.chunk.path}\n{h.chunk.symbol or ''}\n{h.chunk.text}") for h in hits]
        self.truncated_pairs = sum(
            len(self.model.tokenizer(a, b, truncation=False)["input_ids"]) > self.config.reranker_max_tokens
            for a, b in pairs
        )
        return self.model.predict(pairs, batch_size=self.config.reranker_batch_size,
                                  show_progress_bar=False, convert_to_numpy=True)


def rerank(query, hits, reranker, config):
    if not hits:
        return [], [], False
    if reranker is None:
        return hits[:config.evidence_top_k], ["reranker_unavailable"] if config.reranker_enabled else [], False
    try:
        def score():
            values = np.asarray(reranker.score(query, hits), dtype=float).reshape(-1)
            if len(values) != len(hits) or not np.isfinite(values).all():
                raise ValueError("invalid_rerank_scores")
            return values
        scores = run_stage(
            "rag.rerank", {"candidate_count": len(hits), "model": config.reranker_model}, score,
            lambda values: {"scored_count": len(values),
                            "truncated_pairs": getattr(reranker, "truncated_pairs", 0)},
        )
        ranked = sorted(zip(hits, scores), key=lambda item: (-float(item[1]), item[0].chunk.chunk_id))
        result = [hit.model_copy(update={"rerank_score": float(value)}) for hit, value in ranked[:config.evidence_top_k]]
        warnings = ["reranker_input_truncated"] if getattr(reranker, "truncated_pairs", 0) else []
        return result, warnings, True
    except Exception:
        return hits[:config.evidence_top_k], ["reranker_failed_rrf_fallback"], False
