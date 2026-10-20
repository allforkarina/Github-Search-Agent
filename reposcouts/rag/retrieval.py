"""Repository-scoped dense + BM25 retrieval with rank-based RRF fusion."""
import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25L

from .config import RagConfig
from .schemas import EvidenceChunk, RetrievalHit, RetrievalResult
from .tracing import run_stage


# 为 Sparse retrieval BM25L 做分词
def lexical_tokens(text: str) -> list[str]:
    tokens = []
    # 从text中提取 英文字符、数字、中文以及标识符
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]+", text):
        # 中文字符
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            # 保存单字
            tokens.extend(word)
            # 保存双字
            tokens.extend(word[index:index + 2] for index in range(len(word) - 1))
        # 英文字符或数字
        else:
            # 保存原始单词的小写形式
            tokens.append(word.lower())
            parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|_|$)|[A-Z]?[a-z]+|\d+", word)
            # 保存分词的小写形式，忽略与原始单词相同的分词
            tokens.extend(part.lower() for part in parts if part.lower() != word.lower())
    return tokens


def rrf_fuse(dense_ids, sparse_ids, *, k=60):
    if k <= 0:
        raise ValueError("RRF k must be positive")
    
    scores = {}
    # 将 dense 和 sparse retrieval 的结果按照 RRF 算法进行融合
    for ids in (dense_ids, sparse_ids):
        for rank, chunk_id in enumerate(dict.fromkeys(ids), 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _unit_matrix(value, rows):
    # 二维 nparray
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError("invalid embedding matrix shape")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite embedding values")
    
    # 归一化
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise ValueError("zero or invalid embedding norm")
    return matrix / norms


class HybridIndex:
    def __init__(
        self, chunks: Sequence[EvidenceChunk], 
        *, 
        embedder=None,
        config: RagConfig | None = None,
    ):
        self.config = config or RagConfig()
        self.chunks = tuple(chunk.model_copy(deep=True) for chunk in chunks)
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk IDs")
        
        versions = {(chunk.full_name, chunk.commit_sha) for chunk in self.chunks}
        if len(versions) > 1:
            raise ValueError("one index must contain exactly one repository version")
        
        # 创建一个字典，将 chunk_id 映射到其在 self.chunks 中的索引
        self._by_id = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        # 创建一个列表，将 chunk 的路径、符号和文本内容进行分词，作为 BM25L 的输入
        self._terms = [lexical_tokens(f"{c.path}\n{c.symbol or ''}\n{c.text}") for c in self.chunks]
        self._term_sets = [set(terms) for terms in self._terms]
        # BM25L uses positive IDF even for 适合小规模文档集合
        self._sparse = BM25L([terms or ["__empty_document__"] for terms in self._terms]) if self.chunks else None
        self._embedder = embedder
        self._vectors = None
        self.warnings = []
        if self.chunks and embedder is not None:
            try:
                self._vectors = _unit_matrix(embedder.embed_documents(self.chunks), len(self.chunks))
            except Exception:
                self.warnings.append("document_embedding_failed")
        elif self.chunks:
            self.warnings.append("dense_disabled")

    def _sparse_search(self, query):
        terms = list(dict.fromkeys(lexical_tokens(query)))
        if not terms or self._sparse is None:
            return [], {}
        
        # 在 BM25L 中用 terms 匹配每一个 chunk 的得分，query terms 在 chunk 中出现越多分数越高
        scores = self._sparse.get_scores(terms)
        query_set = set(terms)
        indices = [
            index for index, score in enumerate(scores)
            # query terms 与 chunk terms 有交集，且 score 为有限数值且大于 0
            if self._term_sets[index] & query_set and np.isfinite(score) and score > 0
        ]
        # 排序 按照 query terms 在 chunk 中的匹配得分降序
        indices.sort(key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id))
        # 取 top-k 的 chunk_id 和对应的 score
        indices = indices[:self.config.sparse_top_k]
        return (
            [self.chunks[index].chunk_id for index in indices],
            {self.chunks[index].chunk_id: float(scores[index]) for index in indices},
        )

    def search(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        if not self.chunks:
            return RetrievalResult(query=query, mode="unavailable", warnings=["empty_index"])
        
        sparse_ids, sparse_scores = run_stage(
            "rag.bm25", 
            {"chunk_count": len(self.chunks)}, 
            lambda: self._sparse_search(query),
            lambda value: {"hit_count": len(value[0]), "chunk_ids": value[0]},
        )
        
        dense_ids, dense_scores = [], {}
        warnings = list(self.warnings)
        dense_available = self._vectors is not None
        if dense_available:
            try:
                dense_ids, dense_scores = run_stage(
                    "rag.dense", 
                    {"chunk_count": len(self.chunks)}, 
                    lambda: self._dense_search(query),
                    lambda value: {"hit_count": len(value[0]), "chunk_ids": value[0]},
                )
            except Exception:
                dense_available = False
                warnings.append("query_embedding_failed")
                
        dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, 1)}
        sparse_ranks = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ids, 1)}
        fused = run_stage(
            "rag.rrf", 
            {
                "dense_count": len(dense_ids), 
                "bm25_count": len(sparse_ids), 
            },
            lambda: rrf_fuse(dense_ids, sparse_ids, k=self.config.rrf_k)[:top_k or self.config.evidence_top_k],
            lambda value: {"hit_count": len(value), "chunk_ids": [item[0] for item in value]},
        )
        
        hits = [
            RetrievalHit(
                chunk=self.chunks[self._by_id[chunk_id]].model_copy(deep=True), 
                rrf_score=score,
                dense_rank=dense_ranks.get(chunk_id), 
                bm25_rank=sparse_ranks.get(chunk_id),
                dense_score=dense_scores.get(chunk_id), 
                bm25_score=sparse_scores.get(chunk_id),
            )
            for chunk_id, score in fused
        ]
        return RetrievalResult(
            query=query, mode="hybrid" if dense_available else "bm25",
            hits=hits, warnings=warnings,
        )

    def search_many(self, queries, *, reranker=None):
        from .reranking import rerank
        queries = list(dict.fromkeys(queries))
        if not queries or len(queries) > 4:
            raise ValueError("one_to_four_queries_required")
        branches = [self.search(query, top_k=self.config.rerank_candidates) for query in queries]
        scores, catalog = {}, {}
        for branch in branches:
            for rank, hit in enumerate(branch.hits, 1):
                key = hit.chunk.chunk_id
                scores[key] = scores.get(key, 0.0) + 1 / (self.config.rrf_k + rank)
                catalog.setdefault(key, hit)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))[:self.config.rerank_candidates]
        hits = [catalog[key].model_copy(update={"rrf_score": scores[key]}) for key in ordered]
        hits, warnings, reranked = rerank(queries[0], hits, reranker, self.config)
        mode = "hybrid" if all(b.mode == "hybrid" for b in branches) else "bm25"
        if not self.chunks:
            mode = "unavailable"
        return RetrievalResult(query=queries[0], mode=mode, hits=hits, reranked=reranked,
                               warnings=sorted(set(warnings + [w for b in branches for w in b.warnings])))

    def _dense_search(self, query):
        # query embedding vector
        vector = _unit_matrix(self._embedder.embed_query(query), 1)
        if vector.shape[1] != self._vectors.shape[1]:
            raise ValueError("query/document embedding dimension mismatch")
        
        # 计算 query 与每个 chunk 的相似度
        scores = self._vectors @ vector[0]
        indices = sorted(range(len(self.chunks)), key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id))[:self.config.dense_top_k]
        return (
            [self.chunks[index].chunk_id for index in indices],
            {self.chunks[index].chunk_id: float(scores[index]) for index in indices},
        )
