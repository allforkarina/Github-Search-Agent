"""One ResearchSession per request: pin snapshots and reuse indexes across retries."""
from contextlib import contextmanager
from contextvars import ContextVar

from reposcouts.core.schemas import UserRequirements

from .config import RagConfig
from .embeddings import LocalEmbedder
from .evidence import build_evidence
from .github_source import collect_repository_snapshot
from .github_source import expand_repository_snapshot
from .index_cache import RequestIndexCache
from .requirements import decompose_requirements
from .schemas import RepositoryResearch, RepositorySnapshot
from .tracing import run_stage
from .cache import SQLiteArtifactStore
from .incremental import IncrementalArtifacts, CachedEmbedder
from .reranking import LocalReranker
from .security import sanitize_content
from .context import add_symbol_context, related_paths, source_kind
from .planning import functional_requirement


class ResearchSession:
    """
    RAG 各模块总调度器，具体的流程如下：
    GitHub Source
        ↓
    Snapshot
        ↓
    Requirement Decompose
        ↓
    Chunking
        ↓
    Embedding / BM25 Index
        ↓
    Retrieval
        ↓
    Evidence
        ↓
    RepositoryResearch
    """
    def __init__(self, 
                 config=None, 
                 *, 
                 snapshot_loader=None, 
                 embedder_factory=None,
                 reranker_factory=None, 
                 snapshot_expander=None, 
                 artifact_store=None):
        self.config = config or RagConfig()
        self._snapshot_loader = snapshot_loader or collect_repository_snapshot
        self._default_loader = snapshot_loader is None
        self._snapshot_expander = snapshot_expander or expand_repository_snapshot
        self._embedder_factory = embedder_factory or LocalEmbedder
        self._reranker_factory = reranker_factory or LocalReranker
        
        self._snapshots = {}
        self._chunked = {}
        self._metadata = {}
        
        self._embedder = None
        self._model_attempted = False   # 本地 embedding 模型尝试加载次数
        
        self._warnings = []
        self._plans = {}
        self._artifacts = {}
        self._reranker = None
        self._reranker_attempted = False
        self._store = artifact_store
        self._owns_store = artifact_store is None
        # One serialized store per request; LangGraph may move it across threads.
        if self._store is None and self.config.cache_enabled:
            try:
                self._store = SQLiteArtifactStore(self.config)
            except Exception:
                self._warnings.append("persistent_cache_unavailable")
        self.indexes = RequestIndexCache(max_entries=64)

    # 采集仓库源码并缓存
    def snapshot(self, full_name: str, tasks=()) -> RepositorySnapshot:
        # 忽略大小写的 full name 作为当前 repo 缓存 key
        key = full_name.casefold()
        # 当前仓库还没有缓存过
        if key not in self._snapshots:
            try:
                value = run_stage(
                    "rag.ingestion", 
                    {"repository": full_name},
                    lambda: (
                        self._snapshot_loader(
                            full_name, 
                            self.config, 
                            queries=[task.query for task in tasks]
                        ) if self._default_loader else self._snapshot_loader(full_name, self.config)
                    ),
                    lambda snapshot: {"commit_sha": snapshot.commit_sha, **snapshot.report.model_dump()},
                )
            except Exception:
                # 直接生成一个 ingestion failed 的 snapshot 实例
                value = RepositorySnapshot(full_name=full_name)
                value.report.warnings.append("ingestion_failed")
                
            # 直接缓存 snapshot，避免重复采集源码    
            self._snapshots[key] = value
        return self._snapshots[key]

    def metadata(self, full_name: str) -> dict:
        key = full_name.casefold()
        if key not in self._metadata:
            # 延迟导入，避免不必要的依赖
            from reposcouts.github.tools import get_repository_info
            from .security import sanitize_data
            try:
                self._metadata[key] = sanitize_data(
                    get_repository_info.invoke({"full_name": full_name})
                )
            except Exception:
                # 失败时 metadata 为空字典，继续流程
                self._metadata[key] = {}
        return self._metadata[key]

    def _get_embedder(self) -> LocalEmbedder | None:
        # 当前 embedding 模型还没有尝试加载，可以 load_model
        if not self._model_attempted:
            # 加载失败也标注为 True，避免后续多次失败加载
            self._model_attempted = True
            try:
                self._embedder = self._embedder_factory(self.config)
            except Exception:
                self._warnings.append("local_embedding_unavailable")
        return self._embedder

    def research(self, full_name: str, requirements: UserRequirements) -> RepositoryResearch:
        plan_key = requirements.model_dump_json()
        if plan_key not in self._plans:
            self._plans[plan_key] = run_stage(
                "rag.query_rewrite", {"must_have_count": len(requirements.must_have)},
                lambda: decompose_requirements(requirements),
                lambda tasks: {"task_count": len(tasks), "query_count": sum(len(t.queries) or 1 for t in tasks),
                               "warnings": requirements.plan_warnings},
            )
        tasks = self._plans[plan_key]
        snapshot = self.snapshot(full_name, tasks)
        # Sanitize entire files BEFORE chunking, embedding or persistent writes.
        self._sanitize_snapshot(snapshot)
        embedder = self._get_embedder() if snapshot.files else None
        if snapshot.files and self.config.reranker_enabled and not self._reranker_attempted:
            self._reranker_attempted = True
            try:
                self._reranker = self._reranker_factory(self.config)
            except Exception:
                self._warnings.append("local_reranker_unavailable")
        key = full_name.casefold()
        if key not in self._artifacts:
            self._artifacts[key] = IncrementalArtifacts(self._store, self.config, embedder)
        artifacts = self._artifacts[key]
        count = embedder.token_count if embedder is not None else lambda text: len(text.encode("utf-8")) + 2
        indexed_embedder = CachedEmbedder(embedder, artifacts) if embedder is not None else None
        for round_number in range(self.config.max_expansion_rounds + 1):
            if key not in self._chunked:
                try:
                    self._chunked[key] = run_stage(
                        "rag.chunking", {"repository": full_name, "file_count": len(snapshot.files)},
                        lambda: artifacts.chunks(snapshot, count),
                        lambda chunks: {"chunk_count": len(chunks.chunks), "fallback_files": chunks.fallback_files},
                    )
                except Exception:
                    self._embedder = embedder = indexed_embedder = None
                    self._warnings.append("tokenizer_unavailable")
                    artifacts = self._artifacts[key] = IncrementalArtifacts(self._store, self.config)
                    count = lambda text: len(text.encode("utf-8")) + 2
                    self._chunked[key] = artifacts.chunks(snapshot, count)
            chunked = self._chunked[key]
            index, reused = run_stage(
                "rag.index", {"repository": full_name, "chunk_count": len(chunked.chunks)},
                lambda: self.indexes.get_or_build(chunked.chunks, embedder=indexed_embedder, config=self.config),
                lambda value: {"cache_hit": value[1], "warnings": value[0].warnings,
                               "artifact_stats": dict(artifacts.stats)},
            )
            queries, results = {}, {}
            for task in tasks:
                expressions = tuple(task.queries or [task.query])
                if expressions not in queries:
                    result = run_stage(
                        "rag.retrieve", {"repository": full_name, "task_id": task.task_id},
                        lambda: index.search_many(expressions, reranker=self._reranker),
                        lambda result: {"mode": result.mode, "hit_count": len(result.hits),
                                        "reranked": result.reranked, "warnings": result.warnings},
                    )
                    queries[expressions] = run_stage(
                        "rag.context", {"task_id": task.task_id},
                        lambda: add_symbol_context(index, result, self.config.evidence_top_k),
                        lambda value: {"chunk_ids": [hit.chunk.chunk_id for hit in value.hits]},
                    )
                results[task.task_id] = queries[expressions]
            hits = [hit for result in results.values() for hit in result.hits]
            paths = related_paths(snapshot, [q for task in tasks for q in task.queries or [task.query]], hits)
            missing = any(not any(source_kind(h.chunk.path) in ("source", "example")
                                  for h in results[task.task_id].hits)
                          for task in tasks if functional_requirement(task.parent_requirement))
            if (round_number == self.config.max_expansion_rounds or not paths
                    or snapshot.report.expansion_rounds >= self.config.max_expansion_rounds):
                break
            reason = "functional_source_gap" if missing else "related_import_or_symbol_context"
            snapshot.report.expansion_reasons.append(reason)
            try:
                loaded = run_stage(
                    "rag.ingestion.expand", {"repository": full_name, "reason": reason, "candidate_paths": len(paths)},
                    lambda: self._snapshot_expander(snapshot, paths, self.config),
                    lambda value: {"loaded_files": value, "total_files": len(snapshot.files),
                                   "round": snapshot.report.expansion_rounds},
                )
            except Exception:
                self._warnings.append("expansion_failed")
                break
            if not loaded:
                break
            self._sanitize_snapshot(snapshot)
            self._chunked.pop(key, None)
        manifest_stats = run_stage(
            "rag.index.publish", {"repository": full_name, "commit_sha": snapshot.commit_sha},
            lambda: artifacts.publish(snapshot) if snapshot.commit_sha else {},
            lambda stats: {**stats, **artifacts.stats, "warnings": sorted(set(artifacts.warnings))},
        )
        pack = run_stage(
            "rag.evidence", 
            {"repository": full_name, "task_count": len(tasks)},
            lambda: build_evidence(
                snapshot, 
                tasks, 
                results, 
                chunked, 
                config=self.config, 
                index_reused=reused
            ),
            lambda value: {"snippet_count": len(value.snippets), "warnings": value.warnings},
        )
        pack.cache_stats = {**artifacts.stats, **manifest_stats}
        pack.reranker_model = self.config.reranker_model if self._reranker is not None else None
        pack.warnings = sorted(set(pack.warnings + self._warnings + artifacts.warnings + requirements.plan_warnings))
        if embedder is None and snapshot.files:
            pack.warnings.append("byte_budget_chunking")
        return pack

    @staticmethod
    def _sanitize_snapshot(snapshot):
        for file in snapshot.files:
            safe = sanitize_content(file.content)
            file.redacted = file.redacted or safe != file.content
            file.content = safe

    def close(self):
        """
        缓存清空函数
        """
        self.indexes.clear()
        self._snapshots.clear()
        self._chunked.clear()
        self._metadata.clear()
        self._embedder = None
        self._reranker = None
        self._plans.clear()
        self._artifacts.clear()
        if self._store is not None and self._owns_store:
            try:
                self._store.close()
            except Exception:
                pass
        self._store = None


_current: ContextVar[ResearchSession | None] = ContextVar("reposcouts_research", default=None)


def current_research() -> ResearchSession | None:
    return _current.get()


@contextmanager
def research_scope(session: ResearchSession | None = None):
    existing = current_research()
    if existing is not None:
        yield existing
        return
    
    # 如果没有现有的 ResearchSession，则创建一个新的 ResearchSession
    active = session or ResearchSession()
    token = _current.set(active)
    try:
        yield active
    finally:
        # 请求最后清理，释放缓存
        active.close()
        _current.reset(token)
