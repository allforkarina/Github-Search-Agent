"""One ResearchSession per request: pin snapshots and reuse indexes across retries."""
from contextlib import contextmanager
from contextvars import ContextVar

from reposcouts.core.schemas import UserRequirements

from .chunking import chunk_snapshot
from .config import RagConfig
from .embeddings import LocalEmbedder
from .evidence import build_evidence
from .github_source import collect_repository_snapshot
from .index_cache import RequestIndexCache
from .requirements import decompose_requirements
from .schemas import RepositoryResearch, RepositorySnapshot
from .tracing import run_stage


class ResearchSession:
    def __init__(self, config=None, *, snapshot_loader=None, embedder_factory=None):
        self.config = config or RagConfig()
        self._snapshot_loader = snapshot_loader or collect_repository_snapshot
        self._embedder_factory = embedder_factory or LocalEmbedder
        self._snapshots = {}
        self._chunked = {}
        self._metadata = {}
        self._embedder = None
        self._model_attempted = False
        self._warnings = []
        self.indexes = RequestIndexCache()

    def snapshot(self, full_name: str) -> RepositorySnapshot:
        key = full_name.casefold()
        if key not in self._snapshots:
            try:
                value = run_stage(
                    "rag.ingestion", {"repository": full_name},
                    lambda: self._snapshot_loader(full_name, self.config),
                    lambda snapshot: {"commit_sha": snapshot.commit_sha, **snapshot.report.model_dump()},
                )
            except Exception:
                value = RepositorySnapshot(full_name=full_name)
                value.report.warnings.append("ingestion_failed")
            self._snapshots[key] = value
        return self._snapshots[key]

    def metadata(self, full_name: str) -> dict:
        key = full_name.casefold()
        if key not in self._metadata:
            from reposcouts.github.tools import get_repository_info
            from .security import sanitize_data
            try:
                self._metadata[key] = sanitize_data(get_repository_info.invoke({"full_name": full_name}))
            except Exception:
                self._metadata[key] = {}
        return self._metadata[key]

    def _get_embedder(self) -> LocalEmbedder | None:
        if not self._model_attempted:
            self._model_attempted = True
            try:
                self._embedder = self._embedder_factory(self.config)
            except Exception:
                self._warnings.append("local_embedding_unavailable")
        return self._embedder

    def research(self, full_name: str, requirements: UserRequirements) -> RepositoryResearch:
        snapshot = self.snapshot(full_name)
        tasks = decompose_requirements(requirements)
        embedder = self._get_embedder() if snapshot.files else None
        key = full_name.casefold()
        if key not in self._chunked:
            def split():
                count = embedder.token_count if embedder is not None else lambda text: len(text.encode("utf-8")) + 2
                return chunk_snapshot(snapshot, token_count=count, config=self.config)
            try:
                self._chunked[key] = run_stage(
                    "rag.chunking", {"repository": full_name, "file_count": len(snapshot.files)}, split,
                    lambda chunks: {"chunk_count": len(chunks.chunks), "fallback_files": chunks.fallback_files,
                                    "oversized_line_count": sum(map(len, chunks.oversized_lines.values()))},
                )
            except Exception:
                # No tokenizer => BM25 can still index byte-bounded, source-grounded chunks.
                self._embedder = embedder = None
                self._warnings.append("tokenizer_unavailable")
                self._chunked[key] = chunk_snapshot(
                    snapshot, token_count=lambda text: len(text.encode("utf-8")) + 2, config=self.config,
                )
        chunked = self._chunked[key]
        index, reused = run_stage(
            "rag.index", {"repository": full_name, "chunk_count": len(chunked.chunks)},
            lambda: self.indexes.get_or_build(chunked.chunks, embedder=embedder, config=self.config),
            lambda value: {"cache_hit": value[1], "warnings": value[0].warnings},
        )
        queries = {}
        results = {}
        for task in tasks:
            if task.query not in queries:
                queries[task.query] = run_stage(
                    "rag.retrieve", {"repository": full_name, "task_id": task.task_id},
                    lambda: index.search(task.query),
                    lambda result: {"mode": result.mode, "hit_count": len(result.hits),
                                    "chunk_ids": [hit.chunk.chunk_id for hit in result.hits], "warnings": result.warnings},
                )
            results[task.task_id] = queries[task.query]
        pack = run_stage(
            "rag.evidence", {"repository": full_name, "task_count": len(tasks)},
            lambda: build_evidence(snapshot, tasks, results, chunked, config=self.config, index_reused=reused),
            lambda value: {"snippet_count": len(value.snippets), "warnings": value.warnings},
        )
        pack.warnings = sorted(set(pack.warnings + self._warnings))
        if embedder is None and snapshot.files:
            pack.warnings.append("byte_budget_chunking")
        return pack

    def close(self):
        self.indexes.clear()
        self._snapshots.clear()
        self._chunked.clear()
        self._metadata.clear()
        self._embedder = None


_current: ContextVar[ResearchSession | None] = ContextVar("reposcouts_research", default=None)


def current_research() -> ResearchSession | None:
    return _current.get()


@contextmanager
def research_scope(session: ResearchSession | None = None):
    existing = current_research()
    if existing is not None:
        yield existing
        return
    active = session or ResearchSession()
    token = _current.set(active)
    try:
        yield active
    finally:
        active.close()
        _current.reset(token)
