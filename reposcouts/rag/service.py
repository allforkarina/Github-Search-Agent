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
    def __init__(self, config=None, *, snapshot_loader=None, embedder_factory=None):
        self.config = config or RagConfig()
        self._snapshot_loader = snapshot_loader or collect_repository_snapshot
        self._embedder_factory = embedder_factory or LocalEmbedder
        
        self._snapshots = {}    # repo snapshot 缓存
        self._chunked = {}      # repo source chunk 缓存
        self._metadata = {}     # repo metadata 缓存
        
        self._embedder = None
        self._model_attempted = False   # 本地 embedding 模型尝试加载次数
        
        self._warnings = []
        self.indexes = RequestIndexCache()

    # 采集仓库源码并缓存
    def snapshot(self, full_name: str) -> RepositorySnapshot:
        # 忽略大小写的 full name 作为当前 repo 缓存 key
        key = full_name.casefold() 
        # 当前仓库还没有缓存过
        if key not in self._snapshots:
            try:
                value = run_stage(
                    "rag.ingestion",            # 阶段：ingestion
                    {"repository": full_name},  # 输入摘要 repo: full_name
                    # 执行采集源码操作 collect_repository_snapshot
                    lambda: self._snapshot_loader(full_name, self.config),  
                    # 执行结果摘要 当前仓库的 commit_sha 和 report 的 model_dump
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
        snapshot = self.snapshot(full_name)             # 仓库快照
        tasks = decompose_requirements(requirements)    # 对用户需求拆分得到的任务列表
        embedder = self._get_embedder() if snapshot.files else None # 获取到了 snapshot 再加载模型
        key = full_name.casefold()
        
        if key not in self._chunked:
            # 当前仓库还不在 chunk 缓存
            def split():
                # 根据当前的 embedder 确定最大的 chunk 的大小
                count = embedder.token_count if embedder is not None else lambda text: len(text.encode("utf-8")) + 2
                return chunk_snapshot(snapshot, token_count=count, config=self.config)
            
            try:
                # trace: 对 repo_snapshot 中的所有文件执行 split() 分块操作
                self._chunked[key] = run_stage(
                    "rag.chunking", 
                    {"repository": full_name, "file_count": len(snapshot.files)}, 
                    split,
                    lambda chunks: {"chunk_count": len(chunks.chunks), "fallback_files": chunks.fallback_files,
                                    "oversized_line_count": sum(map(len, chunks.oversized_lines.values()))},
                )
            except Exception:
                # 没有 embedder 按照文件的字节数进行分块，避免 chunking 失败
                self._embedder = embedder = None
                self._warnings.append("tokenizer_unavailable")
                self._chunked[key] = chunk_snapshot(
                    snapshot, 
                    token_count=lambda text: len(text.encode("utf-8")) + 2, 
                    config=self.config,
                )
                
        # 获取当前文件的 chunked 结果，创建或复用索引
        chunked = self._chunked[key]
        index, reused = run_stage(
            "rag.index", 
            {"repository": full_name, "chunk_count": len(chunked.chunks)},
            lambda: self.indexes.get_or_build(chunked.chunks, embedder=embedder, config=self.config),
            lambda value: {"cache_hit": value[1], "warnings": value[0].warnings},
        )
        
        queries = {}    # 用户需求拆分后的查询列表
        results = {}    # 用户需求拆分后的查询结果列表
        for task in tasks:
            # 遍历任务列表
            if task.query not in queries:
                # 当前任务的 query 还没有执行过，执行索引检索操作
                queries[task.query] = run_stage(
                    "rag.retrieve", 
                    {"repository": full_name, "task_id": task.task_id},
                    lambda: index.search(task.query),
                    lambda result: {
                        "mode": result.mode, 
                        "hit_count": len(result.hits),
                        "chunk_ids": [hit.chunk.chunk_id for hit in result.hits], 
                        "warnings": result.warnings
                    },
                )
            # 用 query 到检索结果的索引中寻找 evidence，而按照 task_id 表示不同任务，通过 query 复用检索结果，避免重复检索
            results[task.task_id] = queries[task.query]
        
        # 构建证据包
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
        
        pack.warnings = sorted(set(pack.warnings + self._warnings))
        
        # 如果嵌入器为 None 且快照中有文件，则添加字节预算分块警告
        if embedder is None and snapshot.files:
            pack.warnings.append("byte_budget_chunking")
        return pack

    def close(self):
        """
        缓存清空函数
        """
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
    
    # 如果没有现有的 ResearchSession，则创建一个新的 ResearchSession
    active = session or ResearchSession()
    token = _current.set(active)
    try:
        yield active
    finally:
        # 请求最后清理，释放缓存
        active.close()
        _current.reset(token)
