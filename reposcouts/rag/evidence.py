"""Build a bounded, sanitized evidence pack with program-generated citations."""
from urllib.parse import quote

from .config import RagConfig
from .schemas import (
    ChunkingResult, EvidenceCitation, EvidenceSnippet, RepositoryResearch,
    RepositorySnapshot, RequirementEvidence, RequirementTask, RetrievalResult,
)
from .security import sanitize_content


def build_evidence(
    snapshot: RepositorySnapshot, 
    tasks: list[RequirementTask],
    results: dict[str, RetrievalResult], 
    chunked: ChunkingResult,
    *, 
    config: RagConfig, 
    index_reused: bool = False,
) -> RepositoryResearch:
    
    pack = RepositoryResearch(
        full_name=snapshot.full_name, 
        commit_sha=snapshot.commit_sha,
        ingestion=snapshot.report.model_copy(deep=True), 
        index_reused=index_reused,
        embedding_model=config.embedding_model, embedding_revision=config.embedding_revision,
    )
    # {path: splited content} 对 snapshot 中每一个文件内容进行行分割
    raw_lines = {file.path: file.content.splitlines(keepends=True) for file in snapshot.files}
    # 同理，加入敏感内容的处理
    safe_lines = {file.path: sanitize_content(file.content).splitlines(keepends=True) for file in snapshot.files}
    
    snippets = {}
    chars = 0
    for rank in range(config.evidence_top_k):
        # 用户要求拆分出的检索任务
        for task in tasks:
            # 提取每个任务的检索结果 top_k_evidence
            result = results[task.task_id]
            if rank >= len(result.hits):
                continue
            
            # 检索结果中的 chunk
            chunk = result.hits[rank].chunk
            evidence_id = "E" + chunk.chunk_id
            if evidence_id in snippets:
                # 如果当前这个 chunk 已经被别的任务检索了 跳过
                continue
            
            # 通过原始内容 raw_lines 进行比对校验
            original = "".join(raw_lines.get(chunk.path, [])[chunk.start_line - 1:chunk.end_line])
            if (
                chunk.full_name != snapshot.full_name or 
                chunk.commit_sha != snapshot.commit_sha or
                original != chunk.text or not original
            ):
                pack.warnings.append("chunk_source_mismatch")
                continue
            
            safe = "".join(safe_lines[chunk.path][chunk.start_line - 1:chunk.end_line])
            citation = EvidenceCitation(
                evidence_id=evidence_id, 
                full_name=chunk.full_name, 
                commit_sha=chunk.commit_sha,
                path=chunk.path, 
                symbol=chunk.symbol, 
                start_line=chunk.start_line, 
                end_line=chunk.end_line,
                url=(f"https://github.com/{quote(chunk.full_name, safe='/')}/blob/"
                     f"{chunk.commit_sha}/{quote(chunk.path, safe='/')}#L{chunk.start_line}-L{chunk.end_line}"),
            )
            snippet = EvidenceSnippet(
                citation=citation, 
                text=safe, 
                redacted=safe != chunk.text
            )
            size = len(snippet.model_dump_json())
            if len(snippets) >= config.max_evidence_chunks or chars + size > config.max_evidence_chars:
                pack.warnings.append("evidence_budget")
                continue
            
            snippets[evidence_id] = snippet
            chars += size
            
    pack.snippets = list(snippets.values())
    for task in tasks:
        result = results[task.task_id]
        ids = ["E" + hit.chunk.chunk_id for hit in result.hits if "E" + hit.chunk.chunk_id in snippets]
        # 将 requirement 与 evidence 关联起来
        pack.requirements.append(
            RequirementEvidence(
                task=task, 
                mode=result.mode, 
                evidence_ids=list(dict.fromkeys(ids)), 
                warnings=result.warnings,
            )
        )
    if chunked.fallback_files:
        pack.warnings.append("python_parser_fallback")
    if chunked.oversized_lines:
        pack.warnings.append("oversized_source_lines_skipped")
    pack.warnings = sorted(set(pack.warnings + snapshot.report.warnings))
    return pack
