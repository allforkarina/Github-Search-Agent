"""Frozen-corpus ablations. Human review is required before reporting quality."""
import time
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from dataclasses import asdict
from collections import Counter
from .context import add_symbol_context, source_kind

from .cache import digest, sanitizer_fingerprint
from .evaluation import retrieval_metrics, judgment_metrics, citation_precision
from .retrieval import HybridIndex
from .schemas import EvidenceChunk


def validate_manifest(manifest, *, reviewed=False):
    repositories = manifest["repositories"]
    tasks = manifest["tasks"]
    names = [repo["full_name"] for repo in repositories]
    if len(names) != len(set(names)) or len(tasks) != len({t["id"] for t in tasks}):
        raise ValueError("duplicate_repository_or_task")
    if len(repositories) != 5 or len(tasks) != 20:
        raise ValueError("benchmark_requires_five_repositories_twenty_tasks")
    mapping = {repo["full_name"]: repo for repo in repositories}
    for repo in repositories:
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", repo.get("commit_sha") or ""):
            raise ValueError("repository_commit_not_frozen")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo["full_name"]):
            raise ValueError("invalid_repository_name")
        if repo["split"] not in ("development", "test"):
            raise ValueError("invalid_repository_split")
    for task in tasks:
        if task["repository"] not in mapping:
            raise ValueError("foreign_task_repository")
        if reviewed:
            gold = task["gold"]
            if (not task.get("reviewed_by") or not task.get("reviewed_at")
                    or gold.get("status") not in ("satisfied", "not_satisfied", "uncertain")
                    or not isinstance(gold.get("relevance"), dict)
                    or not gold.get("rationale", "").strip()):
                raise ValueError("human_review_incomplete")
            if any(type(grade) is not int or grade not in (0, 1, 2) for grade in gold["relevance"].values()):
                raise ValueError("invalid_relevance_grade")
            # A non-satisfied judgment needs explicit counterevidence, not zero recall.
            if gold["status"] == "not_satisfied" and not gold.get("contradiction_ids"):
                raise ValueError("gold_contradiction_missing")
    if reviewed and (manifest.get("label_status") != "human_reviewed" or not manifest.get("frozen_at")):
        raise ValueError("gold_not_frozen")
    if reviewed and set(Counter(task["gold"]["status"] for task in tasks)) != {"satisfied", "not_satisfied", "uncertain"}:
        raise ValueError("benchmark_must_cover_all_three_judgments")
    if reviewed and manifest.get("annotation_coverage") != "exhaustive":
        if set(manifest.get("annotation_pool_modes", [])) != {"bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError("annotation_pool_must_cover_all_ablations")
    if reviewed:
        for task in tasks:
            gold = task["gold"]
            if not set(gold.get("contradiction_ids", [])) <= {key for key, grade in gold["relevance"].items() if grade > 0}:
                raise ValueError("gold_contradiction_not_relevant")
    return mapping


def score_benchmark(manifest, corpora, *, mode="bm25", split="test", config,
                    embedder=None, reranker=None, predictions=None, citation_review=None):
    mapping = validate_manifest(manifest, reviewed=True)
    if mode not in ("bm25", "hybrid", "hybrid_rerank"):
        raise ValueError("unknown_ablation")
    if mode != "bm25" and embedder is None:
        raise ValueError("hybrid_model_required")
    if mode == "hybrid_rerank" and reranker is None:
        raise ValueError("reranker_required")
    indexes, rows, labels = {}, [], []
    for task in manifest["tasks"]:
        repo = mapping[task["repository"]]
        if repo["split"] != split:
            continue
        corpus = corpora[task["repository"]]
        if repo["full_name"] not in indexes:
            chunks = validated_chunks(repo, corpus)
            indexes[repo["full_name"]] = HybridIndex(chunks, embedder=embedder, config=config)
        index = indexes[repo["full_name"]]
        known = {c.chunk_id for c in index.chunks}
        if not set(task["gold"]["relevance"]) <= known:
            raise ValueError("gold_chunk_not_in_frozen_corpus")
        started = time.monotonic()
        result = index.search_many([task["query"]], reranker=reranker if mode == "hybrid_rerank" else None)
        if mode != "bm25" and result.mode != "hybrid":
            raise ValueError("hybrid_ablation_degraded")
        if mode == "hybrid_rerank" and result.hits and not result.reranked:
            raise ValueError("rerank_ablation_degraded")
        result = add_symbol_context(index, result, config.evidence_top_k)
        ranking = [hit.chunk.chunk_id for hit in result.hits]
        rows.append({"id": task["id"], "mode": result.mode, "reranked": result.reranked,
                     "latency_s": time.monotonic() - started, "chunk_ids": ranking,
                     **retrieval_metrics(task["gold"]["relevance"], ranking, config.evidence_top_k)})
        labels.append(task["gold"]["status"])
    if not rows:
        raise ValueError("empty_benchmark_split")
    metrics = ("recall_at_k", "mrr_at_k", "ndcg_at_k", "hit_rate_at_k")
    report = {"manifest_digest": digest(manifest), "mode": mode, "split": split,
              "k": config.evidence_top_k, "rows": rows, "quality_verified_on_human_labels": True,
              "mean": {metric: _mean([row[metric] for row in rows]) for metric in metrics},
              "judgment": None, "citation_precision": None,
              "reproducibility": reproducibility(config),
              "latency_scope": "search, rerank and context; excludes index/model load",
              "annotation_pool_depth": manifest.get("annotation_pool_depth"),
              "notes": ["Unjudged chunks are treated as nonrelevant; report annotation-pool depth.",
                        "Small benchmark; no claim of statistical significance."]}
    if predictions is not None:
        by_id = {row["id"]: row for row in predictions}
        if len(by_id) != len(predictions) or any(row["id"] not in by_id for row in rows):
            raise ValueError("prediction_alignment_invalid")
        report["judgment"] = judgment_metrics(labels, [by_id[row["id"]]["status"] for row in rows])
        if citation_review is not None:
            if (citation_review.get("predictions_digest") != digest(predictions)
                    or not citation_review.get("reviewed_by") or not citation_review.get("reviewed_at")):
                raise ValueError("citation_review_not_bound_to_predictions")
            support = {(item["task_id"], item["evidence_id"]): item["supports"] for item in citation_review["labels"]}
            emitted = [(row["id"], key) for row in rows for key in dict.fromkeys(by_id[row["id"]].get("evidence_ids", []))]
            if any(pair not in support for pair in emitted):
                raise ValueError("citation_review_incomplete")
            report["citation_precision"] = citation_precision([support[pair] for pair in emitted])
    return report


def validated_chunks(repo, corpus):
    if (corpus["commit_sha"] != repo["commit_sha"] or corpus["full_name"] != repo["full_name"]
            or corpus.get("expires_at", 0) <= time.time()):
        raise ValueError("corpus_version_or_retention_invalid")
    if digest(corpus["chunks"]) != repo.get("corpus_digest"):
        raise ValueError("corpus_changed_after_annotation")
    if corpus.get("sanitizer_policy") != sanitizer_fingerprint():
        raise ValueError("corpus_sanitizer_changed_reprepare_and_review")
    chunks = [EvidenceChunk.model_validate(c) for c in corpus["chunks"]]
    if any(c.full_name != repo["full_name"] or c.commit_sha != repo["commit_sha"] for c in chunks):
        raise ValueError("foreign_corpus_chunk")
    return chunks


def reproducibility(config):
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("numpy", "torch", "sentence-transformers", "transformers", "rank-bm25",
                 "tree-sitter", "tree-sitter-java", "tree-sitter-javascript", "tree-sitter-typescript"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.rglob("*.py"))}
    return {"source_digest": digest(sources), "packages": packages, "config": asdict(config)}


def evaluate_frozen_task(task, corpus, index, result, config):
    """Explicit paid judgment entry point; no gold labels enter the LLM input."""
    from reposcouts.core.schemas import UserRequirements, RepositoryEvidence
    from .evidence import citation_for_chunk
    from .schemas import (EvidenceSnippet, RepositoryResearch, IngestionReport,
                          RequirementTask, RequirementEvidence)
    from .planning import task_identity
    from .security import sanitize_content
    from reposcouts.stages.repository_evaluator import evaluate_repository

    requirement = task["query"]
    result = add_symbol_context(index, result, config.evidence_top_k)
    snippets, chars = [], 0
    for hit in result.hits:
        chunk = hit.chunk
        if chunk.full_name != corpus["full_name"] or chunk.commit_sha != corpus["commit_sha"]:
            raise ValueError("foreign_judgment_evidence")
        safe = sanitize_content(chunk.text)
        snippet = EvidenceSnippet(citation=citation_for_chunk(chunk), text=safe,
                                   redacted=safe != chunk.text, source_kind=source_kind(chunk.path))
        size = len(snippet.model_dump_json())
        if chars + size > config.max_evidence_chars or len(snippets) >= config.max_evidence_chunks:
            continue
        snippets.append(snippet)
        chars += size
    retrieval_task = RequirementTask(task_id=task_identity("must_have", requirement, requirement),
        requirement=requirement, parent_requirement=requirement, kind="must_have", query=requirement)
    pack = RepositoryResearch(full_name=corpus["full_name"], commit_sha=corpus["commit_sha"],
        ingestion=IngestionReport.model_validate(corpus["ingestion"]), snippets=snippets,
        requirements=[RequirementEvidence(task=retrieval_task, mode=result.mode,
                         evidence_ids=[s.citation.evidence_id for s in snippets], warnings=result.warnings)])
    request = UserRequirements(topic="repository capability verification", language=[], technology_stack=[],
        difficulty="any", must_have=[requirement], nice_to_have=[], exclusions=[], learning_goal="integration",
        time_budget_in_hours=None)
    evidence = RepositoryEvidence(full_name=corpus["full_name"], metadata={}, readme="", structure={}, research=pack)
    check = evaluate_repository(request, evidence).must_have_checks[0]
    return {"id": task["id"], "status": check.status, "evidence_ids": check.evidence_ids,
            "implementation_level": check.implementation_level, "evidence": check.evidence,
            "citations": [c.model_dump() for c in check.citations]}


def invoke_frozen_judgment(task, corpus, index, result, config, *, tracing=False,
                           project_name="reposcouts-rag-evaluation-v2"):
    """Optional sanitized LangSmith trace; never inherit an implicit global client."""
    from langsmith import tracing_context
    if not tracing:
        with tracing_context(enabled=False):
            return evaluate_frozen_task(task, corpus, index, result, config)
    import os
    import uuid
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tracers.langchain import LangChainTracer
    from reposcouts.observability.tracing import build_trace_client
    from reposcouts.observability.metadata import build_run_metadata
    from .security import configured_secrets

    client = build_trace_client(secrets=configured_secrets(), max_string_chars=4000)
    root_id = uuid.uuid4()
    project = project_name
    outputs = []

    def execute(_):
        value = evaluate_frozen_task(task, corpus, index, result, config)
        outputs.append(value)
        return {"task_id": value["id"], "status": value["status"], "evidence_ids": value["evidence_ids"]}

    metadata = build_run_metadata(run_id=str(root_id), environment="benchmark",
        agent_version="repository-research-v2", model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        prompt_version="research-v2")
    try:
        with tracing_context(enabled=True, client=client, project_name=project):
            RunnableLambda(execute, name="RepositoryResearchBenchmark").invoke(
                {"task_id": task["id"], "commit_sha": corpus["commit_sha"]},
                config={"run_id": root_id, "metadata": metadata,
                        "callbacks": [LangChainTracer(client=client, project_name=project)]},
            )
        return {**outputs[0], "trace_id": str(root_id)}
    finally:
        client.close(timeout=5)


def _mean(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
