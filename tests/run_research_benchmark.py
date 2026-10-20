"""Prepare review material or score a frozen, human-reviewed repository benchmark.

Preparation calls GitHub only. Scoring is local-only and never calls an LLM.
The original ten-request end-to-end baseline is not modified.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reposcouts.rag.benchmark import validate_manifest, score_benchmark, validated_chunks, invoke_frozen_judgment
from reposcouts.rag.cache import digest, sanitizer_fingerprint
from reposcouts.rag.chunking import chunk_snapshot
from reposcouts.rag.config import RagConfig
from reposcouts.rag.context import related_paths
from reposcouts.rag.github_source import collect_repository_snapshot, expand_repository_snapshot
from reposcouts.rag.security import sanitize_content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("rag") / "benchmark_v2.json")
    parser.add_argument("--corpus-dir", type=Path, default=Path("tests/artifacts/research-benchmark-v2"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Explicitly call the configured cloud Evaluator LLM")
    parser.add_argument("--tracing", action="store_true", help="Trace --judge with the project's sanitized LangSmith client")
    parser.add_argument("--project", default="reposcouts-rag-evaluation-v2", help="Separate LangSmith evaluation project")
    parser.add_argument("--mode", choices=("bm25", "hybrid", "hybrid_rerank"), default="bm25")
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--citation-review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare and args.judge:
        parser.error("--prepare and --judge are separate steps")
    if args.judge and args.output is None:
        parser.error("--judge requires --output to preserve predictions")
    if args.tracing and not args.judge:
        parser.error("--tracing applies to --judge only")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    config = replace(RagConfig(), cache_enabled=False, reranker_enabled=args.mode == "hybrid_rerank")
    if args.prepare:
        args.corpus_dir.mkdir(parents=True, exist_ok=True)
        pool_config = replace(config, evidence_top_k=20, rerank_candidates=20)
        pool_embedder = pool_reranker = None
        if args.mode != "bm25":
            from reposcouts.rag.embeddings import LocalEmbedder
            pool_embedder = LocalEmbedder(pool_config)
        if args.mode == "hybrid_rerank":
            from reposcouts.rag.reranking import LocalReranker
            pool_reranker = LocalReranker(pool_config)
        pool_modes = ["bm25"] + (["hybrid"] if pool_embedder else []) + (["hybrid_rerank"] if pool_reranker else [])
        pools = []
        for repo in manifest["repositories"]:
            queries = [task["query"] for task in manifest["tasks"] if task["repository"] == repo["full_name"]]
            snapshot = collect_repository_snapshot(repo["full_name"], config, queries=queries, commit_sha=repo["commit_sha"])
            if not snapshot.files:
                raise SystemExit("Corpus collection failed: " + repo["full_name"])
            for _ in range(config.max_expansion_rounds):
                paths = related_paths(snapshot, queries, [])
                if not paths or not expand_repository_snapshot(snapshot, paths, config):
                    break
            for file in snapshot.files:
                file.content = sanitize_content(file.content)
            # Freeze one conservative byte-bounded corpus for ALL ablations.
            # Changing chunking/tokenizer requires a new corpus digest and review.
            chunks = chunk_snapshot(snapshot, token_count=lambda text: len(text.encode()) + 2, config=config)
            serialized = [c.model_dump() for c in chunks.chunks]
            artifact = {"full_name": repo["full_name"], "commit_sha": repo["commit_sha"],
                        "created_at": time.time(), "expires_at": time.time() + config.cache_ttl_s,
                        "chunking": "semantic-v2-byte-budget", "ingestion": snapshot.report.model_dump(),
                        "sanitizer_policy": sanitizer_fingerprint(),
                        "chunks": serialized}
            path = args.corpus_dir / (repo["full_name"].replace("/", "__") + ".json")
            path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if repo.get("corpus_digest") not in (None, digest(serialized)):
                raise SystemExit("Corpus changed; create a new benchmark version and review labels")
            repo["corpus_digest"] = digest(serialized)
            from reposcouts.rag.retrieval import HybridIndex
            from reposcouts.rag.context import add_symbol_context
            from reposcouts.rag.evidence import citation_for_chunk
            sparse = HybridIndex(chunks.chunks, config=pool_config)
            hybrid = HybridIndex(chunks.chunks, embedder=pool_embedder, config=pool_config) if pool_embedder else None
            for task in (t for t in manifest["tasks"] if t["repository"] == repo["full_name"]):
                candidates, branches = {}, {}
                for mode in pool_modes:
                    index = sparse if mode == "bm25" else hybrid
                    result = index.search_many([task["query"]], reranker=pool_reranker if mode == "hybrid_rerank" else None)
                    if mode != "bm25" and result.mode != "hybrid":
                        raise SystemExit("Hybrid annotation pool degraded")
                    if mode == "hybrid_rerank" and result.hits and not result.reranked:
                        raise SystemExit("Rerank annotation pool degraded")
                    result = add_symbol_context(index, result, pool_config.evidence_top_k)
                    branches[mode] = [h.chunk.chunk_id for h in result.hits]
                    for hit in result.hits:
                        candidates[hit.chunk.chunk_id] = {"citation": citation_for_chunk(hit.chunk).model_dump(), "text": hit.chunk.text}
                pools.append({"task_id": task["id"], "query": task["query"], "branches": branches,
                              "candidates": candidates})
        manifest["label_status"] = "human_review_pending"
        manifest["annotation_pool_modes"] = pool_modes
        manifest["annotation_coverage"] = "pooled"
        review = args.corpus_dir / "manifest-for-review.json"
        review.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.corpus_dir / "annotation-pools.json").write_text(json.dumps(pools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Prepared sanitized chunks and pending labels: " + str(review.resolve()))
        return
    # Check labels before attempting any model load.
    if not args.judge:
        validate_manifest(manifest, reviewed=True)
    embedder = reranker = None
    if args.mode != "bm25":
        from reposcouts.rag.embeddings import LocalEmbedder
        embedder = LocalEmbedder(config)
    if args.mode == "hybrid_rerank":
        from reposcouts.rag.reranking import LocalReranker
        reranker = LocalReranker(config)
    corpora = {repo["full_name"]: json.loads((args.corpus_dir / (repo["full_name"].replace("/", "__") + ".json")).read_text(encoding="utf-8"))
               for repo in manifest["repositories"] if repo["split"] == args.split}
    read_optional = lambda path: json.loads(path.read_text(encoding="utf-8")) if path else None
    if args.judge:
        from reposcouts.rag.retrieval import HybridIndex
        indexes, report = {}, []
        for task in manifest["tasks"]:
            if task["repository"] not in corpora:
                continue
            repo = next(r for r in manifest["repositories"] if r["full_name"] == task["repository"])
            corpus = corpora[repo["full_name"]]
            if repo["full_name"] not in indexes:
                indexes[repo["full_name"]] = HybridIndex(validated_chunks(repo, corpus), embedder=embedder, config=config)
            index = indexes[repo["full_name"]]
            result = index.search_many([task["query"]], reranker=reranker)
            if args.mode != "bm25" and result.mode != "hybrid":
                raise SystemExit("Hybrid judgment mode degraded; no quality comparison allowed")
            if args.mode == "hybrid_rerank" and result.hits and not result.reranked:
                raise SystemExit("Reranker unavailable; no quality comparison allowed")
            report.append(invoke_frozen_judgment(task, corpus, index, result, config,
                                                 tracing=args.tracing, project_name=args.project))
    else:
        report = score_benchmark(manifest, corpora, mode=args.mode, split=args.split, config=config,
            embedder=embedder, reranker=reranker, predictions=read_optional(args.predictions),
            citation_review=read_optional(args.citation_review))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
