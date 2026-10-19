"""Small offline sanity benchmark. Hybrid mode only uses locally cached weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reposcouts.rag.chunking import chunk_snapshot
from reposcouts.rag.config import RagConfig
from reposcouts.rag.embeddings import LocalEmbedder
from reposcouts.rag.evaluation import retrieval_metrics
from reposcouts.rag.retrieval import HybridIndex
from reposcouts.rag.schemas import RepositoryFile, RepositorySnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name("rag").joinpath("benchmark_v1.json").read_text(encoding="utf-8"))
    config = RagConfig()
    embedder = LocalEmbedder(config) if args.mode == "hybrid" else None
    snapshot = RepositorySnapshot(
        full_name=data["full_name"], commit_sha=data["commit_sha"],
        files=[RepositoryFile(**file, blob_sha="b" * 40) for file in data["files"]],
    )
    count = embedder.token_count if embedder else lambda text: len(text.encode("utf-8")) + 2
    chunks = chunk_snapshot(snapshot, token_count=count, config=config)
    index = HybridIndex(chunks.chunks, embedder=embedder, config=config)
    rows = []
    for query in data["queries"]:
        result = index.search(query["query"])
        paths = list(dict.fromkeys(hit.chunk.path for hit in result.hits))
        rows.append({"id": query["id"], "mode": result.mode, "files": paths,
                     **retrieval_metrics(query["relevance"], paths, k=5)})
    metrics = ("recall_at_k", "mrr_at_k", "ndcg_at_k", "hit_rate_at_k")
    report = {"benchmark": data["version"], "label_status": data["label_status"], "unit": data["unit"],
              "requested_mode": args.mode, "actual_modes": sorted({row["mode"] for row in rows}),
              "k": 5, "rows": rows,
              "mean": {metric: sum(row[metric] for row in rows) / len(rows) for metric in metrics}}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.mode == "hybrid" and any(row["mode"] != "hybrid" for row in rows):
        raise SystemExit("Hybrid verification failed: at least one query fell back to BM25")


if __name__ == "__main__":
    main()
