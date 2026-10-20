"""Explicit opt-in model download: python -m reposcouts.rag.download_model."""
from dataclasses import replace
import argparse

from .config import RagConfig
from .embeddings import LocalEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("embedding", "reranker", "all"), default="embedding")
    args = parser.parse_args()
    config = replace(RagConfig(), local_files_only=False)
    if args.kind in ("embedding", "all"):
        embedder = LocalEmbedder(config)
        print(f"Model cached: {embedder.fingerprint}")
    if args.kind in ("reranker", "all"):
        from .reranking import LocalReranker
        reranker = LocalReranker(config)
        print(f"Model cached: {reranker.fingerprint}")


if __name__ == "__main__":
    main()
