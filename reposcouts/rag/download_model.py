"""Explicit opt-in model download: python -m reposcouts.rag.download_model."""
from dataclasses import replace

from .config import RagConfig
from .embeddings import LocalEmbedder


def main():
    config = replace(RagConfig(), local_files_only=False)
    embedder = LocalEmbedder(config)
    print(f"Model cached: {embedder.fingerprint}")


if __name__ == "__main__":
    main()
