"""Reuse sanitized file chunks and passage vectors across immutable commits."""
import numpy as np

from .cache import digest, sanitizer_fingerprint
from .chunking import chunk_identity, chunk_snapshot, format_passage
from .schemas import ChunkingResult
from .semantic import parser_fingerprint


class IncrementalArtifacts:
    def __init__(self, store, config, embedder=None):
        self.store, self.config, self.embedder = store, config, embedder
        self.stats = {"chunk_file_hits": 0, "chunk_file_misses": 0,
                      "vector_hits": 0, "vector_misses": 0}
        self.warnings = []
        self._memory = {}
        self.policy = digest(["chunks-v2", parser_fingerprint(), sanitizer_fingerprint(),
                              config.chunk_tokens, config.overlap_tokens,
                              embedder.fingerprint if embedder else "byte-budget-v1"])

    def _read(self, key):
        if key in self._memory:
            return self._memory[key]
        try:
            value = self.store.get(key) if self.store else None
            if value is not None:
                self._memory[key] = value
            return value
        except Exception:
            self.warnings.append("persistent_cache_read_failed")
            return None

    def _write(self, key, data, kind):
        self._memory[key] = data
        try:
            if self.store and not self.store.put(key, data, kind=kind):
                self.warnings.append("persistent_cache_budget")
        except Exception:
            self.warnings.append("persistent_cache_write_failed")

    def chunks(self, snapshot, count):
        result = ChunkingResult()
        for file in snapshot.files:
            key = digest(["file", snapshot.full_name.casefold(), file.path, file.blob_sha,
                          digest(file.content), self.policy])
            cached = self._read(key)
            parsed = None
            if isinstance(cached, dict):
                try:
                    parsed = ChunkingResult.model_validate(cached)
                    lines = file.content.splitlines(keepends=True)
                    if any(c.path != file.path or c.text != "".join(lines[c.start_line - 1:c.end_line])
                           for c in parsed.chunks):
                        parsed = None
                except (ValueError, TypeError):
                    parsed = None
            if parsed is None:
                self.stats["chunk_file_misses"] += 1
                single = snapshot.model_copy(update={"files": [file]})
                parsed = chunk_snapshot(single, token_count=count, config=self.config)
                # No commit-specific citation identity is reused without rebinding below.
                self._write(key, parsed.model_dump(), "chunks")
            else:
                self.stats["chunk_file_hits"] += 1
            for chunk in parsed.chunks:
                result.chunks.append(chunk.model_copy(update={
                    "full_name": snapshot.full_name, "commit_sha": snapshot.commit_sha,
                    "chunk_id": chunk_identity(snapshot.full_name, snapshot.commit_sha, chunk.path,
                                               chunk.start_line, chunk.end_line, chunk.text),
                }))
            result.fallback_files.extend(parsed.fallback_files)
            result.oversized_lines.update(parsed.oversized_lines)
        return result

    def publish(self, snapshot):
        if not snapshot.files or snapshot.report.tree_truncated:
            # A failed/partial tree must not erase the last successfully indexed
            # manifest or turn temporarily unseen paths into deletion events.
            self.warnings.append("incomplete_snapshot_manifest_not_published")
            return {"added_files": 0, "changed_files": 0, "deleted_files": 0}
        manifest = {entry["path"]: entry["sha"] for entry in snapshot.available_files}
        if not manifest:
            manifest = {file.path: file.blob_sha for file in snapshot.files}
        key = digest(["manifest", snapshot.full_name.casefold(), self.policy])
        previous = self._read(key) or {}
        old = previous.get("files", {})
        # A truncated tree cannot establish that a previously indexed file was deleted.
        stats = {
            "added_files": len(manifest.keys() - old.keys()),
            "changed_files": sum(old[path] != sha for path, sha in manifest.items() if path in old),
            "deleted_files": len(old.keys() - manifest.keys()),
        }
        self._write(key, {"commit_sha": snapshot.commit_sha, "files": manifest,
                          "status": snapshot.report.status}, "manifest")
        return stats


class CachedEmbedder:
    def __init__(self, embedder, artifacts):
        self.inner, self.artifacts = embedder, artifacts
        self.fingerprint = embedder.fingerprint + ":" + artifacts.policy

    def embed_query(self, query):
        return self.inner.embed_query(query)

    def embed_documents(self, chunks):
        keys, rows, missing = [], {}, []
        for index, chunk in enumerate(chunks):
            key = digest(["vector", chunk.full_name.casefold(), self.fingerprint,
                          format_passage(chunk.path, chunk.symbol, chunk.text)])
            keys.append(key)
            value = self.artifacts._read(key)
            try:
                row = np.asarray(value, dtype=np.float32)
                valid = row.ndim == 1 and len(row) > 0 and np.isfinite(row).all() and np.linalg.norm(row) > 0
            except (ValueError, TypeError):
                valid = False
            if valid:
                rows[index] = row
                self.artifacts.stats["vector_hits"] += 1
            else:
                missing.append(index)
                self.artifacts.stats["vector_misses"] += 1
        if missing:
            values = np.asarray(self.inner.embed_documents([chunks[i] for i in missing]), dtype=np.float32)
            if (values.ndim != 2 or len(values) != len(missing) or not np.isfinite(values).all()
                    or (np.linalg.norm(values, axis=1) <= 0).any()):
                raise ValueError("invalid_document_vectors")
            for index, vector in zip(missing, values):
                rows[index] = vector
                self.artifacts._write(keys[index], vector.tolist(), "vector")
        return np.stack([rows[index] for index in range(len(chunks))])
