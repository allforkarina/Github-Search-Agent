"""Bounded single-user SQLite artifact store; no pickle, raw files or credentials.

Artifacts expire from creation time (reads do not extend retention). Each write
is committed atomically. A manifest is published only after an index is built.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from functools import wraps
from threading import RLock
from typing import Protocol


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class ArtifactStore(Protocol):
    def get(self, key: str): ...
    def put(self, key: str, value, *, kind: str): ...
    def close(self): ...


def _serialized(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return invoke


class SQLiteArtifactStore:
    def __init__(self, config, *, clock=time.time):
        self.config, self.clock = config, clock
        self._lock = RLock()
        directory = Path(config.cache_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "artifacts.sqlite3"
        # LangGraph can move a single request between worker threads. Serialize
        # this connection's transactions; this is not multi-user index serving.
        self.db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("CREATE TABLE IF NOT EXISTS artifacts ("
                        "key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,"
                        "checksum TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL)")
        self.db.commit()
        self.prune()

    @_serialized
    def get(self, key):
        row = self.db.execute("SELECT payload,checksum,expires FROM artifacts WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row[2] <= self.clock() or hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            with self.db:
                self.db.execute("DELETE FROM artifacts WHERE key=?", (key,))
            return None
        try:
            return json.loads(row[0])
        except (ValueError, TypeError):
            with self.db:
                self.db.execute("DELETE FROM artifacts WHERE key=?", (key,))
            return None

    @_serialized
    def put(self, key, value, *, kind):
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(payload.encode()) + 32768 > self.config.cache_max_bytes:
            return False
        now = self.clock()
        # Do not renew an identical artifact's absolute retention on every request.
        existing = self.get(key)
        if existing == value:
            return True
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?)", (
                key, kind, payload, hashlib.sha256(payload.encode()).hexdigest(),
                now, now + self.config.cache_ttl_s,
            ))
        self.prune()
        return self.get(key) is not None

    @_serialized
    def prune(self):
        with self.db:
            deleted = self.db.execute("DELETE FROM artifacts WHERE expires<=?", (self.clock(),)).rowcount
        if deleted or self.path.stat().st_size > self.config.cache_max_bytes:
            self.db.execute("VACUUM")
        while self.path.stat().st_size > self.config.cache_max_bytes:
            oldest = self.db.execute("SELECT key FROM artifacts ORDER BY created,key LIMIT 1").fetchone()
            if oldest is None:
                break
            with self.db:
                self.db.execute("DELETE FROM artifacts WHERE key=?", oldest)
            self.db.execute("VACUUM")

    @_serialized
    def clear(self):
        with self.db:
            self.db.execute("DELETE FROM artifacts")
        self.db.execute("VACUUM")

    @_serialized
    def close(self):
        self.prune()
        self.db.close()


def sanitizer_fingerprint():
    from . import security
    from .security import configured_secrets
    from reposcouts.observability import tracing
    # Policy implementation + a digest of environment secrets; never persist their values.
    return digest([Path(tracing.__file__).read_text(encoding="utf-8"),
                   Path(security.__file__).read_text(encoding="utf-8"),
                   sorted(hashlib.sha256(v.encode()).hexdigest() for v in configured_secrets())])


def main():
    from .config import RagConfig
    parser = argparse.ArgumentParser(description="Prune expired RAG artifacts or clear the local cache")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    store = SQLiteArtifactStore(RagConfig())
    if args.clear:
        store.clear()
    store.close()
    print("RAG cache cleared" if args.clear else "RAG cache pruned")


if __name__ == "__main__":
    main()
