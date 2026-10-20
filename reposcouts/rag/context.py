"""Bounded static context hints; these do not prove runtime behavior."""
from pathlib import PurePosixPath
import re

from .schemas import RetrievalHit


def source_kind(path):
    parts = [p.lower() for p in PurePosixPath(path).parts]
    name = PurePosixPath(path).name
    if (any(p in ("test", "tests", "__tests__") for p in parts[:-1])
            or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")
            or re.search(r"\.(?:test|spec)\.[jt]sx?$", parts[-1])
            or name.endswith(("Test.java", "Tests.java"))):
        return "test"
    directories = parts[:-1]
    # com/example under a Maven source root is a Java package, not an examples folder.
    for index in range(len(parts) - 2):
        if parts[index:index + 3] == ["src", "main", "java"]:
            directories = parts[:index]
            break
    if any(re.search(r"(?:^|[-_])(?:examples?|tutorials?|demos?)(?:$|[-_])", p) for p in directories):
        return "example"
    if (parts[-1] in ("pom.xml", "build.gradle", "build.gradle.kts", "setup.py", "dockerfile")
            or parts[-1].startswith("requirements") and parts[-1].endswith(".txt")):
        return "config"
    if PurePosixPath(path).suffix.lower() in (".md", ".mdx", ".rst", ".txt"):
        return "document"
    if PurePosixPath(path).suffix.lower() in (".toml", ".json", ".yaml", ".yml", ".cfg", ".ini", ".xml", ".gradle", ".properties"):
        return "config"
    return "source"


def add_symbol_context(index, result, limit):
    symbols = {h.chunk.symbol.rsplit(".", 1)[-1] for h in result.hits if h.chunk.symbol}
    symbols = {s for s in symbols if re.fullmatch(r"[A-Za-z_$][\w$]*", s)}
    if not symbols:
        return result
    selected = {h.chunk.chunk_id for h in result.hits}
    contexts = []
    for chunk in index.chunks:
        if chunk.chunk_id in selected or source_kind(chunk.path) not in ("source", "example"):
            continue
        lines = [line for line in chunk.text.splitlines()
                 if not re.match(r"\s*(?:async\s+)?(?:def|class)\s", line)]
        if any(re.search(r"(?<![\w$])" + re.escape(s) + r"\s*\(", "\n".join(lines)) for s in symbols):
            contexts.append(RetrievalHit(chunk=chunk.model_copy(deep=True), rrf_score=0))
    if not contexts:
        return result
    # Reserve at most two slots; preserve the strongest retrieved evidence.
    contexts = sorted(contexts, key=lambda h: (source_kind(h.chunk.path) != "source", h.chunk.path, h.chunk.start_line))[:min(2, max(0, limit - 1))]
    return result.model_copy(update={"hits": result.hits[:limit - len(contexts)] + contexts})


def related_paths(snapshot, queries, hits):
    """Score unfetched paths using requirements, nearby files and local imports."""
    from .retrieval import lexical_tokens
    terms = set(lexical_tokens(" ".join(queries)))
    imports = []
    for file in snapshot.files:
        for module in re.findall(r"(?:from\s+|import\s+)([\w.]+)", file.content):
            imports.append(module.replace(".", "/"))
        imports.extend(re.findall(r"(?:from\s*|require\s*\()\s*['\"]([^'\"]+)['\"]", file.content))
    anchors = {str(PurePosixPath(h.chunk.path).parent) for h in hits}
    attempted = set(snapshot.report.attempted_paths) | {file.path for file in snapshot.files}
    ranked = []
    for entry in snapshot.available_files:
        path = entry["path"]
        if path in attempted:
            continue
        stem = str(PurePosixPath(path).with_suffix(""))
        overlap = len(set(lexical_tokens(path)) & terms)
        imported = any(part and (stem.endswith(part.strip("./")) or part.endswith(PurePosixPath(stem).name))
                       for part in imports)
        nearby = str(PurePosixPath(path).parent) in anchors
        entrypoint = PurePosixPath(path).stem.lower() in ("main", "app", "server", "graph", "cli", "application")
        score = 8 * imported + 4 * overlap + 2 * nearby + entrypoint
        if score:
            ranked.append((-score, path))
    return [path for _, path in sorted(ranked)]
