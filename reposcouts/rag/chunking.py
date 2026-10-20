"""Partition original source lines; token counting is injected by the embedder."""
import ast
import hashlib
import re
from collections.abc import Callable
from pathlib import PurePosixPath

from .config import RagConfig
from .schemas import ChunkingResult, EvidenceChunk, RepositorySnapshot
from .semantic import semantic_sections


def chunk_identity(full_name, commit_sha, path, start, end, text):
    identity = "\0".join((full_name, commit_sha, path, str(start), str(end), text))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def format_passage(path: str, symbol: str | None, text: str) -> str:
    return f"passage: {path}\n{symbol or ''}\n{text}"


def _python_sections(text, line_count):
    tree = ast.parse(text)
    spans = []

    def visit(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = f"{prefix}.{child.name}" if prefix else child.name
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                spans.append((start, child.end_lineno, symbol))
                visit(child, symbol)

    visit(tree)
    boundaries = {1, line_count + 1}
    for start, end, _ in spans:
        boundaries.update((start, end + 1))

    points = sorted(boundaries)
    sections = []
    for start, next_start in zip(points, points[1:]):
        active = [span for span in spans if span[0] <= start <= span[1]]
        symbol = min(active, key=lambda span: span[1] - span[0])[2] if active else None
        sections.append((start, next_start - 1, symbol))
    return sections


def _markdown_sections(lines):
    starts = [(1, None)]
    headings = []
    fence = None
    for number, line in enumerate(lines, 1):
        if fence is not None:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*\r?\n?", line):
                fence = None
            continue
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            fence = (marker[0], len(marker))
            continue
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
        heading_start = number
        if match:
            level = len(match.group(1))
            title = re.sub(r"[ \t]+#+$", "", match.group(2))
        elif (number > 1 and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*\r?\n?", line)
              and lines[number - 2].strip()
              and not re.match(r"\s*(?:[#>`~]|[-*+]\s)", lines[number - 2])):
            heading_start = number - 1
            level = 1 if line.lstrip().startswith("=") else 2
            title = lines[number - 2].strip()
        else:
            continue
        if title:
            headings = [(depth, title_) for depth, title_ in headings if depth < level]
            headings.append((level, title))
            symbol = " > ".join(title_ for _, title_ in headings)
            if heading_start == 1:
                starts[0] = (1, symbol)
            else:
                starts.append((heading_start, symbol))
    return [
        (start, starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines), symbol)
        for index, (start, symbol) in enumerate(starts)
    ]


def chunk_snapshot(
    snapshot: RepositorySnapshot,
    *,
    token_count: Callable[[str], int],
    config: RagConfig | None = None,
) -> ChunkingResult:
    config = config or RagConfig()
    result = ChunkingResult()
    if not snapshot.files:
        return result
    if snapshot.commit_sha is None:
        raise ValueError("a snapshot with files must have a commit SHA")

    for file in snapshot.files:
        # Keep line terminators: no strip(), normalization or reconstruction of source.
        lines = file.content.splitlines(keepends=True)
        if not lines:
            continue

        suffix = PurePosixPath(file.path).suffix.lower()
        if suffix == ".py":
            try:
                sections = _python_sections(file.content, len(lines))
            except (SyntaxError, ValueError, RecursionError):
                result.fallback_files.append(file.path)
                sections = [(1, len(lines), None)]

        elif suffix in (".md", ".mdx"):
            sections = _markdown_sections(lines)

        elif suffix in (".js", ".jsx", ".ts", ".tsx", ".java"):
            try:
                sections = semantic_sections(file.content, suffix, len(lines))
            except (ImportError, ValueError, TypeError, RecursionError):
                result.fallback_files.append(file.path)
                sections = [(1, len(lines), None)]

        else:
            sections = [(1, len(lines), None)]

        for first, last, symbol in sections:
            start = first - 1
            while start < last:
                end = start
                while end < last:
                    # parse line by line.
                    passage = format_passage(file.path, symbol, "".join(lines[start:end + 1]))
                    if token_count(passage) > config.chunk_tokens:
                        break
                    end += 1

                if end == start:
                    # A single long line cannot be shortened while claiming a full-line citation.
                    result.oversized_lines.setdefault(file.path, []).append(start + 1)
                    # start next round from new line.
                    start += 1
                    continue
                text = "".join(lines[start:end])
                if text.strip():
                    # text identity with name and sha.
                    result.chunks.append(EvidenceChunk(
                        chunk_id=chunk_identity(snapshot.full_name, snapshot.commit_sha, file.path,
                                                start + 1, end, text),
                        full_name=snapshot.full_name, commit_sha=snapshot.commit_sha,
                        path=file.path, symbol=symbol,
                        start_line=start + 1, end_line=end, text=text,
                    ))
                if end == last:
                    break
                next_start = end
                for overlap_start in range(end - 1, start, -1):
                    if token_count("".join(lines[overlap_start:end])) > config.overlap_tokens:
                        break
                    next_start = overlap_start
                start = next_start
    return result
