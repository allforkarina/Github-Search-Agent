"""Offline ingestion and chunking tests for the actual application modules."""
import base64
import hashlib
import importlib
import json
import unittest
from unittest.mock import Mock, patch

import requests
from pydantic import ValidationError


PACKAGE = "reposcouts.rag"
source = importlib.import_module(PACKAGE + ".github_source")
chunking = importlib.import_module(PACKAGE + ".chunking")
schemas = importlib.import_module(PACKAGE + ".schemas")
RagConfig = importlib.import_module(PACKAGE + ".config").RagConfig
COMMIT = "a" * 40
TREE = "b" * 40


class Response:
    def __init__(self, data=None, status=200, raw=None):
        self.status_code = status
        self.raw = raw if raw is not None else json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        for index in range(0, len(self.raw), chunk_size):
            yield self.raw[index:index + chunk_size]


def file_entry(path, text, **changes):
    raw = text.encode() if isinstance(text, str) else text
    sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    entry = dict(path=path, type="blob", mode="100644", sha=sha, size=len(raw))
    return entry | changes, dict(content=base64.b64encode(raw).decode(), encoding="base64")


def session_for(files, *, truncated=False):
    entries = [entry for entry, _ in files]
    blobs = {entry["sha"]: data for entry, data in files}
    session = Mock()

    def get(url, **kwargs):
        if url.endswith("/commits"):
            return Response([dict(sha=COMMIT, commit=dict(tree=dict(sha=TREE)))])
        if url.endswith("/git/trees/" + TREE):
            return Response(dict(tree=entries, truncated=truncated))
        return Response(blobs[url.rsplit("/", 1)[-1]])

    session.get.side_effect = get
    return session


def snapshot(text, path="src/main.py", sha=COMMIT):
    return schemas.RepositorySnapshot(
        full_name="owner/repo", commit_sha=sha,
        files=[schemas.RepositoryFile(path=path, blob_sha="c" * 40, content=text)],
    )


class ContractTests(unittest.TestCase):
    def test_actual_evidence_chunk_contract(self):
        from reposcouts.rag.schemas import EvidenceChunk
        data = dict(chunk_id="id", full_name="owner/repo", commit_sha=COMMIT,
                    path="main.py", start_line=2, end_line=3, text="x\ny\n")
        EvidenceChunk(**data)
        for change in ({"path": "../secret"}, {"end_line": 1},
                       {"commit_sha": "main"}, {"text": "one line"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                EvidenceChunk(**(data | change))

    def test_config_rejects_invalid_budgets(self):
        for changes in ({"max_files": 0}, {"repository_timeout_s": 0},
                        {"overlap_tokens": 400}, {"overlap_tokens": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                RagConfig(**changes)


class SourceTests(unittest.TestCase):
    def test_reads_tree_and_blobs_from_one_pinned_commit(self):
        files = [file_entry("README.md", "# Intro\n"), file_entry("src/main.py", "x = 1\n")]
        session = session_for(files)
        result = source.collect_repository_snapshot("owner/repo", session=session)
        self.assertEqual(result.commit_sha, COMMIT)
        self.assertEqual(result.report.status, "ok")
        self.assertEqual(result.report.loaded_files, 2)
        self.assertEqual(result.files[0].content, "# Intro\n")
        urls = [call.args[0] for call in session.get.call_args_list]
        self.assertIn("/git/trees/" + TREE, urls[1])
        for entry, _ in files:
            self.assertTrue(any(url.endswith("/git/blobs/" + entry["sha"]) for url in urls))
        for call in session.get.call_args_list:
            self.assertFalse(call.kwargs["allow_redirects"])
            self.assertTrue(call.kwargs["stream"])
        session.close.assert_not_called()

    def test_selection_priority_is_deterministic_and_count_bounded(self):
        files = [file_entry("tests/test_a.py", "test\n"),
                 file_entry("src/a.py", "source\n"), file_entry("README.md", "readme\n")]
        first = source.collect_repository_snapshot(
            "owner/repo", RagConfig(max_files=1), session=session_for(files),
        )
        second = source.collect_repository_snapshot(
            "owner/repo", RagConfig(max_files=1), session=session_for(list(reversed(files))),
        )
        self.assertEqual([f.path for f in first.files], ["README.md"])
        self.assertEqual(first.files, second.files)
        self.assertEqual(first.report.skipped_counts["file_count_budget"], 2)
        self.assertEqual(first.report.status, "partial")

    def test_file_and_repository_byte_limits(self):
        files = [file_entry("README.md", "12345678"), file_entry("src/a.py", "1234"),
                 file_entry("src/b.py", "5678")]
        result = source.collect_repository_snapshot(
            "owner/repo", RagConfig(max_file_bytes=6, max_repository_bytes=6),
            session=session_for(files),
        )
        self.assertEqual([f.path for f in result.files], ["src/a.py"])
        self.assertEqual(result.report.fetched_bytes, 4)
        self.assertEqual(result.report.skipped_counts["file_size_budget"], 1)
        self.assertEqual(result.report.skipped_counts["repository_size_budget"], 1)

    def test_filters_secrets_generated_files_symlinks_and_unsafe_paths(self):
        paths = [".env", ".env.example", "node_modules/index.ts", "dist/a.js",
                 "src/generated_pb2.py", "package-lock.json", "private.pem", "../secret.py"]
        files = [file_entry(path, "secret") for path in paths]
        files += [file_entry("link.py", "target", mode="120000"), file_entry("src/main.py", "ok")]
        session = session_for(files)
        result = source.collect_repository_snapshot("owner/repo", session=session)
        self.assertEqual([f.path for f in result.files], ["src/main.py"])
        self.assertEqual(session.get.call_count, 3)

    def test_truncated_tree_is_partial(self):
        result = source.collect_repository_snapshot(
            "owner/repo", session=session_for([file_entry("main.py", "ok")], truncated=True),
        )
        self.assertTrue(result.report.tree_truncated)
        self.assertEqual(result.report.status, "partial")

    def test_tampered_blob_is_not_accepted(self):
        entry, data = file_entry("main.py", "real")
        data["content"] = base64.b64encode(b"fake").decode()
        result = source.collect_repository_snapshot("owner/repo", session=session_for([(entry, data)]))
        self.assertEqual(result.files, [])
        self.assertIn("blob_hash_mismatch", result.report.warnings)

    def test_wrong_blob_size_is_rejected(self):
        entry, data = file_entry("main.py", "real", size=2)
        result = source.collect_repository_snapshot("owner/repo", session=session_for([(entry, data)]))
        self.assertIn("blob_size_mismatch", result.report.warnings)

    def test_binary_and_non_utf8_are_skipped_and_reported(self):
        result = source.collect_repository_snapshot("owner/repo", session=session_for([
            file_entry("a.py", b"x\0y"), file_entry("b.py", b"\xff"), file_entry("c.py", "ok"),
        ]))
        self.assertEqual([f.path for f in result.files], ["c.py"])
        self.assertEqual(result.report.skipped_counts["binary"], 1)
        self.assertEqual(result.report.skipped_counts["non_utf8"], 1)
        self.assertEqual(result.report.status, "partial")

    def test_network_errors_do_not_copy_raw_secrets(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("Authorization: Bearer private-value")
        result = source.collect_repository_snapshot("owner/repo", session=session)
        self.assertEqual(result.report.status, "failed")
        self.assertEqual(result.report.warnings, ["network_error"])
        self.assertNotIn("private-value", result.model_dump_json())

    def test_failed_attempts_also_consume_file_budget(self):
        session = session_for([file_entry("a.py", "a"), file_entry("b.py", "b")])
        delegate = session.get.side_effect
        session.get.side_effect = lambda url, **kwargs: (
            Response(status=404) if "/git/blobs/" in url else delegate(url, **kwargs)
        )
        result = source.collect_repository_snapshot("owner/repo", RagConfig(max_files=1), session=session)
        self.assertEqual(session.get.call_count, 3)
        self.assertEqual(result.report.selected_files, 1)
        self.assertEqual(result.report.failed_paths, ["a.py"])

    def test_failed_file_keeps_successful_files_and_marks_partial(self):
        first = file_entry("a.py", "available")
        second = file_entry("b.py", "missing")
        session = session_for([first, second])
        delegate = session.get.side_effect
        session.get.side_effect = lambda url, **kwargs: (
            Response(status=404) if url.endswith(second[0]["sha"]) else delegate(url, **kwargs)
        )
        result = source.collect_repository_snapshot("owner/repo", session=session)
        self.assertEqual([file.path for file in result.files], ["a.py"])
        self.assertEqual(result.report.failed_paths, ["b.py"])
        self.assertEqual(result.report.status, "partial")

    def test_rate_limit_stops_remaining_blob_calls(self):
        session = session_for([file_entry("a.py", "a"), file_entry("b.py", "b")])
        delegate = session.get.side_effect
        session.get.side_effect = lambda url, **kwargs: (
            Response(status=429) if "/git/blobs/" in url else delegate(url, **kwargs)
        )
        result = source.collect_repository_snapshot("owner/repo", session=session)
        self.assertEqual(session.get.call_count, 3)
        self.assertIn("access_or_rate_limit", result.report.warnings)

    def test_deadline_prevents_network_request(self):
        session = Mock()
        with patch.object(source.time, "monotonic", side_effect=[0, 100]):
            result = source.collect_repository_snapshot("owner/repo", session=session)
        session.get.assert_not_called()
        self.assertIn("repository_deadline", result.report.warnings)

    def test_response_size_limit_and_invalid_json(self):
        for response, message in [(Response(raw=b"x" * (1024 * 1024 + 1)), "response_too_large"),
                                  (Response(raw=b"not json"), "invalid_json")]:
            session = Mock()
            session.get.return_value = response
            result = source.collect_repository_snapshot("owner/repo", session=session)
            self.assertIn(message, result.report.warnings)

    def test_invalid_repository_name_never_reaches_network(self):
        session = Mock()
        for name in ("owner/repo?token=value", "../repo", "owner/repo/extra", "https://other"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                source.collect_repository_snapshot(name, session=session)
        session.get.assert_not_called()

    def test_empty_repository_fails_without_fabricated_sha(self):
        session = Mock()
        session.get.return_value = Response([])
        result = source.collect_repository_snapshot("owner/repo", session=session)
        self.assertIsNone(result.commit_sha)
        self.assertEqual(result.report.status, "failed")


class ChunkingTests(unittest.TestCase):
    def assert_grounded(self, original, result, config):
        files = {file.path: file.content.splitlines(keepends=True) for file in original.files}
        for chunk in result.chunks:
            self.assertEqual(chunk.text, "".join(files[chunk.path][chunk.start_line - 1:chunk.end_line]))
            self.assertEqual(chunk.commit_sha, original.commit_sha)
            self.assertLessEqual(len(chunking.format_passage(chunk.path, chunk.symbol, chunk.text)), config.chunk_tokens)

    def test_python_symbols_decorators_and_module_prelude(self):
        text = ('import os\n\n@decorate\nclass Store:\n    version = 1\n'
                '    @staticmethod\n    def put(key):\n        return key\n\n'
                'async def fetch():\n    return 1\n')
        original = snapshot(text)
        config = RagConfig(chunk_tokens=400)
        result = chunking.chunk_snapshot(original, token_count=len, config=config)
        symbols = {chunk.symbol for chunk in result.chunks}
        self.assertTrue({None, "Store", "Store.put", "fetch"}.issubset(symbols))
        store = next(chunk for chunk in result.chunks if chunk.symbol == "Store")
        self.assertEqual(store.start_line, 3)
        method = next(chunk for chunk in result.chunks if chunk.symbol == "Store.put")
        self.assertTrue(method.text.startswith("    @staticmethod"))
        self.assert_grounded(original, result, config)

    def test_invalid_python_falls_back_without_executing_it(self):
        original = snapshot("def broken(:\n    pass\n")
        result = chunking.chunk_snapshot(original, token_count=len)
        self.assertEqual(result.fallback_files, ["src/main.py"])
        self.assertTrue(result.chunks)

    def test_markdown_heading_inside_fence_is_not_a_section(self):
        text = "# Guide\nintro\n```python\n# not heading\nx = 1\n```\n## Memory\ncheckpoint\n"
        original = snapshot(text, "README.md")
        result = chunking.chunk_snapshot(original, token_count=len)
        self.assertEqual([c.symbol for c in result.chunks], ["Guide", "Guide > Memory"])
        self.assertIn("# not heading", result.chunks[0].text)

    def test_long_sections_respect_budget_and_preserve_crlf(self):
        original = snapshot("".join(f"line {n:02d} value\r\n" for n in range(20)), "config.txt")
        config = RagConfig(chunk_tokens=90, overlap_tokens=20)
        result = chunking.chunk_snapshot(original, token_count=len, config=config)
        self.assertGreater(len(result.chunks), 1)
        self.assert_grounded(original, result, config)
        covered = {n for c in result.chunks for n in range(c.start_line, c.end_line + 1)}
        self.assertEqual(covered, set(range(1, 21)))
        self.assertLessEqual(result.chunks[1].start_line, result.chunks[0].end_line)

    def test_oversized_single_line_is_reported_without_fabricating_shorter_source(self):
        original = snapshot("x" * 200 + "\nshort\n", "a.txt")
        result = chunking.chunk_snapshot(original, token_count=len, config=RagConfig(chunk_tokens=80))
        self.assertEqual(result.oversized_lines, {"a.txt": [1]})
        self.assertEqual([(c.start_line, c.text) for c in result.chunks], [(2, "short\n")])

    def test_ids_stable_for_same_version_and_different_across_commits(self):
        first = chunking.chunk_snapshot(snapshot("x = 1\n"), token_count=len)
        again = chunking.chunk_snapshot(snapshot("x = 1\n"), token_count=len)
        changed = chunking.chunk_snapshot(snapshot("x = 1\n", sha="d" * 40), token_count=len)
        self.assertEqual(first.chunks[0].chunk_id, again.chunks[0].chunk_id)
        self.assertNotEqual(first.chunks[0].chunk_id, changed.chunks[0].chunk_id)

    def test_blank_file_produces_no_useless_chunks(self):
        result = chunking.chunk_snapshot(snapshot("\n \n", "a.txt"), token_count=len)
        self.assertEqual(result.chunks, [])


if __name__ == "__main__":
    unittest.main()
