"""Offline contracts for research plans, real parsers, incremental state and grounding."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from reposcouts.core.schemas import UserRequirements, RepositoryEvidence
from reposcouts.rag.config import RagConfig
from reposcouts.rag.cache import SQLiteArtifactStore
from reposcouts.rag.chunking import chunk_snapshot
from reposcouts.rag.incremental import IncrementalArtifacts, CachedEmbedder
from reposcouts.rag.planning import validate_parsed_payload, RequirementValidationError
from reposcouts.rag.requirements import decompose_requirements
from reposcouts.rag.reranking import rerank
from reposcouts.rag.retrieval import HybridIndex
from reposcouts.rag.schemas import RepositoryFile, RepositorySnapshot, RetrievalHit, RetrievalPlanItem
from reposcouts.rag.grounding import ground_evaluation
from reposcouts.rag.service import ResearchSession, research_scope
from reposcouts.rag.security import sanitize_content
from reposcouts.rag.context import source_kind
from reposcouts.rag.github_source import collect_repository_snapshot, expand_repository_snapshot
from tests.test_rag_source_chunking import file_entry, session_for, Response, COMMIT, TREE
from tests.test_run_manager import sample_evaluation
from concurrent.futures import ThreadPoolExecutor


def requirements(**updates):
    values = dict(topic="agent", language=[], technology_stack=[], difficulty="any",
                  must_have=["persistent memory"], nice_to_have=[], exclusions=[],
                  learning_goal="integration", time_budget_in_hours=None)
    return UserRequirements(**(values | updates))


def plan_payload():
    value = requirements(must_have=["LangGraph or RAG"]).model_dump()
    value["retrieval_plan"] = [dict(kind="must_have", parent_requirement="LangGraph or RAG",
                                   source_quote="LangGraph or RAG", atomic_requirement="LangGraph or RAG",
                                   queries=["LangGraph persistent graph", "RAG"])]
    return value


def snapshot(sha="a" * 40, changed=False):
    return RepositorySnapshot(full_name="owner/repo", commit_sha=sha, files=[
        RepositoryFile(path="memory.py", blob_sha="b" * 40,
                       content="class PersistentMemory:\n    def save(self, value):\n        return self.storage.write(value)\n"),
        RepositoryFile(path="app.py", blob_sha=("d" if changed else "c") * 40,
                       content="from memory import PersistentMemory\nmemory = PersistentMemory()\n" + ("memory.save(42)\n" if changed else "")),
    ])


class Embedder:
    fingerprint = "controlled@v2"
    def __init__(self):
        self.documents = []
    def token_count(self, text):
        return len(text)
    def embed_documents(self, chunks):
        self.documents.extend(c.path for c in chunks)
        return np.ones((len(chunks), 3), dtype=np.float32)
    def embed_query(self, query):
        return np.ones((1, 3), dtype=np.float32)


class PlanTests(unittest.TestCase):
    def test_valid_plan_preserves_parent_and_original_query(self):
        value = validate_parsed_payload(plan_payload(), "Find LangGraph or RAG")
        task = decompose_requirements(value)[0]
        self.assertEqual(task.parent_requirement, "LangGraph or RAG")
        self.assertEqual(task.queries[0], "LangGraph or RAG")

    def test_protected_or_or_version_cannot_be_split(self):
        value = plan_payload()
        value["retrieval_plan"][0]["atomic_requirement"] = "LangGraph"
        parsed = validate_parsed_payload(value, "Find LangGraph or RAG")
        self.assertFalse(parsed.retrieval_plan)
        self.assertIn("retrieval_plan_invalid_rule_fallback", parsed.plan_warnings)
        self.assertEqual(decompose_requirements(parsed)[0].requirement, "LangGraph or RAG")

    def test_missing_or_fake_source_anchor_rejects_entire_parse(self):
        for quote in ("invented", ""):
            payload = plan_payload()
            payload["retrieval_plan"][0]["source_quote"] = quote
            with self.assertRaises(RequirementValidationError):
                validate_parsed_payload(payload, "Find LangGraph or RAG")

    def test_invalid_query_shape_falls_back_after_anchor_validation(self):
        payload = plan_payload()
        payload["retrieval_plan"][0]["queries"] = "not a list"
        parsed = validate_parsed_payload(payload, "Find LangGraph or RAG")
        self.assertFalse(parsed.retrieval_plan)
        self.assertEqual(parsed.must_have, ["LangGraph or RAG"])

    def test_unknown_base_fields_and_coerced_values_are_rejected(self):
        for update in ({"surprise": "x"}, {"time_budget_in_hours": "8"}):
            with self.assertRaises(RequirementValidationError):
                validate_parsed_payload(plan_payload() | update, "Find LangGraph or RAG")

    def test_new_rewrite_term_never_becomes_hard_requirement(self):
        payload = plan_payload()
        payload["retrieval_plan"][0]["queries"] = ["Redis persistence"]
        value = validate_parsed_payload(payload, "Find LangGraph or RAG")
        self.assertEqual(value.must_have, ["LangGraph or RAG"])

    def test_foreign_task_and_unhashable_parent_do_not_escape_validation(self):
        payload = plan_payload()
        payload["retrieval_plan"].append({"kind": [], "parent_requirement": []})
        value = validate_parsed_payload(payload, "Find LangGraph or RAG")
        self.assertFalse(value.retrieval_plan)


class SemanticTests(unittest.TestCase):
    def test_java_packages_are_not_mistaken_for_examples_and_test_files_are_classified(self):
        self.assertEqual(source_kind("src/main/java/com/example/Memory.java"), "source")
        self.assertEqual(source_kind("agent-examples/src/main/java/com/example/Memory.java"), "example")
        self.assertEqual(source_kind("src/main/java/com/example/MemoryTest.java"), "test")
        self.assertEqual(source_kind("src/memory.spec.ts"), "test")

    def test_setext_markdown_headings_keep_original_lines(self):
        snap = snapshot()
        snap.files = [RepositoryFile(path="README.md", blob_sha="b" * 40, content="Intro\n=====\nsetup\nMemory\n------\npersistence\n")]
        result = chunk_snapshot(snap, token_count=len)
        self.assertEqual([(c.start_line, c.end_line, c.symbol) for c in result.chunks], [(1, 3, "Intro"), (4, 6, "Intro > Memory")])

    def test_java_xml_credentials_and_basic_auth_are_redacted_with_lines_preserved(self):
        raw = '<server><password>private-value\nsecond-line</password></server>\nAuthorization: Basic dXNlcjpwYXNz\n'
        safe = sanitize_content(raw)
        for secret in ("private-value", "second-line", "dXNlcjpwYXNz"):
            self.assertNotIn(secret, safe)
        self.assertEqual(len(raw.splitlines()), len(safe.splitlines()))

    def test_real_grammars_emit_symbols_and_exact_lines(self):
        samples = {
            "Memory.java": ("class Memory {\n public void save() { store(); }\n}\n", "Memory.save"),
            "memory.ts": ("export class Memory {\n save(): void { store(); }\n}\n", "Memory.save"),
            "memory.js": ("export const save = () => {\n return store();\n};\n", "save"),
            "view.tsx": ("export function View() {\n return <div>hello</div>;\n}\n", "View"),
            "view.jsx": ("export function View() {\n return <div>hello</div>;\n}\n", "View"),
        }
        for path, (text, symbol) in samples.items():
            with self.subTest(path=path):
                snap = RepositorySnapshot(full_name="owner/repo", commit_sha="a" * 40,
                    files=[RepositoryFile(path=path, blob_sha="b" * 40, content=text)])
                result = chunk_snapshot(snap, token_count=len)
                self.assertEqual(result.fallback_files, [])
                self.assertIn(symbol, {c.symbol for c in result.chunks})
                for chunk in result.chunks:
                    self.assertEqual(chunk.text, "".join(text.splitlines(keepends=True)[chunk.start_line - 1:chunk.end_line]))

    def test_malformed_java_has_visible_fallback(self):
        snap = snapshot()
        snap.files = [RepositoryFile(path="bad.java", blob_sha="b" * 40, content="class { ???\n")]
        self.assertEqual(chunk_snapshot(snap, token_count=len).fallback_files, ["bad.java"])


class PersistentTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / "artifacts" / "rag-v2-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        assert Path(self.temp.name).resolve().is_relative_to(root.resolve())
        self.addCleanup(self.temp.cleanup)
        self.config = replace(RagConfig(), cache_dir=self.temp.name, reranker_enabled=False)

    def test_absolute_ttl_is_not_extended_by_reads_or_identical_writes(self):
        now = [1000.0]
        store = SQLiteArtifactStore(replace(self.config, cache_ttl_s=10), clock=lambda: now[0])
        self.addCleanup(store.close)
        store.put("key", {"value": "safe"}, kind="chunks")
        now[0] = 1008
        self.assertEqual(store.get("key"), {"value": "safe"})
        store.put("key", {"value": "safe"}, kind="chunks")
        now[0] = 1011
        self.assertIsNone(store.get("key"))

    def test_corrupted_payload_is_a_miss_and_cache_can_be_cleared(self):
        store = SQLiteArtifactStore(self.config)
        self.addCleanup(store.close)
        store.put("x", [1, 2], kind="vector")
        with store.db:
            store.db.execute("UPDATE artifacts SET payload='invalid'")
        self.assertIsNone(store.get("x"))
        store.put("y", [3], kind="vector")
        store.clear()
        self.assertIsNone(store.get("y"))

    def test_capacity_evicts_old_artifacts(self):
        store = SQLiteArtifactStore(replace(self.config, cache_max_bytes=65536))
        self.addCleanup(store.close)
        for i in range(8):
            store.put(str(i), "x" * 12000, kind="chunks")
        self.assertLessEqual(store.path.stat().st_size, 65536)
        self.assertIsNone(store.get("0"))

    def test_new_commit_reuses_unchanged_chunks_and_vectors_only(self):
        one = Embedder()
        with research_scope(ResearchSession(self.config, snapshot_loader=lambda *a: snapshot(), embedder_factory=lambda _: one)) as session:
            first = session.research("owner/repo", requirements())
        two = Embedder()
        with research_scope(ResearchSession(self.config, snapshot_loader=lambda *a: snapshot("f" * 40, True), embedder_factory=lambda _: two)) as session:
            second = session.research("owner/repo", requirements())
        self.assertIn("memory.py", one.documents)
        self.assertNotIn("memory.py", two.documents)
        self.assertEqual(set(two.documents), {"app.py"})
        self.assertGreater(second.cache_stats["chunk_file_hits"], 0)
        self.assertGreater(second.cache_stats["vector_hits"], 0)
        self.assertTrue(all(s.citation.commit_sha == "f" * 40 for s in second.snippets))
        self.assertFalse({s.citation.evidence_id for s in first.snippets} & {s.citation.evidence_id for s in second.snippets})

    def test_deleted_files_are_removed_from_active_index(self):
        with research_scope(ResearchSession(self.config, snapshot_loader=lambda *a: snapshot(), embedder_factory=lambda _: Embedder())) as session:
            session.research("owner/repo", requirements())
        newer = snapshot("f" * 40)
        newer.files = newer.files[:1]
        with research_scope(ResearchSession(self.config, snapshot_loader=lambda *a: newer, embedder_factory=lambda _: Embedder())) as session:
            result = session.research("owner/repo", requirements())
        self.assertEqual(result.cache_stats["deleted_files"], 1)
        self.assertNotIn("app.py", {s.citation.path for s in result.snippets})

    def test_raw_secrets_never_reach_embeddings_or_disk(self):
        snap = snapshot()
        snap.files[0].content += 'api_key = "unlisted-private-test-value"\n'
        embedder = Embedder()
        original = embedder.embed_documents
        def embed(chunks):
            self.assertNotIn("unlisted-private-test-value", "".join(c.text for c in chunks))
            return original(chunks)
        embedder.embed_documents = embed
        with research_scope(ResearchSession(self.config, snapshot_loader=lambda *a: snap, embedder_factory=lambda _: embedder)) as session:
            session.research("owner/repo", requirements())
        self.assertNotIn(b"unlisted-private-test-value", Path(self.temp.name, "artifacts.sqlite3").read_bytes())

    def test_changed_policy_does_not_reuse_old_chunks(self):
        store = SQLiteArtifactStore(self.config)
        self.addCleanup(store.close)
        a = IncrementalArtifacts(store, self.config)
        a.chunks(snapshot(), len)
        b = IncrementalArtifacts(store, replace(self.config, chunk_tokens=300))
        b.chunks(snapshot(), len)
        self.assertEqual(b.stats["chunk_file_hits"], 0)

    def test_failed_or_truncated_snapshot_does_not_replace_last_manifest(self):
        store = SQLiteArtifactStore(self.config)
        self.addCleanup(store.close)
        IncrementalArtifacts(store, self.config).publish(snapshot())
        for reason in ("empty", "truncated"):
            partial = snapshot("f" * 40)
            if reason == "empty":
                partial.files = []
            else:
                partial.report.tree_truncated = True
                partial.files = partial.files[:1]
            artifacts = IncrementalArtifacts(store, self.config)
            self.assertEqual(artifacts.publish(partial)["deleted_files"], 0)
            self.assertIn("incomplete_snapshot_manifest_not_published", artifacts.warnings)
        stats = IncrementalArtifacts(store, self.config).publish(snapshot("e" * 40))
        self.assertEqual(stats["added_files"], 0)
        self.assertEqual(stats["deleted_files"], 0)

    def test_store_created_on_request_thread_works_on_graph_worker(self):
        session = ResearchSession(self.config, snapshot_loader=lambda *a: snapshot(), embedder_factory=lambda _: Embedder())
        with research_scope(session), ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(session.research, "owner/repo", requirements()).result(timeout=5)
            self.assertNotIn("persistent_cache_write_failed", result.warnings)
        store = SQLiteArtifactStore(self.config)
        self.addCleanup(store.close)
        self.assertGreater(store.db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 0)


class ExpansionTests(unittest.TestCase):
    def test_benchmark_collects_explicit_commit_without_resolving_head(self):
        transport = session_for([file_entry("README.md", "# fixed commit\n")])
        original = transport.get.side_effect
        def get(url, **kwargs):
            if url.endswith("/git/commits/" + COMMIT):
                return Response({"sha": COMMIT, "tree": {"sha": TREE}})
            return original(url, **kwargs)
        transport.get.side_effect = get
        snap = collect_repository_snapshot("owner/repo", session=transport, commit_sha=COMMIT)
        self.assertEqual(snap.commit_sha, COMMIT)
        self.assertTrue(snap.files)
        self.assertFalse(any(c.args[0].endswith("/commits") for c in transport.get.call_args_list))

    def test_expansion_keeps_sha_and_shares_total_budget(self):
        files = [file_entry("README.md", "# persistent memory\n"),
                 file_entry("src/memory.py", "def persist():\n return store()\n"),
                 file_entry("src/main.py", "persist()\n")]
        transport = session_for(files)
        config = replace(RagConfig(), initial_files=1, max_files=2)
        snap = collect_repository_snapshot("owner/repo", config, session=transport)
        self.assertEqual(len(snap.files), 1)
        original_sha = snap.commit_sha
        loaded = expand_repository_snapshot(snap, ["src/memory.py", "src/main.py"], config, session=transport)
        self.assertEqual(loaded, 1)
        self.assertEqual(snap.commit_sha, original_sha)
        self.assertEqual(snap.report.selected_files, 2)
        self.assertEqual(snap.report.expansion_rounds, 1)
        self.assertEqual(sum(call.args[0].endswith("/commits") for call in transport.get.call_args_list), 1)

    def test_real_service_expansion_retrieves_new_source(self):
        files = [file_entry("README.md", "# persistent memory\n"),
                 file_entry("src/memory.py", "def persistent_memory():\n return store()\n")]
        transport = session_for(files)
        config = replace(RagConfig(), initial_files=1, cache_enabled=False, reranker_enabled=False)
        session = ResearchSession(config,
            snapshot_loader=lambda name, cfg: collect_repository_snapshot(name, cfg, session=transport),
            snapshot_expander=lambda snap, paths, cfg: expand_repository_snapshot(snap, paths, cfg, session=transport),
            embedder_factory=lambda _: Embedder())
        with research_scope(session):
            pack = session.research("owner/repo", requirements())
            self.assertIn("src/memory.py", {s.citation.path for s in pack.snippets})
            self.assertEqual(pack.ingestion.expansion_rounds, 1)
            session.research("owner/repo", requirements())
        self.assertEqual(sum(c.args[0].endswith("/commits") for c in transport.get.call_args_list), 1)


class RerankerTests(unittest.TestCase):
    def test_model_loader_is_pinned_and_local_only_by_default(self):
        from reposcouts.rag.reranking import LocalReranker, _load_reranker
        module = types.ModuleType("sentence_transformers")
        module.CrossEncoder = Mock()
        config = RagConfig()
        _load_reranker.cache_clear()
        self.addCleanup(_load_reranker.cache_clear)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            LocalReranker(config)
        module.CrossEncoder.assert_called_once_with(config.reranker_model, revision=config.reranker_revision,
            max_length=config.reranker_max_tokens, local_files_only=True, trust_remote_code=False)

    def test_reranker_scores_override_rrf_order(self):
        chunks = chunk_snapshot(snapshot(), token_count=len).chunks
        hits = [RetrievalHit(chunk=c, rrf_score=1 / (i + 1)) for i, c in enumerate(chunks)]
        model = Mock()
        model.truncated_pairs = 0
        model.score.return_value = list(range(len(hits)))
        values, warnings, used = rerank("memory", hits, model, RagConfig())
        self.assertTrue(used)
        self.assertEqual(values[0].chunk.chunk_id, hits[-1].chunk.chunk_id)
        self.assertEqual(warnings, [])

    def test_nonfinite_scores_fall_back_and_do_not_erase_hits(self):
        hit = RetrievalHit(chunk=chunk_snapshot(snapshot(), token_count=len).chunks[0], rrf_score=1)
        model = Mock()
        model.score.return_value = [float("nan")]
        values, warnings, used = rerank("memory", [hit], model, RagConfig())
        self.assertFalse(used)
        self.assertEqual(values, [hit])
        self.assertIn("reranker_failed_rrf_fallback", warnings)


class ImplementationGateTests(unittest.TestCase):
    def evidence(self):
        session = ResearchSession(replace(RagConfig(), cache_enabled=False, reranker_enabled=False),
            snapshot_loader=lambda *a: snapshot(), embedder_factory=lambda _: Embedder())
        with research_scope(session):
            pack = session.research("owner/repo", requirements())
        return RepositoryEvidence(full_name="owner/repo", metadata={}, readme="", structure={}, research=pack)

    def value(self, evidence):
        value = sample_evaluation()
        check = value.must_have_checks[0]
        check.requirement = "persistent memory"
        check.evidence_ids = next(item.evidence_ids for item in evidence.research.requirements if item.task.kind == "must_have")
        return value

    def test_implementation_without_wiring_is_uncertain(self):
        evidence = self.evidence()
        value = self.value(evidence)
        value.must_have_checks[0].implementation_evidence_ids = [s.citation.evidence_id for s in evidence.research.snippets if s.citation.symbol]
        result = ground_evaluation(requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].status, "uncertain")
        self.assertEqual(result.must_have_checks[0].implementation_level, "implemented")

    def test_implementation_and_callsite_can_satisfy(self):
        evidence = self.evidence()
        value = self.value(evidence)
        check = value.must_have_checks[0]
        check.implementation_evidence_ids = [s.citation.evidence_id for s in evidence.research.snippets if s.citation.symbol]
        check.integration_evidence_ids = [s.citation.evidence_id for s in evidence.research.snippets if s.citation.path == "app.py"]
        result = ground_evaluation(requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].status, "satisfied")
        self.assertEqual(result.must_have_checks[0].implementation_level, "integrated")

    def test_not_satisfied_requires_explicit_contradiction_ids(self):
        evidence = self.evidence()
        value = self.value(evidence)
        value.must_have_checks[0].status = "not_satisfied"
        result = ground_evaluation(requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].status, "uncertain")

    def test_unrelated_function_call_does_not_prove_wiring(self):
        evidence = self.evidence()
        value = self.value(evidence)
        check = value.must_have_checks[0]
        check.implementation_evidence_ids = [s.citation.evidence_id for s in evidence.research.snippets if s.citation.symbol]
        check.integration_evidence_ids = [s.citation.evidence_id for s in evidence.research.snippets if s.citation.path == "app.py"]
        for snippet in evidence.research.snippets:
            if snippet.citation.path == "app.py":
                snippet.text = "print('unrelated')\n"
        result = ground_evaluation(requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].status, "uncertain")


if __name__ == "__main__":
    unittest.main()
