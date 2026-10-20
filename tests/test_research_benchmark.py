"""Synthetic contract fixtures only; not the real human-reviewed benchmark."""
from copy import deepcopy
from dataclasses import replace
import time
import unittest
from unittest.mock import Mock, patch

from reposcouts.rag.benchmark import validate_manifest, score_benchmark, invoke_frozen_judgment
from reposcouts.rag.cache import digest, sanitizer_fingerprint
from reposcouts.rag.chunking import chunk_snapshot
from reposcouts.rag.config import RagConfig
from tests.test_rag_v2 import snapshot, Embedder, plan_payload


def fixture():
    manifest = {"repositories": [], "tasks": [], "label_status": "human_reviewed", "frozen_at": "unit-fixture",
                "annotation_pool_modes": ["bm25", "hybrid", "hybrid_rerank"], "annotation_pool_depth": 20}
    corpora = {}
    for i in range(5):
        name = f"fixture/repo{i}"
        snap = snapshot()
        snap.full_name = name
        chunks = [c.model_dump() for c in chunk_snapshot(snap, token_count=len).chunks]
        manifest["repositories"].append({"full_name": name, "commit_sha": snap.commit_sha,
            "split": "development" if i < 3 else "test", "corpus_digest": digest(chunks)})
        corpora[name] = {"full_name": name, "commit_sha": snap.commit_sha, "chunks": chunks,
                         "expires_at": time.time() + 3600, "sanitizer_policy": sanitizer_fingerprint()}
        for j in range(4):
            label = ("satisfied", "not_satisfied", "uncertain")[j % 3]
            manifest["tasks"].append({"id": f"{i}-{j}", "repository": name, "query": "PersistentMemory",
                "reviewed_by": "synthetic-contract-fixture", "reviewed_at": "unit-fixture",
                "gold": {"status": label, "relevance": {chunks[0]["chunk_id"]: 2},
                         "contradiction_ids": [chunks[0]["chunk_id"]] if label == "not_satisfied" else [],
                         "rationale": "Synthetic test; no real repository quality claim"}})
    return manifest, corpora


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.corpora = fixture()
        self.config = replace(RagConfig(), cache_enabled=False, reranker_enabled=False)

    def test_unreviewed_manifest_cannot_produce_quality_scores(self):
        self.manifest["label_status"] = "human_review_pending"
        with self.assertRaisesRegex(ValueError, "gold_not_frozen"):
            score_benchmark(self.manifest, self.corpora, config=self.config)

    def test_annotation_pool_cannot_favor_only_one_retrieval_arm(self):
        self.manifest["annotation_pool_modes"] = ["bm25"]
        with self.assertRaisesRegex(ValueError, "all_ablations"):
            validate_manifest(self.manifest, reviewed=True)

    def test_frozen_corpus_digest_and_ttl_are_checked(self):
        for changed in ("digest", "ttl"):
            corpora = deepcopy(self.corpora)
            if changed == "digest":
                corpora["fixture/repo3"]["chunks"][0]["text"] = "changed\n"
            else:
                corpora["fixture/repo3"]["expires_at"] = 0
            with self.assertRaises(ValueError):
                score_benchmark(self.manifest, corpora, config=self.config)

    def test_scoring_uses_only_requested_repository_split(self):
        report = score_benchmark(self.manifest, self.corpora, config=self.config)
        self.assertEqual(len(report["rows"]), 8)
        self.assertEqual(report["judgment"], None)
        self.assertTrue(all(row["id"].startswith(("3-", "4-")) for row in report["rows"]))
        self.assertIn("source_digest", report["reproducibility"])

    def test_hybrid_quality_mode_must_not_silently_degrade(self):
        embedder = Embedder()
        embedder.embed_query = Mock(side_effect=ValueError("unavailable"))
        with self.assertRaisesRegex(ValueError, "hybrid_ablation_degraded"):
            score_benchmark(self.manifest, self.corpora, config=self.config, mode="hybrid", embedder=embedder)

    def test_predicted_judgments_and_independent_citation_review(self):
        predictions = [{"id": t["id"], "status": t["gold"]["status"], "evidence_ids": ["E" + next(iter(t["gold"]["relevance"]))]}
                       for t in self.manifest["tasks"] if t["repository"].endswith(("3", "4"))]
        review = {"predictions_digest": digest(predictions), "reviewed_by": "synthetic-fixture", "reviewed_at": "unit-fixture",
                  "labels": [{"task_id": p["id"], "evidence_id": p["evidence_ids"][0], "supports": False} for p in predictions]}
        report = score_benchmark(self.manifest, self.corpora, config=self.config, predictions=predictions, citation_review=review)
        self.assertEqual(report["judgment"]["macro_f1"], 1)
        self.assertEqual(report["citation_precision"], 0)  # Valid IDs do not imply semantic support.
        review["predictions_digest"] = "wrong"
        with self.assertRaisesRegex(ValueError, "not_bound"):
            score_benchmark(self.manifest, self.corpora, config=self.config, predictions=predictions, citation_review=review)

    def test_no_trace_client_is_created_without_explicit_flag(self):
        with patch("reposcouts.rag.benchmark.evaluate_frozen_task", return_value={"id": "test"}), \
                patch("reposcouts.observability.tracing.build_trace_client") as client:
            result = invoke_frozen_judgment({}, {}, None, None, self.config)
        client.assert_not_called()
        self.assertEqual(result, {"id": "test"})

    def test_benchmark_native_trace_uses_sanitized_client_and_separate_project(self):
        from langsmith import Client
        client = Client(api_key="offline", api_url="http://127.0.0.1:1", auto_batch_tracing=False)
        with patch("reposcouts.observability.tracing.build_trace_client", return_value=client), \
                patch("reposcouts.rag.benchmark.evaluate_frozen_task", return_value={"id": "test", "status": "uncertain", "evidence_ids": []}), \
                patch.object(Client, "create_run") as create, patch.object(Client, "update_run"), \
                patch("requests.Session.request", side_effect=AssertionError("No HTTP")):
            result = invoke_frozen_judgment({"id": "test"}, {"commit_sha": "a" * 40}, None, None, self.config, tracing=True)
        self.assertIn("trace_id", result)
        roots = [call.kwargs for call in create.call_args_list if call.kwargs["name"] == "RepositoryResearchBenchmark"]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["session_name"], "reposcouts-rag-evaluation-v2")


class ParserCallTests(unittest.TestCase):
    def test_one_existing_llm_response_is_validated_without_second_call(self):
        import json
        from langchain_core.messages import AIMessage
        from reposcouts.stages import requirement_parser
        with patch.object(requirement_parser, "requirement_parser") as chain:
            chain.invoke.return_value = {"raw": AIMessage(content=json.dumps(plan_payload())), "parsed": None}
            parsed = requirement_parser.parse_user_requirements("Find LangGraph or RAG")
        chain.invoke.assert_called_once()
        self.assertEqual(parsed.must_have, ["LangGraph or RAG"])
        self.assertTrue(parsed.retrieval_plan)


if __name__ == "__main__":
    unittest.main()
