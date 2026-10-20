import json
import os
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, tracing_context

with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-model-key"}):
    from reposcouts.stages import repository_evaluator, ranker
    from reposcouts.api.run_manager import RunManager, RunRecord
    from reposcouts.workflow import nodes

from reposcouts.core.schemas import (
    CandidateRepository, CandidateSearchResult, CriticResult, RepositoryEvidence,
    RankedRepository, RankingResult,
)
from reposcouts.rag.config import RagConfig
from reposcouts.rag.grounding import ground_evaluation
from reposcouts.rag.requirements import decompose_requirements
from reposcouts.rag.schemas import RepositoryFile, RepositorySnapshot, RetrievalResult
from reposcouts.rag.security import sanitize_content
from reposcouts.rag.service import ResearchSession, current_research, research_scope
from reposcouts.rag.evaluation import retrieval_metrics, judgment_metrics, citation_precision
from tests.test_run_manager import sample_requirements, sample_evaluation
from tests.test_rag_source_chunking import file_entry, session_for


def fixture():
    return RepositorySnapshot(
        full_name="example/agent", commit_sha="a" * 40,
        files=[RepositoryFile(path="README.md", blob_sha="b" * 40,
                              content="# clear README\nInstall and run the agent with Python.\n"),
               RepositoryFile(path="src/graph.py", blob_sha="c" * 40,
                              content="from langgraph.graph import StateGraph\ndef build():\n    return StateGraph(dict)\n")],
    )


class Embedder:
    fingerprint = "fake@v1"
    def __init__(self):
        self.document_calls = 0
    def token_count(self, text):
        return len(text)
    def embed_documents(self, chunks):
        self.document_calls += 1
        return np.ones((len(chunks), 2))
    def embed_query(self, query):
        return np.ones((1, 2))


def make_session(config=None):
    config = replace(config or RagConfig(), cache_enabled=False, reranker_enabled=False)
    embedder = Embedder()
    loader = Mock(return_value=fixture())
    factory = Mock(return_value=embedder)
    return ResearchSession(config, snapshot_loader=loader, embedder_factory=factory), loader, factory, embedder


def make_evidence(session=None):
    session = session or make_session()[0]
    pack = session.research("example/agent", sample_requirements())
    return RepositoryEvidence(full_name="example/agent", metadata={}, readme="", structure={}, research=pack)


def cited_evaluation(evidence):
    value = sample_evaluation()
    task = next(task for task in evidence.research.requirements if task.task.kind == "must_have")
    value.must_have_checks[0].evidence_ids = task.evidence_ids[:1]
    return value


class ResearchTests(unittest.TestCase):
    def test_snapshot_chunking_and_document_vectors_reused_across_retries(self):
        session, loader, factory, embedder = make_session()
        first = session.research("example/agent", sample_requirements())
        second = session.research("example/agent", sample_requirements())
        self.assertFalse(first.index_reused)
        self.assertTrue(second.index_reused)
        self.assertEqual(first.commit_sha, second.commit_sha)
        loader.assert_called_once()
        factory.assert_called_once()
        self.assertEqual(embedder.document_calls, 1)

    def test_missing_model_degrades_without_downloading(self):
        factory = Mock(side_effect=RuntimeError("weights absent"))
        session = ResearchSession(replace(RagConfig(), cache_enabled=False, reranker_enabled=False),
                                  snapshot_loader=Mock(return_value=fixture()), embedder_factory=factory)
        pack = session.research("example/agent", sample_requirements())
        self.assertTrue(pack.snippets)
        self.assertEqual({task.mode for task in pack.requirements}, {"bm25"})
        self.assertIn("local_embedding_unavailable", pack.warnings)
        self.assertIn("byte_budget_chunking", pack.warnings)
        self.assertTrue(factory.call_args.args[0].local_files_only)

    def test_failed_repository_does_not_attempt_model_loading(self):
        factory = Mock()
        session = ResearchSession(replace(RagConfig(), cache_enabled=False, reranker_enabled=False),
                                  snapshot_loader=Mock(side_effect=RuntimeError("network failure")), embedder_factory=factory)
        pack = session.research("example/agent", sample_requirements())
        factory.assert_not_called()
        self.assertEqual(pack.snippets, [])
        self.assertTrue(all(task.mode == "unavailable" for task in pack.requirements))

    def test_scope_is_nested_and_cleared_on_success_or_failure(self):
        session, _, _, _ = make_session()
        with self.assertRaises(RuntimeError):
            with research_scope(session) as active:
                self.assertIs(active, current_research())
                with research_scope() as nested:
                    self.assertIs(nested, active)
                active.research("example/agent", sample_requirements())
                raise RuntimeError("stop")
        self.assertIsNone(current_research())
        self.assertEqual(session._snapshots, {})
        self.assertEqual(len(session.indexes._indexes), 0)

    def test_requirement_conjunctions_preserve_or_negation_and_cpp(self):
        requirements = sample_requirements().model_copy(update={
            "must_have": ["LangGraph+RAG+FastAPI", "FAISS or Chroma", "C++", "no Redis and no cloud"],
        })
        tasks = [task for task in decompose_requirements(requirements) if task.kind == "must_have"]
        values = [task.requirement for task in tasks]
        self.assertEqual(values[:3], ["LangGraph", "RAG", "FastAPI"])
        self.assertIn("FAISS or Chroma", values)
        self.assertIn("C++", values)
        self.assertIn("no Redis and no cloud", values)
        self.assertEqual(tasks[0].parent_requirement, "LangGraph+RAG+FastAPI")

    def test_evidence_budget_omits_whole_chunks_and_records_limit(self):
        session, _, _, _ = make_session(replace(RagConfig(), max_evidence_chars=1))
        evidence = make_evidence(session)
        self.assertEqual(evidence.research.snippets, [])
        self.assertIn("evidence_budget", evidence.research.warnings)
        self.assertTrue(all(not task.evidence_ids for task in evidence.research.requirements))

    def test_content_secrets_and_private_keys_are_masked_preserving_lines(self):
        raw = ('api_key = "unlisted-secret-value"\n'
               'Authorization: Bearer unknown.value\n'
               'token = "ghp_unknownvaluesecret"\n'
               'x = "sk-abcdefghijklmnopqrstuv"\n'
               '-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----\n')
        safe = sanitize_content(raw)
        for secret in ("unlisted-secret-value", "unknown.value", "ghp_unknownvaluesecret", "sk-abcdefghijklmnopqrstuv", "abc123"):
            self.assertNotIn(secret, safe)
        self.assertEqual(len(raw.splitlines()), len(safe.splitlines()))

    def test_multiline_configured_secret_preserves_source_lines(self):
        with patch.dict(os.environ, {"TEST_PRIVATE_SECRET": "first-line\nsecond-line"}):
            safe = sanitize_content("before\nfirst-line\nsecond-line\nafter\n")
        self.assertEqual(safe.splitlines(), ["before", "[REDACTED]", "[REDACTED]", "after"])

    def test_short_configured_secret_is_still_redacted(self):
        with patch.dict(os.environ, {"CUSTOM_API_KEY": "abc123"}):
            self.assertEqual(sanitize_content("value abc123 here"), "value [REDACTED] here")


class GroundingTests(unittest.TestCase):
    def test_valid_ids_get_fixed_commit_citations_from_code(self):
        evidence = make_evidence()
        result = ground_evaluation(sample_requirements(), evidence, cited_evaluation(evidence))
        check = result.must_have_checks[0]
        self.assertEqual(check.status, "satisfied")
        self.assertTrue(check.citations)
        self.assertIn("/blob/" + "a" * 40 + "/", check.citations[0].url)
        self.assertIn("#L", check.citations[0].url)
        self.assertEqual(result.requirement_match, 100)

    def test_invented_ids_downgrade_and_prevent_recommendation(self):
        evidence = make_evidence()
        value = cited_evaluation(evidence)
        value.must_have_checks[0].evidence_ids = ["invented-id"]
        result = ground_evaluation(sample_requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].status, "uncertain")
        self.assertFalse(result.suitable_for_next_stage)
        self.assertEqual(result.must_have_checks[0].citations, [])

    def test_wrong_commit_or_foreign_repository_is_not_a_valid_citation(self):
        for changes in ({"commit_sha": "f" * 40}, {"full_name": "other/repo"}):
            evidence = make_evidence()
            value = cited_evaluation(evidence)
            target = value.must_have_checks[0].evidence_ids[0]
            snippet = next(s for s in evidence.research.snippets if s.citation.evidence_id == target)
            snippet.citation = snippet.citation.model_copy(update=changes)
            result = ground_evaluation(sample_requirements(), evidence, value)
            self.assertEqual(result.must_have_checks[0].status, "uncertain")

    def test_missing_checks_are_restored_as_uncertain(self):
        evidence = make_evidence()
        value = sample_evaluation().model_copy(update={"must_have_checks": []})
        result = ground_evaluation(sample_requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].requirement, "clear README")
        self.assertEqual(result.must_have_checks[0].status, "uncertain")

    def test_empty_research_skips_evaluator(self):
        evidence = make_evidence()
        evidence.research.snippets.clear()
        with patch.object(repository_evaluator, "repository_evaluator") as runnable:
            result = repository_evaluator.evaluate_repository(sample_requirements(), evidence)
        runnable.invoke.assert_not_called()
        self.assertFalse(result.suitable_for_next_stage)

    def test_evaluator_input_budget_is_checked_before_model_call(self):
        evidence = make_evidence()
        session, _, _, _ = make_session(replace(RagConfig(), max_evaluator_input_chars=1))
        with research_scope(session), patch.object(repository_evaluator, "repository_evaluator") as runnable:
            result = repository_evaluator.evaluate_repository(sample_requirements(), evidence)
        runnable.invoke.assert_not_called()
        self.assertIn("evaluator_input_budget", result.uncertainties)

    def test_llm_receives_redacted_bounded_source_evidence(self):
        snapshot = fixture()
        snapshot.files[0].content += 'api_key = "unlisted-sensitive-value"\n'
        session = ResearchSession(replace(RagConfig(), cache_enabled=False, reranker_enabled=False),
                                  snapshot_loader=Mock(return_value=snapshot), embedder_factory=lambda _: Embedder())
        evidence = make_evidence(session)
        with patch.object(repository_evaluator, "repository_evaluator") as runnable:
            runnable.invoke.return_value = cited_evaluation(evidence)
            result = repository_evaluator.evaluate_repository(sample_requirements(), evidence)
        self.assertTrue(result.must_have_checks[0].citations)
        self.assertNotIn("unlisted-sensitive-value", runnable.invoke.call_args.args[0]["evidence"])

    def test_citations_do_not_cover_an_unretrieved_composite_subtask(self):
        evidence = make_evidence()
        value = cited_evaluation(evidence)
        item = next(task for task in evidence.research.requirements if task.task.kind == "must_have")
        missing = item.model_copy(deep=True)
        missing.task.task_id = "missing-subtask"
        missing.evidence_ids = []
        evidence.research.requirements.append(missing)
        result = ground_evaluation(sample_requirements(), evidence, value)
        self.assertEqual(result.must_have_checks[0].status, "uncertain")

    def test_model_evaluation_failure_does_not_discard_other_repositories(self):
        evidence = make_evidence()
        candidates = CandidateSearchResult(search_queries=[], candidates=[
            CandidateRepository(full_name="example/agent", reason="test"),
            CandidateRepository(full_name="example/agent", reason="test"),
        ])
        with patch.object(repository_evaluator, "collect_repository_evidence", return_value=evidence), \
                patch.object(repository_evaluator, "evaluate_repository", side_effect=[RuntimeError("bad JSON"), cited_evaluation(evidence)]):
            results = repository_evaluator.analyze_candidate_repositories(sample_requirements(), candidates)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].suitable_for_next_stage)

    def test_empty_ranking_abstains_without_model_and_critic_does_not_retry(self):
        with patch.object(ranker, "repository_ranker") as runnable:
            result = ranker.rank_repositories(sample_requirements(), [])
        runnable.invoke.assert_not_called()
        self.assertEqual(result.top_repositories, [])
        reviewed = nodes.critic_node({"requirements": sample_requirements(), "evaluations": [], "ranking": result})
        self.assertTrue(reviewed["critic_result"].passed)
        self.assertEqual(reviewed["critic_result"].retry_target, "none")


class ObservabilityTests(unittest.TestCase):
    def test_real_callbacks_capture_nested_stages_tools_and_summaries_only(self):
        content = "# clear README\nNEVER_UPLOAD_RAW_SOURCE\napi_key = 'unlisted-sensitive-value'\n"
        transport = session_for([file_entry("README.md", content)])
        from reposcouts.rag.github_source import collect_repository_snapshot
        session = ResearchSession(
            replace(RagConfig(), cache_enabled=False, reranker_enabled=True),
            snapshot_loader=lambda name, config: collect_repository_snapshot(name, config, session=transport),
            embedder_factory=lambda _: Embedder(),
            reranker_factory=lambda _: type("Reranker", (), {"score": lambda self, query, hits: np.arange(len(hits))})(),
        )
        client = Client(api_key="offline", api_url="http://127.0.0.1:1", auto_batch_tracing=False)
        self.addCleanup(client.close, timeout=0)
        tracer = LangChainTracer(client=client, project_name="offline-rag")

        def root(_):
            result = session.research("example/agent", sample_requirements())
            return {"snippet_count": len(result.snippets)}

        with tracing_context(enabled=True, client=client), \
                patch.object(Client, "create_run") as create, patch.object(Client, "update_run") as update, \
                patch("requests.Session.request", side_effect=AssertionError("Network forbidden")):
            RunnableLambda(root, name="OfflineRagRoot").invoke({}, config={"callbacks": [tracer]})
        starts = [call.kwargs for call in create.call_args_list]
        names = {item["name"] for item in starts}
        self.assertTrue({"rag.ingestion", "github.repository_read", "rag.chunking", "rag.index",
                         "rag.retrieve", "rag.dense", "rag.bm25", "rag.rrf", "rag.evidence",
                         "rag.rerank", "rag.query_rewrite", "rag.context", "rag.index.publish"} <= names)
        github = next(item for item in starts if item["name"] == "github.repository_read")
        self.assertEqual(github["run_type"], "tool")
        ingestion = next(item for item in starts if item["name"] == "rag.ingestion")
        self.assertEqual(github["parent_run_id"], ingestion["id"])
        payload = json.dumps(starts + [call.kwargs for call in update.call_args_list], default=str)
        self.assertNotIn("NEVER_UPLOAD_RAW_SOURCE", payload)
        self.assertNotIn("unlisted-sensitive-value", payload)


class WorkflowTests(unittest.TestCase):
    def test_actual_graph_retry_uses_same_request_snapshot_and_index(self):
        session, loader, _, embedder = make_session()
        candidates = CandidateSearchResult(search_queries=[], candidates=[CandidateRepository(full_name="example/agent", reason="fixture")])
        ranking = RankingResult(
            best_repository="example/agent", final_reason="fixture",
            top_repositories=[RankedRepository(
                full_name="example/agent", base_score=90, score_rank=1, final_rank=1,
                recommendation_reason="fixture", tradeoffs=[], learning_path=[],
                ranking_override=False, override_reason=None,
            )],
        )

        def evaluate(payload):
            return cited_evaluation(RepositoryEvidence.model_validate(json.loads(payload["evidence"])))

        record = RunRecord(run_id="offline-rag-retry", user_request="Find a LangGraph project")
        manager = RunManager()
        manager._current = record
        with research_scope(session), patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "false"}), \
                patch.object(nodes, "parse_user_requirements", return_value=sample_requirements()), \
                patch.object(nodes, "search_candidate_repositories", return_value=candidates), \
                patch.object(session, "metadata", return_value={"full_name": "example/agent"}), \
                patch.object(repository_evaluator, "repository_evaluator") as evaluator, \
                patch.object(ranker, "repository_ranker") as rank_model, \
                patch.object(nodes, "critique_recommendation", side_effect=[
                    CriticResult(passed=False, confidence=80, issues=[], retry_target="evaluator", feedback="Recheck citations"),
                    CriticResult(passed=True, confidence=95, issues=[], retry_target="none", feedback="Supported"),
                ]):
            evaluator.invoke.side_effect = evaluate
            rank_model.invoke.return_value = ranking
            manager._execute(record)
            self.assertEqual(evaluator.invoke.call_count, 2)
            self.assertEqual(evaluator.invoke.call_args.kwargs, {})
            self.assertEqual(evaluator.invoke.call_args.args[0]["critic_feedback"], "Recheck citations")
        self.assertEqual(record.status, "completed")
        events = list(record.events.queue)
        completed = next(event for event in events if event.type == "run.completed")
        self.assertEqual(completed.data["retry_count"], 1)
        self.assertTrue(completed.data["evaluations"][0]["research_summary"]["index_reused"])
        loader.assert_called_once()
        self.assertEqual(embedder.document_calls, 1)
        self.assertIsNone(current_research())

    def test_actual_graph_with_zero_candidates_returns_empty_recommendation(self):
        record = RunRecord(run_id="offline-empty", user_request="Find a project")
        manager = RunManager()
        manager._current = record
        with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "false"}), \
                patch.object(nodes, "parse_user_requirements", return_value=sample_requirements()), \
                patch.object(nodes, "search_candidate_repositories", return_value=CandidateSearchResult(search_queries=[], candidates=[])), \
                patch.object(repository_evaluator, "repository_evaluator") as evaluator, \
                patch.object(ranker, "repository_ranker") as rank_model:
            manager._execute(record)
        evaluator.invoke.assert_not_called()
        rank_model.invoke.assert_not_called()
        self.assertEqual(record.status, "completed")
        completed = next(event for event in list(record.events.queue) if event.type == "run.completed")
        self.assertEqual(completed.data["ranking"]["top_repositories"], [])
        self.assertEqual(completed.data["recommendation_status"], "insufficient_evidence")
        self.assertEqual(record.summary["outcome"], "insufficient_evidence")


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics_and_missing_relevance(self):
        result = retrieval_metrics({"a": 2, "b": 1}, ["x", "a", "a"], 3)
        self.assertEqual(result["recall_at_k"], 0.5)
        self.assertEqual(result["mrr_at_k"], 0.5)
        self.assertEqual(result["hit_rate_at_k"], 1)
        self.assertLess(result["ndcg_at_k"], 1)
        self.assertIsNone(retrieval_metrics({}, ["a"])["recall_at_k"])

    def test_judgment_and_citation_metrics_require_independent_labels(self):
        result = judgment_metrics(["satisfied", "uncertain"], ["satisfied", "uncertain"])
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(citation_precision([True, False]), 0.5)
        self.assertIsNone(citation_precision([]))


if __name__ == "__main__":
    unittest.main()
