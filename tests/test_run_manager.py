import os
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from langsmith.run_helpers import get_tracing_context
from langsmith import Client
from langgraph.graph import END, START, StateGraph
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

# The workflow creates model clients on import; tests never need a real key.
with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-deepseek-key"}):
    import reposcouts.api.run_manager as run_manager_module
    from reposcouts.api.run_manager import (
        RunConflictError,
        RunManager,
        RunRecord,
        TERMINAL_EVENT_TYPES,
    )
from reposcouts.core.schemas import (
    CandidateRepository,
    CandidateSearchResult,
    CriticResult,
    RankedRepository,
    RankingResult,
    RepositoryEvaluation,
    UserRequirements,
)
from reposcouts.workflow.state import RepoScoutState
from reposcouts.workflow.nodes import (
    increment_retry_node, route_after_critic, route_retry_target,
)
from reposcouts.github.tools import get_repository_readme


def sample_requirements() -> UserRequirements:
    return UserRequirements(
        topic="LangGraph agent",
        language=["Python"],
        technology_stack=["LangGraph"],
        difficulty="beginner",
        must_have=["clear README"],
        nice_to_have=["small codebase"],
        exclusions=[],
        learning_goal="Learn an agent workflow",
        time_budget_in_hours=2,
    )


def sample_evaluation() -> RepositoryEvaluation:
    return RepositoryEvaluation(
        full_name="example/agent",
        purpose_summary="A small agent example.",
        must_have_checks=[
            {
                "requirement": "clear README",
                "status": "satisfied",
                "evidence": "README includes setup instructions.",
            }
        ],
        requirement_match=95,
        learning_value=90,
        code_clarity=88,
        documentation_quality=92,
        reproduction_feasibility=86,
        maintenance_health=75,
        strengths=["Clear structure"],
        weaknesses=["Small community"],
        uncertainties=[],
        suitable_for_next_stage=True,
    )


class SuccessfulGraph:
    def stream(self, state, stream_mode, config=None):
        self.state = state
        self.stream_mode = stream_mode
        self.config = config
        self.tracing = get_tracing_context()
        requirements = sample_requirements()
        candidates = CandidateSearchResult(
            search_queries=["langgraph agent language:python"],
            candidates=[
                CandidateRepository(
                    full_name="example/agent",
                    reason="Matches the requested stack.",
                )
            ],
        )
        evaluation = sample_evaluation()
        ranking = RankingResult(
            top_repositories=[
                RankedRepository(
                    full_name="example/agent",
                    base_score=89.5,
                    score_rank=1,
                    final_rank=1,
                    recommendation_reason="Strong fit for a short learning project.",
                    tradeoffs=["Small community"],
                    learning_path=["Read the README", "Trace the graph"],
                    ranking_override=False,
                    override_reason=None,
                )
            ],
            best_repository="example/agent",
            final_reason="Best match for the stated goal.",
        )
        critic = CriticResult(
            passed=True,
            confidence=94,
            issues=[],
            retry_target="none",
            feedback="Recommendation is supported by the evaluation.",
        )

        yield {"parse_requirements": {"requirements": requirements}}
        yield {"search_candidates": {"candidates": candidates}}
        yield {"evaluate_repositories": {"evaluations": [evaluation]}}
        yield {"rank_repositories": {"ranking": ranking}}
        yield {
            "critic": {
                "critic_result": critic,
                "critic_feedback": critic.feedback,
            }
        }


class BlockingGraph:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def stream(self, state, stream_mode, config=None):
        try:
            self.started.set()
            self.release.wait(timeout=2)
            yield {
                "parse_requirements": {
                    "requirements": sample_requirements(),
                }
            }
        finally:
            self.closed.set()


class FailingGraph:
    def __init__(self) -> None:
        self.closed = threading.Event()

    def stream(self, state, stream_mode, config=None):
        try:
            yield {
                "parse_requirements": {
                    "requirements": sample_requirements(),
                }
            }
            raise ValueError("Offline graph failure.")
        finally:
            self.closed.set()


def collect_events(record):
    events = []
    while True:
        event = record.events.get(timeout=2)
        events.append(event)
        if event.type in TERMINAL_EVENT_TYPES:
            return events


class RunManagerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(
            os.environ,
            {
                "REPOSCOUTS_TRACING_ENABLED": "false",
                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
                "LANGSMITH_PROJECT": "test-observability",
                "REPOSCOUTS_ENVIRONMENT": "test",
                "REPOSCOUTS_AGENT_VERSION": "test-commit",
                "REPOSCOUTS_PROMPT_VERSION": "test-prompt",
                "DEEPSEEK_MODEL": "test-model",
                "GITHUB_TOKEN": "test-github-token",
                "DEEPSEEK_API_KEY": "test-deepseek-key",
                "LANGSMITH_API_KEY": "test-langsmith-key",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

        factory = patch.object(
            run_manager_module,
            "build_trace_client",
            side_effect=AssertionError("Unexpected trace client creation."),
        )
        self.build_trace_client = factory.start()
        self.addCleanup(factory.stop)

        network = patch(
            "requests.Session.request",
            side_effect=AssertionError("Network requests are forbidden in unit tests."),
        )
        network.start()
        self.addCleanup(network.stop)

    def test_streams_workflow_updates_and_final_result(self):
        graph = SuccessfulGraph()
        manager = RunManager(graph=graph)

        with patch.object(graph, "stream", wraps=graph.stream) as stream:
            record = manager.create_run("Find a beginner LangGraph repository")
            events = collect_events(record)
            stream.assert_called_once()

        self.assertEqual(record.status, "completed")
        self.assertEqual(graph.stream_mode, "updates")
        self.assertEqual(graph.state["user_request"], "Find a beginner LangGraph repository")
        self.assertEqual(
            graph.config,
            {
                "run_name": "RepoScoutsRequest",
                "callbacks": [],
                "metadata": {
                    "run_id": record.run_id,
                    "environment": "test",
                    "agent_version": "test-commit",
                    "model_name": "test-model",
                    "prompt_version": "test-prompt",
                    "retry_count": 0,
                },
            },
        )
        self.assertFalse(graph.tracing["enabled"])
        self.build_trace_client.assert_not_called()
        self.assertEqual(events[0].type, "run.started")
        self.assertEqual(events[-1].type, "run.completed")
        completed_nodes = [
            event.node
            for event in events
            if event.type == "node.completed"
        ]
        self.assertEqual(
            completed_nodes,
            [
                "parse_requirements",
                "search_candidates",
                "evaluate_repositories",
                "rank_repositories",
                "critic",
            ],
        )
        ranking = events[-1].data["ranking"]
        self.assertEqual(ranking["best_repository"], "example/agent")
        self.assertEqual(
            record.summary,
            {
                "execution_status": "completed",
                "retry_count": 0,
                "critic_passed": True,
                "outcome": "passed",
            },
        )

    def test_rejects_a_second_active_run(self):
        graph = BlockingGraph()
        manager = RunManager(graph=graph)
        record = manager.create_run("First request")
        self.assertTrue(graph.started.wait(timeout=2))

        with self.assertRaises(RunConflictError):
            manager.create_run("Second request")

        manager.cancel_run(record.run_id)
        graph.release.set()
        collect_events(record)

    def test_cancels_between_nodes(self):
        graph = BlockingGraph()
        manager = RunManager(graph=graph)
        record = manager.create_run("Find a repository")
        self.assertTrue(graph.started.wait(timeout=2))

        cancelled = manager.cancel_run(record.run_id)
        self.assertEqual(cancelled.status, "cancelling")
        graph.release.set()

        events = collect_events(record)
        self.assertEqual(record.status, "cancelled")
        self.assertEqual(events[-1].type, "run.cancelled")
        self.assertTrue(graph.closed.wait(timeout=2))
        self.assertEqual(record.summary["outcome"], "cancelled")

    def test_disables_global_tracing_when_feature_is_disabled(self):
        graph = SuccessfulGraph()
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            record = RunManager(graph=graph).create_run("Find a repository")
            events = collect_events(record)

        self.assertEqual(events[-1].type, "run.completed")
        self.assertFalse(graph.tracing["enabled"])
        self.assertEqual(graph.config["callbacks"], [])
        self.build_trace_client.assert_not_called()

    def test_tracing_is_disabled_when_feature_flag_is_missing(self):
        del os.environ["REPOSCOUTS_TRACING_ENABLED"]
        graph = SuccessfulGraph()
        record = RunManager(graph=graph).create_run("Find a repository")
        events = collect_events(record)

        self.assertEqual(events[-1].type, "run.completed")
        self.assertFalse(graph.tracing["enabled"])
        self.build_trace_client.assert_not_called()

    def test_enables_tracing_with_sanitizer_secrets_and_closes_client(self):
        graph = SuccessfulGraph()
        client = Mock()
        client_closed = threading.Event()
        client.close.side_effect = lambda **kwargs: client_closed.set()

        with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": " TRUE "}):
            with patch.object(
                run_manager_module, "build_trace_client", return_value=client
            ) as factory:
                record = RunManager(graph=graph).create_run("Find a repository")
                events = collect_events(record)
                self.assertTrue(client_closed.wait(timeout=2))

        self.assertEqual(events[-1].type, "run.completed")
        self.assertTrue(graph.tracing["enabled"])
        self.assertIs(graph.tracing["client"], client)
        self.assertEqual(graph.tracing["project_name"], "test-observability")
        self.assertEqual(len(graph.config["callbacks"]), 1)
        tracer = graph.config["callbacks"][0]
        self.assertIsInstance(tracer, LangChainTracer)
        self.assertIs(tracer.client, client)
        self.assertEqual(tracer.project_name, "test-observability")
        factory.assert_called_once_with(
            secrets=("test-github-token", "test-deepseek-key", "test-langsmith-key"),
            max_string_chars=4000,
        )
        client.close.assert_called_once_with(timeout=5.0)

    def test_explicit_tracer_captures_one_graph_root(self):
        def critic(state):
            return {
                "critic_result": CriticResult(
                    passed=True,
                    confidence=90,
                    issues=[],
                    retry_target="none",
                    feedback="Offline verification.",
                )
            }

        builder = StateGraph(RepoScoutState)
        builder.add_node("critic", critic)
        builder.add_edge(START, "critic")
        builder.add_edge("critic", END)
        manager = RunManager(graph=builder.compile())
        record = RunRecord(run_id="application-request", user_request="Offline request")
        manager._current = record
        client = Client(
            api_key="lsv2_pt_test_offline_key",
            api_url="http://127.0.0.1:1",
            auto_batch_tracing=False,
        )
        self.addCleanup(client.close, timeout=0)

        with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "true"}):
            with patch.object(
                run_manager_module, "build_trace_client", return_value=client
            ), patch.object(Client, "create_run") as create, patch.object(
                Client, "update_run"
            ) as update:
                manager._execute(record)

        roots = [
            call.kwargs for call in create.call_args_list
            if call.kwargs.get("parent_run_id") is None
        ]
        self.assertEqual(record.status, "completed")
        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertEqual(root["name"], "RepoScoutsRequest")
        self.assertEqual(root["extra"]["metadata"]["run_id"], record.run_id)
        self.assertNotEqual(str(root["id"]), record.run_id)
        children = [
            call.kwargs for call in create.call_args_list
            if call.kwargs.get("parent_run_id") == root["id"]
        ]
        self.assertEqual([child["name"] for child in children], ["critic"])
        self.assertTrue(
            any(call.kwargs.get("run_id") == root["id"] for call in update.call_args_list)
        )
        self.assertEqual(record.summary["outcome"], "passed")
        root_updates = [call.kwargs for call in update.call_args_list if call.kwargs["run_id"] == root["id"]]
        self.assertEqual(len(root_updates), 1)
        final_update = root_updates[0]
        self.assertEqual(final_update["outputs"]["observability"], record.summary)
        self.assertEqual(final_update["extra"]["metadata"]["prompt_version"], "test-prompt")
        self.assertEqual(final_update["extra"]["metadata"]["outcome"], "passed")

    def test_native_trace_includes_nested_llm_tool_latency_and_critic_retry(self):
        model = FakeMessagesListChatModel(responses=[AIMessage(
            content="Offline response",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )])

        def rank(state):
            model.invoke("Offline prompt")
            get_repository_readme.invoke({"full_name": "example/agent"})
            return {}

        def critic(state):
            passed = state["retry_count"] > 0
            return {"critic_result": CriticResult(
                passed=passed, confidence=90, issues=[],
                retry_target="none" if passed else "ranker",
                feedback="Passed." if passed else "Please retry ranking.",
            )}

        builder = StateGraph(RepoScoutState)
        builder.add_node("rank_repositories", rank)
        builder.add_node("critic", critic)
        builder.add_node("increment_retry", increment_retry_node)
        builder.add_edge(START, "rank_repositories")
        builder.add_edge("rank_repositories", "critic")
        builder.add_conditional_edges(
            "critic", route_after_critic, {"end": END, "retry": "increment_retry"}
        )
        builder.add_conditional_edges(
            "increment_retry", route_retry_target, {"ranker": "rank_repositories"}
        )
        manager = RunManager(graph=builder.compile())
        record = RunRecord(run_id="retry-request", user_request="Offline request")
        manager._current = record
        client = Client(
            api_key="lsv2_pt_test_offline_key", api_url="http://127.0.0.1:1",
            auto_batch_tracing=False,
        )
        self.addCleanup(client.close, timeout=0)
        response = Mock(status_code=404)

        with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "true"}), \
                patch.object(run_manager_module, "build_trace_client", return_value=client), \
                patch("reposcouts.github.tools.requests.get", return_value=response), \
                patch.object(Client, "create_run") as create, \
                patch.object(Client, "update_run") as update, redirect_stdout(StringIO()):
            manager._execute(record)

        spans = {call.kwargs["id"]: call.kwargs for call in create.call_args_list}
        roots = [span for span in spans.values() if span.get("parent_run_id") is None]
        self.assertEqual(len(roots), 1)
        root = roots[0]
        llms = [span for span in spans.values() if span["run_type"] == "llm"]
        tools = [span for span in spans.values() if span["run_type"] == "tool"]
        self.assertEqual(len(llms), 2)
        self.assertEqual([span["name"] for span in tools], ["get_repository_readme"] * 2)
        self.assertEqual(
            len([span for span in spans.values() if span["name"] == "critic"]), 2
        )
        self.assertEqual(record.summary["retry_count"], 1)
        self.assertEqual(record.summary["outcome"], "passed")
        for span in llms + tools:
            self.assertEqual(spans[span["parent_run_id"]]["name"], "rank_repositories")
            self.assertEqual(span["trace_id"], root["id"])

        # Native end timestamps provide latency, and LLM messages carry reported usage.
        endings = {call.kwargs["run_id"]: call.kwargs for call in update.call_args_list}
        for span_id, span in spans.items():
            self.assertGreaterEqual(endings[span_id]["end_time"], span["start_time"])
        for span in llms:
            generation = endings[span["id"]]["outputs"]["generations"][0][0]
            self.assertEqual(generation["message"]["kwargs"]["usage_metadata"]["total_tokens"], 15)

    def test_native_summary_matches_terminal_state_and_uses_one_root_update(self):
        from reposcouts.observability import tracing as tracing_module

        for case in ("passed", "retry_exhausted", "failed", "cancelled", "summary_error"):
            with self.subTest(case=case):
                def first(state):
                    return {"retry_count": 2 if case == "retry_exhausted" else 1}

                def critic(state):
                    if case == "failed":
                        raise ValueError("Offline failure")
                    return {"critic_result": CriticResult(
                        passed=case != "retry_exhausted", confidence=90,
                        issues=[], retry_target="none", feedback="Offline result",
                    )}

                builder = StateGraph(RepoScoutState)
                builder.add_node("parse_requirements", first)
                builder.add_node("critic", critic)
                builder.add_edge(START, "parse_requirements")
                builder.add_edge("parse_requirements", "critic")
                builder.add_edge("critic", END)
                manager = RunManager(graph=builder.compile())
                record = RunRecord(run_id=f"offline-{case}", user_request="Offline request")
                manager._current = record
                if case == "cancelled":
                    record.cancel_requested.set()
                client = Client(api_key="offline-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False)
                self.addCleanup(client.close, timeout=0)

                def factory(**kwargs):
                    if case == "summary_error":
                        kwargs["summary_provider"] = Mock(side_effect=RuntimeError("Private failure text"))
                    return tracing_module.build_run_tracer(**kwargs)

                with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "true"}), \
                        patch.object(run_manager_module, "build_trace_client", return_value=client), \
                        patch.object(run_manager_module, "build_run_tracer", side_effect=factory), \
                        patch.object(Client, "create_run") as create, \
                        patch.object(Client, "update_run") as update, \
                        patch.object(Client, "close", autospec=True) as close, \
                        patch.object(tracing_module.logger, "warning") as warning:
                    manager._execute(record)

                roots = [call.kwargs for call in create.call_args_list if call.kwargs.get("parent_run_id") is None]
                self.assertEqual(len(roots), 1)
                root_updates = [call.kwargs for call in update.call_args_list if call.kwargs["run_id"] == roots[0]["id"]]
                self.assertEqual(len(root_updates), 1)
                root = root_updates[0]
                if case == "summary_error":
                    self.assertEqual(record.summary["outcome"], "passed")
                    self.assertNotIn("observability", root.get("outputs") or {})
                    warning.assert_called_once_with("Failed to attach run summary to root trace.")
                else:
                    self.assertEqual(record.summary["outcome"], case)
                    self.assertEqual(root["outputs"]["observability"], record.summary)
                    for key, value in record.summary.items():
                        self.assertEqual(root["extra"]["metadata"][key], value)
                    warning.assert_not_called()
                close.assert_called_once_with(client, timeout=5.0)

    def test_closes_stream_and_trace_client_on_cancellation(self):
        graph = BlockingGraph()
        self.addCleanup(graph.closed.wait, 2)
        self.addCleanup(graph.release.set)
        client = Mock()
        client_closed = threading.Event()
        client.close.side_effect = lambda **kwargs: client_closed.set()

        with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "true"}):
            with patch.object(
                run_manager_module, "build_trace_client", return_value=client
            ):
                manager = RunManager(graph=graph)
                record = manager.create_run("Find a repository")
                self.assertTrue(graph.started.wait(timeout=2))
                manager.cancel_run(record.run_id)
                graph.release.set()
                events = collect_events(record)
                self.assertTrue(client_closed.wait(timeout=2))

        self.assertEqual(record.status, "cancelled")
        self.assertEqual(events[-1].type, "run.cancelled")
        self.assertTrue(graph.closed.is_set())
        client.close.assert_called_once_with(timeout=5.0)

    def test_closes_stream_and_trace_client_on_failure(self):
        graph = FailingGraph()
        client = Mock()
        client_closed = threading.Event()
        client.close.side_effect = lambda **kwargs: client_closed.set()

        with patch.dict(os.environ, {"REPOSCOUTS_TRACING_ENABLED": "true"}):
            with patch.object(
                run_manager_module, "build_trace_client", return_value=client
            ):
                record = RunManager(graph=graph).create_run("Find a repository")
                events = collect_events(record)
                self.assertTrue(client_closed.wait(timeout=2))

        self.assertEqual(record.status, "failed")
        self.assertEqual(events[-1].type, "run.failed")
        self.assertEqual(events[-1].data["error_type"], "ValueError")
        self.assertEqual(record.summary["outcome"], "failed")
        self.assertEqual(events[-1].node, "search_candidates")
        self.assertTrue(graph.closed.is_set())
        client.close.assert_called_once_with(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
