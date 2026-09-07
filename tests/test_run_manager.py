import threading
import unittest

from reposcouts.api.run_manager import (
    RunConflictError,
    RunManager,
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
    def stream(self, state, stream_mode):
        self.state = state
        self.stream_mode = stream_mode
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

    def stream(self, state, stream_mode):
        self.started.set()
        self.release.wait(timeout=2)
        yield {
            "parse_requirements": {
                "requirements": sample_requirements(),
            }
        }


def collect_events(record):
    events = []
    while True:
        event = record.events.get(timeout=2)
        events.append(event)
        if event.type in TERMINAL_EVENT_TYPES:
            return events


class RunManagerTests(unittest.TestCase):
    def test_streams_workflow_updates_and_final_result(self):
        graph = SuccessfulGraph()
        manager = RunManager(graph=graph)

        record = manager.create_run("Find a beginner LangGraph repository")
        events = collect_events(record)

        self.assertEqual(record.status, "completed")
        self.assertEqual(graph.stream_mode, "updates")
        self.assertEqual(graph.state["user_request"], "Find a beginner LangGraph repository")
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


if __name__ == "__main__":
    unittest.main()
