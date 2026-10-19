from reposcouts.workflow.state import RepoScoutState
from typing import Literal

from reposcouts.stages.requirement_parser import (
    parse_user_requirements,
)

from reposcouts.stages.search_agent import (
    search_candidate_repositories,
)

from reposcouts.stages.repository_evaluator import (
    analyze_candidate_repositories,
)
from reposcouts.core.schemas import CriticResult

from reposcouts.stages.ranker import (
    rank_repositories,
)

from reposcouts.stages.critic import (
    critique_recommendation,
    validate_ranking_structure
)


def parse_requirements_node(
    state: RepoScoutState,
) -> dict:
    """
    Parse the user's natural-language request.
    """

    print("\n[Node] Parsing user requirements...")

    requirements = parse_user_requirements(
        state["user_request"]
    )

    return {
        "requirements": requirements,
    }


def search_candidates_node(
    state: RepoScoutState,
) -> dict:
    """
    Search GitHub for candidate repositories.
    """

    print("\n[Node] Searching candidate repositories...")

    requirements = state["requirements"]

    if requirements is None:
        raise ValueError(
            "requirements must exist before candidate search."
        )

    candidates = search_candidate_repositories(
        requirements
    )

    return {
        "candidates": candidates,
    }


def evaluate_repositories_node(
    state: RepoScoutState,
) -> dict:
    """
    Collect evidence and evaluate candidate repositories.
    """

    print("\n[Node] Evaluating repositories...")

    requirements = state["requirements"]
    candidates = state["candidates"]

    if requirements is None:
        raise ValueError(
            "requirements must exist before evaluation."
        )

    if candidates is None:
        raise ValueError(
            "candidates must exist before evaluation."
        )

    evaluations = analyze_candidate_repositories(
        requirements,
        candidates,
        critic_feedback=state.get("critic_feedback", ""),
    )

    return {
        "evaluations": evaluations,
    }


def rank_repositories_node(
    state: RepoScoutState,
) -> dict:
    """
    Rank evaluated repositories.
    """

    print("\n[Node] Ranking repositories...")

    requirements = state["requirements"]
    evaluations = state["evaluations"]

    if requirements is None:
        raise ValueError(
            "requirements must exist before ranking."
        )

    ranking = rank_repositories(
        requirements=requirements,
        evaluations=evaluations,
        critic_feedback=state.get(
            "critic_feedback",
            "",
        ),
    )

    return {
        "ranking": ranking,
    }


def critic_node(
    state: RepoScoutState,
) -> dict:
    """
    Review whether the final recommendation is valid.
    """

    print("\n[Node] Reviewing recommendation...")

    requirements = state["requirements"]
    evaluations = state["evaluations"]
    ranking = state["ranking"]

    if requirements is None:
        raise ValueError(
            "requirements are missing."
        )

    if ranking is None:
        raise ValueError(
            "ranking must exist before critique."
        )

    if not ranking.top_repositories and not ranking.best_repository:
        return {
            "critic_result": CriticResult(
                passed=True, confidence=100, issues=[], retry_target="none",
                feedback="No recommendation: available evidence is insufficient. No unsupported recommendation was made.",
            ),
            "critic_feedback": "insufficient_evidence",
        }

    critic_result = critique_recommendation(
        requirements=requirements,
        evaluations=evaluations,
        ranking=ranking,
    )

    return {
        "critic_result": critic_result,
        "critic_feedback": critic_result.feedback,
    }


def increment_retry_node(
    state: RepoScoutState,
) -> dict:
    """
    Increase retry counter before retrying.
    """

    retry_count = state["retry_count"] + 1

    print(
        f"\n[Node] Retry attempt "
        f"{retry_count}/{state['max_retries']}"
    )

    return {
        "retry_count": retry_count,
    }


def route_after_critic(
    state: RepoScoutState,
) -> Literal[
    "end",
    "retry",
]:
    """
    Decide whether to finish or retry.
    """

    critic_result = state["critic_result"]

    if critic_result is None:
        raise ValueError(
            "critic_result is missing."
        )

    if critic_result.passed:
        return "end"

    if (
        state["retry_count"]
        >= state["max_retries"]
    ):
        return "end"

    return "retry"


def route_retry_target(
    state: RepoScoutState,
) -> Literal[
    "ranker",
    "evaluator",
]:
    """
    Route retry according to Critic feedback.
    """

    critic_result = state["critic_result"]

    if critic_result is None:
        return "ranker"

    if critic_result.retry_target == "evaluator":
        return "evaluator"

    return "ranker"
