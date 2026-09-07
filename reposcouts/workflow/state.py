from typing import TypedDict

from reposcouts.core.schemas import (
    UserRequirements,
    CandidateSearchResult,
    RepositoryEvaluation,
    RankingResult,
    CriticResult,
)


class RepoScoutState(TypedDict):
    """
    Global state flowing through the RepoScout LangGraph workflow.
    """

    # Original user input
    user_request: str

    # Module 1
    requirements: UserRequirements | None

    # Module 3
    candidates: CandidateSearchResult | None

    # Module 4
    evaluations: list[RepositoryEvaluation]

    # Module 5
    ranking: RankingResult | None

    # Module 6
    critic_result: CriticResult | None

    # Retry control
    retry_count: int
    max_retries: int

    # Feedback passed from Critic to retry nodes
    critic_feedback: str
