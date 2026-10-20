# Parse the User requirements as research input.
# Date: 2026-09-06
# Author: zKarina

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict
from reposcouts.rag.schemas import EvidenceCitation, RepositoryResearch, RetrievalPlanItem

class UserRequirements(BaseModel): 
    """
    Structured requirements extracted from the user's repo request.
    The important fields in user requests are:
        - topic: what kind of repos user needed.
        - language: the programming language of the repos.
        - technology stack: the technology stack used in the repos.
        - must_have, nice_to_have: the features that the repos must have or nice to have.
        - etc...
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    topic: str = Field(
        description = (
            "The main technical topic or type of GitHub repository "
            "what the user wants to find."
        )
    )

    # projects may be written in multiple programming languages
    language: list[str] = Field(
        description = (
            "Preferred programming languages explicitly mentioned "
            "or strongly implied by the user."
            "If the user does not specify any language, "
            "return an empty list."
        )
    )

    technology_stack: list[str] = Field(
        description = (
            "The technology stack explicitly mentioned "
            "or strongly implied by the user."
            "Frameworks, libraries, or technologies the repository "
            "should preferably use."
        )
    )

    # user role decides the complexity of the repo.
    difficulty: Literal[
        "beginner",
        "intermediate",
        "advanced",
        "any",
    ] = Field(
        description="The desired learning or reproduction difficulty."
    )

    # Personal preferences and constraints
    must_have: list[str] = Field(
        description=(
            "Hard requirements. A repository should normally be rejected "
            "if it does not satisfy these conditions."
        )
    )

    nice_to_have: list[str] = Field(
        description=(
            "Optional preferences that improve the repository's ranking "
            "but are not mandatory."
        )
    )

    exclusions: list[str] = Field(
        description=(
            "Technologies, characteristics, or conditions the user "
            "explicitly wants to avoid."
        )
    )

    # Additional conditions or expectations of the user requests.
    learning_goal: str = Field(
        description=(
            "Why the user wants this repository, such as learning, "
            "reproduction, production reference, or integration."
        )
    )

    time_budget_in_hours: int | None = Field(
        description=(
            "Maximum number of hours the user expects to spend on the repository. "
            "Use null if no time constraint is given."
        )
    )
    retrieval_plan: list[RetrievalPlanItem] = Field(default_factory=list, max_length=32)
    plan_warnings: list[str] = Field(default_factory=list, max_length=8)


# ========================================

class CandidateRepository(BaseModel): 
    """
    Structured information about a candidate Github Repo with reason why it kept.
    """

    full_name: str = Field(
        description = (
            "GitHub repository identifier "
            "in owner/repository format."
        )
    )

    reason: str = Field(
        description=(
            "Why this repository is worth keeping "
            "as a candidate for deeper analysis."
        )
    )


class CandidateSearchResult(BaseModel): 
    """
    Structured candidate search result with user requirements.
    """

    search_queries: list[str] = Field(
        description = (
            "Github search queries using during "
            "candidate repository discovery."
        )
    )

    candidates: list[CandidateRepository] = Field(
        description = (
            "Candidate repositories discovered from the search queries, "
            "with reasons for their selection for deeper analysis."
        )
    )


# ========================================

class RepositoryEvidence(BaseModel):
    """
    Factual evidence collected from a GitHub repository.
    """

    full_name: str

    metadata: dict

    readme: str

    structure: dict

    research: RepositoryResearch | None = None


class RequirementCheck(BaseModel):
    """
    Evaluation of one hard user requirement.
    """

    requirement: str

    status: Literal[
        "satisfied",
        "not_satisfied",
        "uncertain",
    ]

    evidence: str

    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[EvidenceCitation] = Field(default_factory=list)
    implementation_evidence_ids: list[str] = Field(default_factory=list)
    integration_evidence_ids: list[str] = Field(default_factory=list)
    contradiction_evidence_ids: list[str] = Field(default_factory=list)
    implementation_level: Literal["unknown", "declared", "dependency", "implemented", "integrated"] = "unknown"


# ========================================

class RepositoryEvaluation(BaseModel):
    """
    Evaluation of one repository against user requirements.
    """

    full_name: str

    research_summary: dict = Field(default_factory=dict)

    purpose_summary: str

    must_have_checks: list[RequirementCheck]

    requirement_match: int = Field(
        ge=0,
        le=100,
        description=(
            "Percentage of hard requirements satisfied by the repository."
        )
    )

    learning_value: int = Field(
        ge=0,
        le=100,
        description=(
            "Assessment of the repository's value for learning purposes."
        )
    )

    code_clarity: int = Field(
        ge=0,
        le=100,
        description=(
            "Assessment of the repository's code clarity and readability."
        )
    )

    documentation_quality: int = Field(
        ge=0,
        le=100,
        description=(
            "Assessment of the repository's documentation quality."
            "Base on the README, inline comments, and any additional documentation provided."
        )
    )

    reproduction_feasibility: int = Field(
        ge=0,
        le=100,
        description=(
            "Assessment of how feasible it is to reproduce the repository's functionality."
            "Base on the provided instructions, dependencies, and overall complexity."
            "Also consider the user's time budget and skill level when evaluating this aspect."
        )
    )

    maintenance_health: int = Field(
        ge=0,
        le=100,
        description=(
            "Assessment of the repository's maintenance and health."
            "Base on the frequency of updates, issue resolution, and community engagement."
        )
    )

    strengths: list[str]

    weaknesses: list[str]

    uncertainties: list[str]

    suitable_for_next_stage: bool


# ========================================

class RankedRepository(BaseModel):
    full_name: str

    base_score: float
    score_rank: int

    final_rank: int

    recommendation_reason: str
    tradeoffs: list[str]
    learning_path: list[str]

    ranking_override: bool
    override_reason: str | None


class RankingResult(BaseModel):
    top_repositories: list[RankedRepository]

    best_repository: str

    final_reason: str


# ========================================

class CriticIssue(BaseModel):

    severity: Literal[
        "warning",
        "critical",
    ]

    category: Literal[
        "hard_requirement",
        "evidence",
        "ranking",
        "unsupported_claim",
    ]

    repository: str | None

    description: str


class CriticResult(BaseModel):

    passed: bool

    confidence: int = Field(
        ge=0,
        le=100,
    )

    issues: list[CriticIssue]

    retry_target: Literal[
        "none",
        "ranker",
        "evaluator",
    ]

    feedback: str


    

if __name__ == "__main__":
    # Example usage
    example_request = (
        "I am a beginner looking for a Ai Agent projects. "
        "I prefer an overall architecture project but with an eazy to understand codebase. "
        "If the repo could be learned and reporduced within one day is better."
        "The repo should be designed for a specific project purpose, rather than work for work's sake. "
    )
