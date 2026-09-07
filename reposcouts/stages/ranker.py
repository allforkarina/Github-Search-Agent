import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from reposcouts.core.config import get_llm
from reposcouts.core.schemas import (
    RankingResult,
    UserRequirements,
    RepositoryEvaluation, 
)

RANKING_WEIGHTS = {
    "requirement_match": 0.30,
    "learning_value": 0.25,
    "reproduction_feasibility": 0.20,
    "code_clarity": 0.15,
    "documentation_quality": 0.07,
    "maintenance_health": 0.03,
}

def calculate_repository_score(
    evaluation: RepositoryEvaluation,
) -> float:
    """
    Calculate the deterministic ranking score
    for one repository.
    """

    score = (
        evaluation.requirement_match
        * RANKING_WEIGHTS["requirement_match"]

        + evaluation.learning_value
        * RANKING_WEIGHTS["learning_value"]

        + evaluation.reproduction_feasibility
        * RANKING_WEIGHTS["reproduction_feasibility"]

        + evaluation.code_clarity
        * RANKING_WEIGHTS["code_clarity"]

        + evaluation.documentation_quality
        * RANKING_WEIGHTS["documentation_quality"]

        + evaluation.maintenance_health
        * RANKING_WEIGHTS["maintenance_health"]
    )

    return round(score, 2)


def filter_repositories(
    evaluations: list[RepositoryEvaluation],
) -> list[RepositoryEvaluation]:

    return [
        evaluation
        for evaluation in evaluations
        if evaluation.suitable_for_next_stage
    ]


def rank_by_score(
    evaluations: list[RepositoryEvaluation],
) -> list[tuple[RepositoryEvaluation, float]]:

    valid_evaluations = filter_repositories(evaluations)

    scored_repositories = [
        (
            evaluation,
            calculate_repository_score(
                evaluation
            ),
        )
        for evaluation in valid_evaluations
    ]

    scored_repositories.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_repositories


# ========================================

ranker_format_parser = PydanticOutputParser(
    pydantic_object=RankingResult,
)

RANKER_SYSTEM_PROMPT = """
You are the Final Repository Ranker of RepoScout.

Your job is to compare the top candidate repositories
that have already been independently evaluated.

You are NOT responsible for re-evaluating repositories
from scratch.

The numeric score has already been calculated using
a deterministic ranking formula.

Your responsibilities are:

1. Respect the deterministic score ordering unless
   there is a strong evidence-based reason not to.

2. Compare the repositories specifically against
   the user's requirements and learning goal.

3. Explain why a repository is better suited than
   the alternatives.

4. Clearly describe important tradeoffs.

5. Do not invent repository features.

6. Use only the supplied RepositoryEvaluation data.

7. Produce a practical learning path for each repository.

8. The learning path should explain what the user
   should inspect or understand first, rather than
   inventing filenames that are not present in the evidence.

9. Select one best_repository.

Return exactly one valid JSON object matching
the required schema.

Follow these JSON schema instructions exactly:

{format_instructions}

Do not output Markdown or additional text.
"""

def build_ranking_context(
    scored_repositories,
) -> str:
    """
    Build a JSON object of evaluation results.
    Which will be used as context for the final ranking step.
    """

    ranking_data = []

    for evaluation, score in scored_repositories:

        ranking_data.append(
            {
                "calculated_score": score,
                "evaluation":
                    evaluation.model_dump(),
            }
        )

    return json.dumps(
        ranking_data,
        indent=2,
        ensure_ascii=False,
    )


ranker_llm = get_llm()

structured_ranker_llm = (
    ranker_llm.with_structured_output(
        RankingResult,
        method="json_mode",
    )
)

ranker_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            RANKER_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
            USER REQUIREMENTS:

            {requirements}


            TOP REPOSITORIES:

            {repositories}

            
            CRITIC FEEDBACK FROM PREVIOUS ATTEMPT:

            {critic_feedback}   
            """,
        ),
    ]
).partial(
    format_instructions=ranker_format_parser.get_format_instructions()
)

repository_ranker = ranker_prompt | structured_ranker_llm

def rank_repositories(
    requirements: UserRequirements,
    evaluations: list[RepositoryEvaluation],
    top_k: int = 3,
    critic_feedback: str | None = None,
) -> RankingResult:

    # cal the score for each repo based on the evaluation results
    scored_repositories = rank_by_score(evaluations)

    top_candidates = (
        scored_repositories[:top_k]
    )

    ranking_context = (
        build_ranking_context(
            top_candidates
        )
    )

    result = repository_ranker.invoke(
        {
            "requirements":
                requirements.model_dump_json(
                    indent=2
                ),

            "repositories":
                ranking_context,

            "critic_feedback":
                critic_feedback or "None",
        }
    )

    return result


if __name__ == "__main__":

    # what the user input to search
    user_input = (
        "I want to learn how to build an AI Agent using LangGraph. "
        "I want a Python repository that is well-structured, has a clear README, and is feasible to reproduce within 2 hours. "
        "The repository should have a good learning value, with clear code and architecture. "
    )

    print("\n============ User Requirements ============\n")
    from reposcouts.stages.requirement_parser import parse_user_requirements
    requirements = parse_user_requirements(
        user_input
    )
    print(
        f"Parsed User Requirements:\n"
        f"{requirements.model_dump_json(indent=2)}"
    )


    print("\n============ Candidate Repositories ============\n")
    from reposcouts.stages.search_agent import search_candidate_repositories
    candidates = search_candidate_repositories(
        requirements=requirements
    )
    print(f"Found Candidate Repositories:\n")
    for candidate in candidates.candidates:
        print(
            candidate.model_dump_json(indent=2)
        )


    print("\n============ Repository Evaluations ============\n")
    from reposcouts.stages.repository_evaluator import analyze_candidate_repositories
    evaluations = analyze_candidate_repositories(
        candidates=candidates,
        requirements=requirements
    )
    print(f"Repository Evaluations:\n")
    for evaluation in evaluations:
        print(
            evaluation.model_dump_json(indent=2)
        )


    print("\n============ Repository Ranking ============\n")
    ranking_result = rank_repositories(
        requirements=requirements,
        evaluations=evaluations,
        top_k=3,
    )
    print(
        ranking_result.model_dump_json(indent=2)
    )

    
