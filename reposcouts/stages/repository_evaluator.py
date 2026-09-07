from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from reposcouts.core.config import get_llm
from reposcouts.github.repository_collector import collect_repository_evidence

from reposcouts.core.schemas import (
    CandidateSearchResult,  # candidate repo search result
    UserRequirements,       # user requirements
    RepositoryEvidence,     # repo fact
    RepositoryEvaluation,   # evaluation of repo
)

evaluation_format_parser = PydanticOutputParser(
    pydantic_object=RepositoryEvaluation
)

EVALUATOR_SYSTEM_PROMPT = """
You are the Repository Evaluator of RepoScout.

Your job is to evaluate ONE GitHub repository against
the user's technical and learning requirements.

You must base your evaluation ONLY on the supplied
repository evidence.

The repository evidence contains:

- GitHub metadata
- README
- repository file structure

Rules:

1. Never invent facts about the repository.

2. If the available evidence is insufficient to verify
   something, mark it as uncertain.

3. Evaluate every must-have requirement individually.

4. A repository failing an important must-have requirement
   should receive a substantially lower requirement_match score.

5. Do not assume a popular repository is automatically better.

6. Stars and forks are weak signals only.

7. When evaluating learning_value, consider whether the repository
   exposes meaningful Agent concepts and whether a learner can
   understand the end-to-end architecture.

8. When evaluating code_clarity, use the repository structure,
   project organization, apparent component count, and README.

9. When evaluating reproduction_feasibility, explicitly consider
   the user's time budget, project dependencies, setup complexity,
   and number of major components.

10. Do not compare this repository with other repositories.
    Ranking happens in another module.

11. suitable_for_next_stage means the repository is credible enough
    to participate in the final ranking.

Return exactly one valid JSON object matching the required schema.
Follow these JSON schema instructions exactly:

{format_instructions}

Do not output Markdown or additional text.
"""

evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            EVALUATOR_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
            USER REQUIREMENTS:

            {requirements}


            REPOSITORY EVIDENCE:

            {evidence}
            """,
        ),
    ]
).partial(
    format_instructions=evaluation_format_parser.get_format_instructions()
)

evaluator_llm = get_llm(thinking=False)

structured_evaluator_llm = (
    evaluator_llm.with_structured_output(
        RepositoryEvaluation,   # output evaluation result
        method="json_mode",
    )
)

repository_evaluator = evaluator_prompt | structured_evaluator_llm

# define the evaluation interface
def evaluate_repository(
    requirements: UserRequirements,
    evidence: RepositoryEvidence,
) -> RepositoryEvaluation:
    """
    Evaluate one repository using collected evidence.
    """

    return repository_evaluator.invoke(
        {
            "requirements":
                requirements.model_dump_json(
                    indent=2
                ),

            "evidence":
                evidence.model_dump_json(
                    indent=2
                ),
        }
    )


def analyze_candidate_repositories(
    requirements: UserRequirements,
    candidates: CandidateSearchResult,
) -> list[RepositoryEvaluation]:

    # each candidate with own evaluation
    evaluations = []

    for candidate in candidates.candidates:

        print(f"\nAnalyzing: {candidate.full_name}")

        evidence = collect_repository_evidence(
            candidate
        )

        evaluation = evaluate_repository(
            requirements=requirements,
            evidence=evidence,
        )

        evaluations.append(
            evaluation
        )

    return evaluations


if __name__ == "__main__":

    # what the user input to search
    user_input = (
        "I want to learn how to build an AI Agent using LangGraph. "
        "I want a Python repository that is well-structured, has a clear README, and is feasible to reproduce within 2 hours. "
        "The repository should have a good learning value, with clear code and architecture. "
    )

    from reposcouts.stages.requirement_parser import parse_user_requirements
    requirements = parse_user_requirements(
        user_input
    )

    from reposcouts.stages.search_agent import search_candidate_repositories
    candidates = search_candidate_repositories(
        requirements
    )

    evaluations = analyze_candidate_repositories(
        requirements=requirements,
        candidates=candidates,
    )

    print("\n========== Repository Evaluations ==========")
    for evaluation in evaluations:
        print(
            evaluation.model_dump_json(
                indent=2
            )
        )

