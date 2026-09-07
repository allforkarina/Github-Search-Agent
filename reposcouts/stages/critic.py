import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from reposcouts.core.config import get_llm
from reposcouts.core.schemas import (
    UserRequirements,
    RepositoryEvaluation,
    RankingResult,
    CriticResult,
)

critic_format_parser = PydanticOutputParser(
    pydantic_object=CriticResult
)

CRITIC_SYSTEM_PROMPT = """
You are the Recommendation Critic of RepoScout.

Your responsibility is NOT to recommend repositories again.

Your responsibility is to inspect whether the final recommendation
made by the Ranker is logically consistent, supported by evidence,
and aligned with the user's requirements.

Check the following:

1. HARD REQUIREMENTS
   The selected best repository must not violate any explicit
   must-have requirement.

2. EVIDENCE CONSISTENCY
   Recommendation claims must be supported by RepositoryEvaluation.

3. RANKING CONSISTENCY
   If the final ranking differs from the deterministic score ranking,
   there must be a strong and explicit user-specific reason.

4. UNSUPPORTED CLAIMS
   Do not allow claims about code size, architecture, dependencies,
   documentation, features, or reproduction difficulty unless they
   are supported by the supplied evaluation.

5. USER ALIGNMENT
   The final recommendation must prioritize the user's actual
   learning goal and constraints rather than popularity.

Rules:

- Do NOT re-evaluate repositories from scratch.
- Do NOT invent new repository information.
- Do NOT generate a new ranking.
- Minor limitations should be warnings.
- Logical contradictions or hard-requirement violations are critical.
- passed must be false if any critical issue exists.
- If the ranking explanation is the problem, retry_target = "ranker".
- If the underlying RepositoryEvaluation appears internally
  inconsistent, retry_target = "evaluator".
- Otherwise retry_target = "none".

Return exactly one valid JSON object.
Follow these JSON schema instructions exactly:

{format_instructions}

Do not output Markdown or extra text.
"""

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CRITIC_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
            USER REQUIREMENTS:

            {requirements}


            REPOSITORY EVALUATIONS:

            {evaluations}


            FINAL RANKING:

            {ranking}
            """,
        ),
    ]
).partial(
    format_instructions=critic_format_parser.get_format_instructions()
)

critic_llm = get_llm(
    thinking=False
)

structured_critic_llm = (
    critic_llm.with_structured_output(
        CriticResult,
        method="json_mode",
    )
)

recommendation_critic = critic_prompt | structured_critic_llm


def critique_recommendation(
    requirements: UserRequirements,
    evaluations: list[RepositoryEvaluation],
    ranking: RankingResult,
) -> CriticResult:

    top_names = {
        repo.full_name
        for repo in ranking.top_repositories
    }

    top_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.full_name in top_names
    ]

    return recommendation_critic.invoke(
        {
            "requirements":
                requirements.model_dump_json(
                    indent=2
                ),

            "evaluations":
                json.dumps(
                    [
                        evaluation.model_dump()
                        for evaluation
                        in top_evaluations
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),

            "ranking":
                ranking.model_dump_json(
                    indent=2
                ),
        }
    )


def validate_ranking_structure(
    ranking: RankingResult,
) -> list[str]:

    errors = []

    names = [
        repo.full_name
        for repo in ranking.top_repositories
    ]

    if ranking.best_repository not in names:
        errors.append(
            "best_repository is not contained "
            "in top_repositories."
        )

    ranks = [
        repo.final_rank
        for repo in ranking.top_repositories
    ]

    if len(ranks) != len(set(ranks)):
        errors.append(
            "Duplicate final ranks detected."
        )

    return errors

