from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from reposcouts.core.config import get_llm
from reposcouts.core.schemas import UserRequirements
from reposcouts.rag.planning import validate_parsed_payload, RequirementValidationError
import json

requirements_format_parser = PydanticOutputParser(
    pydantic_object=UserRequirements
)

SYSTEM_PROMPT = """
You are the requirement parser of RepoScout.

What the RepoScout do is that helps developers find GitHub repositories 
that best match their technical, learning, and engineering requirements.

Your job is to convert the user's natural-language request into
structured repository requirements.

Rules:

1. Preserve the user's actual intent.
2. Distinguish hard requirements from optional preferences.
3. Do not invent constraints that the user did not express.
4. Put mandatory conditions into must_have.
5. Put optional preferences into nice_to_have.
6. Put explicitly unwanted technologies or characteristics into exclusions.
7. If the user does not specify a programming language, return an empty list.
8. If the user does not specify a framework, return an empty list.
9. The difficulty field must be exactly one of:
   "beginner", "intermediate", "advanced", or "any".
   Normalize "easy", "novice", and "newbie" to "beginner".
   Never return any other value.
   If difficulty cannot be inferred, use "any".
10. If no time constraint is given, set time_budget_in_hours to null.
11. Interpret vague development-time expressions using practical working time:
    - "within one day" / "in a day" -> 8 hours
    - "half a day" -> 4 hours
    Only use 24 hours if the user explicitly means a full 24-hour period.
12. Keep independent mandatory capabilities as separate must_have entries.
    Preserve alternatives (A or B), negations, versions and qualifiers together.
    Explicitly mandatory frameworks belong in must_have as well as technology_stack.

13. Generate retrieval_plan in this SAME response (no repository-specific facts).
    Each entry has kind (must_have/technology/language/exclusion/context),
    parent_requirement copied EXACTLY from its corresponding field,
    source_quote copied EXACTLY from the user's text (not a translation),
    atomic_requirement, and 1-3 short retrieval queries (original identifiers
    plus useful Chinese/English expressions). Cover every must_have, framework,
    language and exclusion. Do not infer hard constraints from preferences.
    Preserve OR, negation and version-qualified requirements as one unchanged
    atomic_requirement. Synonyms are search hints, never additional requirements.
    Do not add Redis/FAISS/etc. as requirements unless explicitly requested.
    Leave plan_warnings empty; the application owns validation warnings.

The JSON object must contain exactly these fields:
topic, language, technology_stack, difficulty, must_have,
nice_to_have, exclusions, learning_goal, time_budget_in_hours,
retrieval_plan, plan_warnings.

Return the result as one valid JSON object.
Do not include Markdown or additional text.

Follow these JSON schema instructions exactly:

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            SYSTEM_PROMPT, 
        ), 
        (
            "human", 
            "{user_request}", 
        )
    ]
).partial(
    format_instructions=requirements_format_parser.get_format_instructions()
)

llm = get_llm()

structured_llm = llm.with_structured_output(
    UserRequirements, 
    method = "json_mode",   # Schema and provenance checks are enforced below.
    include_raw=True,
)

requirement_parser = prompt | structured_llm


def parse_user_requirements(user_query: str) -> UserRequirements:
    """
    Parse a natural-language user query about repo into a structured UserRequirements object.
    """
    response = requirement_parser.invoke(
        {
            "user_request": user_query
        }
    )
    # Read raw JSON too: a malformed retrieval plan can fall back without losing
    # independently validated base requirements and their verbatim anchors.
    try:
        payload = json.loads(response["raw"].content)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise RequirementValidationError("requirements_invalid_json") from None
    return validate_parsed_payload(payload, user_query)

if __name__ == "__main__": 

    example_request = (
            "I am a beginner looking for a Ai Agent projects. "
            "I prefer an overall architecture project but with an eazy to understand codebase. "
            "If the repo could be learned and reporduced within one day is better."
            "The repo should be designed for a specific project purpose, rather than work for work's sake. "
        )

    result = parse_user_requirements(example_request)
    print(type(result))
    print(result.model_dump_json(indent=2))
