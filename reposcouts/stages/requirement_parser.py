from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from reposcouts.core.config import get_llm
from reposcouts.core.schemas import UserRequirements

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

Return the result as one valid JSON object.
Do not include Markdown code fences or any text outside the JSON object.

The JSON object must contain exactly these fields:
topic, language, technology_stack, difficulty, must_have,
nice_to_have, exclusions, learning_goal, time_budget_in_hours.

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
    method = "json_mode",   # deepseek-v4-flash does not support "function_calling"
)

requirement_parser = prompt | structured_llm


def parse_user_requirements(user_query: str) -> UserRequirements:
    """
    Parse a natural-language user query about repo into a structured UserRequirements object.
    """
    return requirement_parser.invoke(
        {
            "user_request": user_query
        }
    )

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
