# Search Agent for Finding Candidate GitHub Repositories.
# Date: 2026-09-06
# Author: zKarina

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser

from reposcouts.core.config import get_llm
from reposcouts.core.schemas import (
    UserRequirements, 
    CandidateSearchResult,
)

candidate_format_parser = PydanticOutputParser(
    pydantic_object=CandidateSearchResult
)

from reposcouts.github.tools import search_github_repositories

# prompt engineering.
SEARCH_AGENT_SYSTEM_PROMPT = """
You are the Candidate Search Agent of RepoScout.

RepoScout helps developers find GitHub repositories that match
their learning and engineering requirements.

Your responsibility is candidate discovery.

You have access to the GitHub repository search tool.

Your workflow:

1. Carefully understand the structured user requirements.

2. Generate useful GitHub search queries based on:
   - topic
   - programming language
   - technology stack
   - must-have requirements
   - learning goal

3. Call the GitHub search tool to discover repositories.

4. If the results are too broad or irrelevant, modify the query
   and search again.

5. Use different search strategies when useful rather than
   repeatedly issuing nearly identical queries.

6. Do not assume repositories with more stars are automatically better.

7. Make at most 3 GitHub search tool calls.

8. Try to collect 6 to 10 distinct and plausible candidate repositories.

9. Each search should use a meaningfully different search strategy.

10. Stop searching early if you already have enough useful candidates.

11. Do not perform deep repository evaluation.
    Another module will inspect README and repository metadata later.

12. When candidate discovery is sufficient,
    stop calling tools and provide a concise summary.
"""


CANDIDATE_FORMATTER_SYSTEM_PROMPT = """
You are the candidate result formatter of RepoScout.

Your job is to convert the search process of the Search Agent
into one valid JSON object.

Only use repositories that actually appeared in the GitHub
search tool results.

Do not invent repository names.

Remove duplicate repositories.

Select approximately 6 to 10 candidates that are worth
deeper inspection.

Return one valid JSON object directly.

Follow these JSON schema instructions exactly:

{format_instructions}

Important:

- This is NOT the final recommendation stage.
- Do not deeply evaluate repositories.
- The reason only explains why the repository deserves further analysis.
- Return valid JSON only.
- Do not include Markdown code fences.
"""


llm = get_llm()
candidate_search_agent = create_agent(
    model=llm, 
    tools=[
        search_github_repositories, 
    ], 
    system_prompt=SEARCH_AGENT_SYSTEM_PROMPT, 

    # limit the number of tool calls to 3
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="search_github_repositories",
            run_limit=3,
            exit_behavior="continue",
        )
    ],
    # # return a candidate list, create a structured output tools.
    # response_format=ToolStrategy(
    #     CandidateSearchResult, 
    # ), 
)


formatter_llm = get_llm()

structured_formatter_llm = formatter_llm.with_structured_output(
    CandidateSearchResult,
    method="json_mode",
)

formatter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            CANDIDATE_FORMATTER_SYSTEM_PROMPT,
        ), 
        (
            "human", 
            """
            User requirements:
            {requirements}

            Search Agent trajectory:
            {search_trace}
            """,
        )
    ]
).partial(
    format_instructions=(
        candidate_format_parser.get_format_instructions()
    )
)

# need user requirements and search history.
candidate_formatter = formatter_prompt | structured_formatter_llm

def build_candidate_trace(message) -> str:
    """
    Extract GitHub search queries and tool results
    from the Search Agent trajectory.
    """

    trace = []

    # message history
    for msg in message:

        # if is AI message
        if isinstance(msg, AIMessage): 

            # may call multi-tool once
            for tool_call in msg.tool_calls: 

                # if is search tool call: search GitHub Repo about ...
                if tool_call["name"] == search_github_repositories.name: 

                    # get the query
                    query = tool_call["args"].get("query", "")
                    trace.append(
                        f"SEARCH QUERY: \n{query}"
                    )

        elif isinstance(msg, ToolMessage): 

            # return the search result: 
            if msg.name == search_github_repositories.name: 

                # get the result
                result = msg.content
                trace.append(
                    f"SEARCH RESULT: \n{result}"
                )

    return "\n\n".join(trace)



def search_candidate_repositories(
        requirements: UserRequirements, 
) -> CandidateSearchResult:
    """
    wrap the candidate search agent with a standard function interface.
    """

    agent_result = candidate_search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user", 
                    "content": (
                        "Find candidate GitHub repositories "
                        "for the following requirements:\n\n"
                        + requirements.model_dump_json(
                            indent=2
                        )
                    )
                }, 
            ]
        }
    )

    search_trace = build_candidate_trace(
        agent_result["messages"]
    )

    candidates = candidate_formatter.invoke(
        {
            "requirements": (
                requirements.model_dump_json(
                    indent=2
                )
            ),
            "search_trace": search_trace,
        }
    )

    return candidates


if __name__ == "__main__": 

    from reposcouts.stages.requirement_parser import parse_user_requirements
    user_request = (
        "I am a beginner looking for an AI Agent project. "
        "I prefer an overall architecture project with an "
        "easy-to-understand codebase. "
        "It would be better if the repository could be "
        "learned and reproduced within one day. "
        "The project must have a specific real purpose "
        "instead of being only a demonstration."
    )

    requirements = parse_user_requirements(user_request)
    print("\n========== Requirements ==========")
    print(
        requirements.model_dump_json(
            indent=2
        )
    )

    # search_candidate_repositories(requirements)
    candidates = search_candidate_repositories(requirements)
    print("\n========== Candidate Repositories ==========")
    print(
        candidates.model_dump_json(
            indent=2
        )
    )

