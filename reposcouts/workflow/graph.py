from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from reposcouts.workflow.state import RepoScoutState

from reposcouts.workflow.nodes import (
    parse_requirements_node,
    search_candidates_node,
    evaluate_repositories_node,
    rank_repositories_node,
    critic_node,
    increment_retry_node,
    route_after_critic,
    route_retry_target,
)

# build a Graph
builder = StateGraph(
    RepoScoutState, 
)

# add nodes to the graph
builder.add_node(
    "parse_requirements",
    parse_requirements_node,
)

builder.add_node(
    "search_candidates",
    search_candidates_node,
)

builder.add_node(
    "evaluate_repositories",
    evaluate_repositories_node,
)

builder.add_node(
    "rank_repositories",
    rank_repositories_node,
)

builder.add_node(
    "critic",
    critic_node,
)

builder.add_node(
    "increment_retry",
    increment_retry_node,
)

# add edges to connect the nodes, currently in a serial way.
builder.add_edge(
    START,
    "parse_requirements",
)

builder.add_edge(
    "parse_requirements",
    "search_candidates",
)

builder.add_edge(
    "search_candidates",
    "evaluate_repositories",
)

builder.add_edge(
    "evaluate_repositories",
    "rank_repositories",
)

builder.add_edge(
    "rank_repositories",
    "critic",
)


# add conditional edges to loop back
builder.add_conditional_edges(
    "critic",

    # two branches:
    # - critic 'pass' -> END
    #   or retry limit reached -> END
    # - critic 'fail' -> retry: increment retry
    route_after_critic,
    {
        "end": END,
        "retry": "increment_retry",
    },
)

builder.add_conditional_edges(
    "increment_retry",

    # two branches:
    # - increment_retry 'retry' -> rank_repositories
    # - increment_retry 'retry' -> evaluate_repositories
    route_retry_target,
    {
        "ranker": "rank_repositories",
        "evaluator": "evaluate_repositories",
    },
)

reposcout_graph = builder.compile()

