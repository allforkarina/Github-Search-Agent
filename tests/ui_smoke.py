import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def sse_event(event_type: str, node=None, data=None) -> str:
    payload = {
        "type": event_type,
        "run_id": "demo-run",
        "node": node,
        "data": data or {},
    }
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


requirements = {
    "topic": "LangGraph AI agent",
    "language": ["Python"],
    "technology_stack": ["LangGraph", "LangChain"],
    "difficulty": "beginner",
    "must_have": ["Clear README", "Understandable architecture"],
    "nice_to_have": ["Can be reproduced in two hours"],
    "exclusions": [],
    "learning_goal": "Understand an end-to-end agent workflow.",
    "time_budget_in_hours": 2,
}

candidates = {
    "search_queries": [
        "langgraph agent language:python",
        "langgraph tutorial agent architecture",
    ],
    "candidates": [
        {
            "full_name": "example/langgraph-starter",
            "reason": "A focused project with a documented graph workflow.",
        },
        {
            "full_name": "example/agent-blueprints",
            "reason": "Useful architecture examples and practical agents.",
        },
    ],
}

evaluation = {
    "full_name": "example/langgraph-starter",
    "purpose_summary": "A compact LangGraph project designed for learning.",
    "must_have_checks": [],
    "requirement_match": 96,
    "learning_value": 93,
    "code_clarity": 89,
    "documentation_quality": 92,
    "reproduction_feasibility": 87,
    "maintenance_health": 78,
    "strengths": ["Clear graph structure", "Focused README"],
    "weaknesses": ["Small contributor base"],
    "uncertainties": [],
    "suitable_for_next_stage": True,
}

ranked_repository = {
    "full_name": "example/langgraph-starter",
    "base_score": 91.35,
    "score_rank": 1,
    "final_rank": 1,
    "recommendation_reason": (
        "The repository is compact enough for the time budget while still "
        "showing a complete LangGraph workflow."
    ),
    "tradeoffs": ["The community is smaller than broader frameworks."],
    "learning_path": [
        "Read the architecture overview.",
        "Trace state transitions through the graph.",
        "Run and modify one tool node.",
    ],
    "ranking_override": False,
    "override_reason": None,
}

ranking = {
    "top_repositories": [ranked_repository],
    "best_repository": "example/langgraph-starter",
    "final_reason": "This repository best balances clarity, learning value, and setup time.",
}

critic = {
    "passed": True,
    "confidence": 94,
    "issues": [],
    "retry_target": "none",
    "feedback": "The recommendation is consistent with the collected evidence.",
}


events = "".join(
    [
        sse_event("run.started", data={"user_request": "demo"}),
        sse_event("node.started", "parse_requirements"),
        sse_event(
            "node.completed",
            "parse_requirements",
            {"output": {"requirements": requirements}},
        ),
        sse_event("node.started", "search_candidates"),
        sse_event(
            "node.completed",
            "search_candidates",
            {"output": {"candidates": candidates}},
        ),
        sse_event("node.started", "evaluate_repositories"),
        sse_event(
            "node.completed",
            "evaluate_repositories",
            {"output": {"evaluations": [evaluation]}},
        ),
        sse_event("node.started", "rank_repositories"),
        sse_event(
            "node.completed",
            "rank_repositories",
            {"output": {"ranking": ranking}},
        ),
        sse_event("node.started", "critic"),
        sse_event(
            "node.completed",
            "critic",
            {"output": {"critic_result": critic, "critic_feedback": critic["feedback"]}},
        ),
        sse_event(
            "run.completed",
            data={
                "requirements": requirements,
                "candidates": candidates,
                "evaluations": [evaluation],
                "ranking": ranking,
                "critic_result": critic,
                "retry_count": 0,
            },
        ),
    ]
)


def fulfill_create_run(route: Route) -> None:
    request = route.request.post_data_json
    assert "LangGraph" in request["user_request"]
    route.fulfill(
        status=202,
        content_type="application/json",
        body=json.dumps({"run_id": "demo-run"}),
    )


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1050})
    console_errors = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )

    page.goto("http://127.0.0.1:8000")
    page.wait_for_load_state("networkidle")
    assert page.get_by_role("heading", name="Find the repository worth your time.").is_visible()
    assert page.get_by_role("textbox", name="你希望找到什么样的项目？").is_visible()
    light_background = page.locator(".app-surface").evaluate(
        "element => getComputedStyle(element).backgroundColor"
    )
    page.screenshot(path=ARTIFACTS / "initial-light.png", full_page=True)

    page.get_by_role("button", name="切换到深色模式").click()
    assert page.get_by_role("button", name="切换到浅色模式").is_visible()
    dark_background = page.locator(".app-surface").evaluate(
        "element => getComputedStyle(element).backgroundColor"
    )
    assert light_background != dark_background
    page.screenshot(path=ARTIFACTS / "initial-dark.png", full_page=True)

    page.route("**/api/runs", fulfill_create_run)
    page.route(
        "**/api/runs/demo-run/events",
        lambda route: route.fulfill(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            body=events,
        ),
    )
    page.get_by_role("textbox", name="你希望找到什么样的项目？").fill(
        "寻找适合初学者的 Python LangGraph Agent 项目"
    )
    page.get_by_role("button", name="Search repositories").click()
    page.get_by_text("搜索完成", exact=True).wait_for()
    page.get_by_role("heading", name="example/langgraph-starter").wait_for()
    page.screenshot(path=ARTIFACTS / "completed-dark.png", full_page=True)

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto("http://127.0.0.1:8000")
    mobile.wait_for_load_state("networkidle")
    assert mobile.get_by_role("textbox", name="你希望找到什么样的项目？").is_visible()
    mobile.screenshot(path=ARTIFACTS / "initial-mobile.png", full_page=True)

    assert not console_errors, console_errors
    browser.close()

print("UI smoke test passed")
