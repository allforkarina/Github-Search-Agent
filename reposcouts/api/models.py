from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints


class RunCreateRequest(BaseModel):
    user_request: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=4000),
    ]


class RunCreateResponse(BaseModel):
    run_id: str


class RunEvent(BaseModel):
    type: Literal[
        "run.started",
        "node.started",
        "node.completed",
        "retry.started",
        "run.completed",
        "run.cancelled",
        "run.failed",
    ]
    run_id: str
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RunCancelResponse(BaseModel):
    run_id: str
    status: Literal["cancelling", "cancelled", "completed", "failed"]
