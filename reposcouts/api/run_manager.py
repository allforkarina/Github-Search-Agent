from __future__ import annotations

import os
import logging
from contextlib import closing

from langsmith import tracing_context

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from reposcouts.api.models import RunEvent
from reposcouts.api.serialization import to_jsonable
from reposcouts.workflow.graph import reposcout_graph
from reposcouts.workflow.state import RepoScoutState
from reposcouts.observability.metadata import build_run_metadata
from reposcouts.observability.tracing import build_trace_client, build_run_tracer
from reposcouts.observability.metrics import build_run_summary
from reposcouts.rag.service import research_scope


logger = logging.getLogger(__name__)

RunStatus = Literal[
    "running",
    "cancelling",
    "cancelled",
    "completed",
    "failed",
]

TERMINAL_EVENT_TYPES = {
    "run.completed",
    "run.cancelled",
    "run.failed",
}


@dataclass
class RunRecord:
    run_id: str
    user_request: str
    status: RunStatus = "running"
    events: queue.Queue[RunEvent] = field(default_factory=queue.Queue)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    summary: dict[str, Any] | None = None # summary of the run, execution and critic.


class RunConflictError(RuntimeError):
    """Raised when a second run is requested while one is active."""


class RunNotFoundError(KeyError):
    """Raised when a run identifier is not the current run."""


class RunManager:
    """Manage the single transient RepoScouts run used by the local UI."""

    def __init__(self, graph: Any = reposcout_graph) -> None:
        self._lock = threading.Lock()
        self._current: RunRecord | None = None
        self._graph = graph

    def create_run(self, user_request: str) -> RunRecord:
        with self._lock:
            if self._current and self._current.status in {
                "running",
                "cancelling",
            }:
                raise RunConflictError("A repository search is already running.")

            record = RunRecord(
                run_id=uuid.uuid4().hex,
                user_request=user_request,
            )
            self._current = record

        # Seperate thread to execute the run asynchronously.
        worker = threading.Thread(
            target=self._execute,
            args=(record,),
            name=f"reposcout-run-{record.run_id[:8]}",
            daemon=True,
        )
        worker.start()
        return record

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            if self._current is None or self._current.run_id != run_id:
                raise RunNotFoundError(run_id)
            return self._current

    def cancel_run(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        if record.status == "running":
            record.cancel_requested.set()
            self._set_status(record, "cancelling")
        return record

    def _set_status(self, record: RunRecord, status: RunStatus) -> None:
        with self._lock:
            if self._current is record:
                record.status = status

    @staticmethod
    def _emit(
        record: RunRecord,
        event_type: str,
        *,
        node: str | None = None,
        data: dict | None = None,
    ) -> None:
        record.events.put(
            RunEvent(
                type=event_type,
                run_id=record.run_id,
                node=node,
                data=to_jsonable(data or {}),
            )
        )

    @staticmethod
    def _next_node(node: str, state: RepoScoutState) -> str | None:
        if node == "parse_requirements":
            return "search_candidates"
        if node == "search_candidates":
            return "evaluate_repositories"
        if node == "evaluate_repositories":
            return "rank_repositories"
        if node == "rank_repositories":
            return "critic"
        if node == "critic":
            critic_result = state.get("critic_result")
            if critic_result is None:
                return None
            if critic_result.passed:
                return None
            if state["retry_count"] >= state["max_retries"]:
                return None
            return "increment_retry"
        if node == "increment_retry":
            critic_result = state.get("critic_result")
            if critic_result and critic_result.retry_target == "evaluator":
                return "evaluate_repositories"
            return "rank_repositories"
        return None

    def _execute(self, record: RunRecord) -> None:
        state: RepoScoutState = {
            "user_request": record.user_request,
            "requirements": None,
            "candidates": None,
            "evaluations": [],
            "ranking": None,
            "critic_result": None,
            "retry_count": 0,
            "max_retries": 2,
            "critic_feedback": "",
        }

        client = None
        tracer = None   # LangChainTracer instance for tracing

        # Start from user request parsing node.
        active_node = "parse_requirements"
        self._emit(
            record,
            "run.started",
            data={"user_request": record.user_request},
        )
        self._emit(record, "node.started", node=active_node)

        try:
            # if REPOSCOUTS_TRACING_ENABLED is true, then enabled is True.
            enabled = (
                os.getenv(
                    "REPOSCOUTS_TRACING_ENABLED", "false"
                ).strip().lower() == "true"
            )

            # Build run metadata for the current run.
            metadata = build_run_metadata(
                run_id=record.run_id, 
                environment=os.getenv("REPOSCOUTS_ENVIRONMENT", "development"), 
                agent_version=os.getenv("REPOSCOUTS_AGENT_VERSION", "dev"), 
                model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), 
                prompt_version=os.getenv("REPOSCOUTS_PROMPT_VERSION", "v1"), 
            )

            project_name = os.getenv(
                "LANGSMITH_PROJECT",
                "reposcouts-development",
            )

            if enabled: 
                # build a trace client with secrets mask and max string length.
                client = build_trace_client(
                    secrets=tuple(
                        os.getenv(name, "")
                        for name in (
                            "GITHUB_TOKEN", 
                            "DEEPSEEK_API_KEY", 
                            "LANGSMITH_API_KEY", 
                        )
                    ), 
                    max_string_chars=4000, 
                )

                def provide_trace_summary(execution_status: str) -> dict: 
                    if record.status == "cancelled":
                        execution_status = "cancelled"

                    critic = state["critic_result"]

                    return build_run_summary(
                        execution_status=execution_status,
                        retry_count=state["retry_count"],
                        max_retries=state["max_retries"],
                        critic_passed=(
                            critic.passed if critic is not None else None
                        ), 
                        recommendation_available=(
                            bool(state["ranking"].top_repositories) if state["ranking"] is not None else None
                        ),
                    )

                # build a tracer for the current run.
                tracer = build_run_tracer(
                    client=client,
                    project_name=project_name,
                    summary_provider=provide_trace_summary,
                )

            with research_scope(), tracing_context(
                enabled=enabled, 
                client=client, 
                project_name=project_name, 
            ): 
                with closing(
                    self._graph.stream(
                        state, 
                        stream_mode="updates", 
                        config={
                            "metadata": metadata,
                            "run_name": "RepoScoutsRequest", 
                            # execution of the run is traced by this tracer.
                            "callbacks": [tracer] if tracer is not None else [], 
                        }, 
                    )
                ) as updates: 
                    # updates == _graph.stream() with metadata and closing.
                    for chunk in updates:
                        for node, update in chunk.items():
                            if update:
                                state.update(update)

                            self._emit(
                                record,
                                "node.completed",
                                node=node,
                                data={"output": update or {}},
                            )

                            # Check if cancellation of request is set.
                            if record.cancel_requested.is_set():
                                # current execution_status == "cancelled"
                                self._set_status(record, "cancelled")
                                critic = state["critic_result"]
                                # build summary and add to record
                                record.summary = build_run_summary(
                                    execution_status=record.status, # "cancelled"
                                    retry_count=state["retry_count"],
                                    max_retries=state["max_retries"],
                                    critic_passed=critic.passed if critic is not None else None, 
                                    recommendation_available=(
                                        bool(state["ranking"].top_repositories) if state["ranking"] is not None else None
                                    ),
                                )
                                self._emit(
                                    record,
                                    "run.cancelled",
                                    node=node,
                                    data={"message": "The run stopped between workflow nodes."},
                                )
                                return

                            # next node to execute based on current node and state.
                            active_node = self._next_node(node, state)
                            if active_node == "increment_retry":
                                self._emit(
                                    record,
                                    "retry.started",
                                    node=active_node,
                                    data={
                                        "attempt": state["retry_count"] + 1,
                                        "max_retries": state["max_retries"],
                                        "feedback": state.get("critic_feedback", ""),
                                    },
                                )
                            elif active_node:
                                self._emit(
                                    record,
                                    "node.started",
                                    node=active_node,
                                )

            # current execution_status == "completed"
            self._set_status(record, "completed")
            critic = state["critic_result"]
            record.summary = build_run_summary(
                execution_status=record.status,
                retry_count=state["retry_count"],
                max_retries=state["max_retries"],
                critic_passed=critic.passed if critic is not None else None,
                recommendation_available=(
                    bool(state["ranking"].top_repositories) if state["ranking"] is not None else None
                ),
            )
            self._emit(
                record,
                "run.completed",
                data={
                    "requirements": state.get("requirements"),
                    "candidates": state.get("candidates"),
                    "evaluations": state.get("evaluations", []),
                    "ranking": state.get("ranking"),
                    "critic_result": state.get("critic_result"),
                    "retry_count": state.get("retry_count", 0),
                    "recommendation_status": (
                        "available" if state.get("ranking") and state["ranking"].top_repositories
                        else "insufficient_evidence"
                    ),
                },
            )
        except Exception as exc:
            # current execution_status == "failed"
            self._set_status(record, "failed")
            critic = state["critic_result"]
            record.summary = build_run_summary(
                execution_status=record.status,
                retry_count=state["retry_count"],
                max_retries=state["max_retries"],
                critic_passed=critic.passed if critic is not None else None,
                recommendation_available=(
                    bool(state["ranking"].top_repositories) if state["ranking"] is not None else None
                ),
            )
            self._emit(
                record,
                "run.failed",
                node=active_node,
                data={
                    "error_type": type(exc).__name__,
                    "message": str(exc) or "The repository search failed.",
                },
            )

        finally: 
            if client is not None: # trace client is existing.
                try: 
                    client.close(timeout=5.0)
                except Exception as e:
                    logger.warning(
                        "Failed to close trace client"
                    )


run_manager = RunManager()
