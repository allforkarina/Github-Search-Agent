from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from reposcouts.api.models import RunEvent
from reposcouts.api.serialization import to_jsonable
from reposcouts.workflow.graph import reposcout_graph
from reposcouts.workflow.state import RepoScoutState


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
        active_node = "parse_requirements"

        self._emit(
            record,
            "run.started",
            data={"user_request": record.user_request},
        )
        self._emit(record, "node.started", node=active_node)

        try:
            for chunk in self._graph.stream(
                state,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if update:
                        state.update(update)

                    self._emit(
                        record,
                        "node.completed",
                        node=node,
                        data={"output": update or {}},
                    )

                    if record.cancel_requested.is_set():
                        self._set_status(record, "cancelled")
                        self._emit(
                            record,
                            "run.cancelled",
                            node=node,
                            data={"message": "The run stopped between workflow nodes."},
                        )
                        return

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

            self._set_status(record, "completed")
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
                },
            )
        except Exception as exc:
            self._set_status(record, "failed")
            self._emit(
                record,
                "run.failed",
                node=active_node,
                data={
                    "error_type": type(exc).__name__,
                    "message": str(exc) or "The repository search failed.",
                },
            )


run_manager = RunManager()
