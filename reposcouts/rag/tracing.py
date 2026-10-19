"""Attach summary-only RAG spans to the existing LangChain callback tree."""
import json
from collections.abc import Callable
from typing import TypeVar

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import Tool

T = TypeVar("T")


class StageError(RuntimeError):
    """Controlled error codes safe to put into traces."""


def run_stage(
    name: str, inputs: dict, operation: Callable[[], T],
    summarize: Callable[[T], dict], *, tool: bool = False,
) -> T:
    # Only summaries go through invoke(); raw documents/vectors stay in this closure.
    result: list[T] = []

    def invoke(_):
        try:
            value = operation()
            summary = summarize(value)
        except StageError:
            raise
        except Exception:
            raise StageError(f"{name}_failed") from None
        result.append(value)
        return summary

    callbacks = ensure_config().get("callbacks")
    if not callbacks or (hasattr(callbacks, "handlers") and not callbacks.handlers):
        # Do not create an implicit global tracer outside the request's explicit callbacks.
        invoke(None)
    elif tool:
        Tool(name=name, func=invoke, description="Read repository data; summary only.").invoke(
            json.dumps(inputs, ensure_ascii=False),
        )
    else:
        RunnableLambda(invoke, name=name).invoke(inputs)
    return result[0]
