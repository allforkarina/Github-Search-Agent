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
    name: str, 
    inputs: dict, 
    operation: Callable[[], T],
    summarize: Callable[[T], dict], 
    *, 
    tool: bool = False,
) -> T:
    """
    Args:
        name: 阶段的名称，用于在跟踪中标识阶段
        inputs: 该阶段输入的摘要，trace不传入完整输入
        operation: 当前阶段需要执行的真实操作
        summarize: 执行操作结果的安全摘要
        tool: Whether to wrap the operation as a LangChain tool.
    """
    # Only summaries go through invoke(); raw documents/vectors stay in this closure.
    result: list[T] = []

    def invoke(_):
        try:
            value = operation()         # 执行当前阶段的真实操作
            summary = summarize(value)  # 获取操作结果的安全摘要
        except StageError:
            raise
        except Exception:
            raise StageError(f"{name}_failed") from None
        
        result.append(value)    # 回传给调用者 真实的输出
        return summary          # trace 只需要结果的摘要

    # 检查当前 run 是否存在 tracing 系统
    callbacks = ensure_config().get("callbacks")
    # 如果没有 tracing 则正常执行业务操作，不创建 tracing，run_stage 不要求必须存在 tracing
    if not callbacks or (hasattr(callbacks, "handlers") and not callbacks.handlers):
        invoke(None)
        
    elif tool:
        Tool(
            name=name, 
            func=invoke, 
            description="Read repository data; summary only."
        ).invoke(
            json.dumps(inputs, ensure_ascii=False),
        )
        
    else:
        RunnableLambda(invoke, name=name).invoke(inputs)
    return result[0]
