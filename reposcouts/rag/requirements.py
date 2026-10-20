"""Deterministic retrieval tasks; preserve each original requirement for judgment."""
import hashlib
import re

from reposcouts.core.schemas import UserRequirements
from .schemas import RequirementTask
from .tracing import run_stage
from .planning import task_identity


def _tasks(requirements: UserRequirements) -> list[RequirementTask]:
    """
    Decompose user requirements into retrieval tasks.
    """
    
    result = []
    # 按照 UserRequirements 对需求进行任务分组
    groups = (
        ("must_have", requirements.must_have),
        ("technology", requirements.technology_stack),
        ("language", requirements.language),
        ("exclusion", requirements.exclusions),
        ("context", [requirements.topic, requirements.learning_goal]),
    )
    for kind, values in groups:
        # 去除重复需求
        for original in dict.fromkeys(values):
            # 去除首尾空白
            if not original.strip():
                continue
            
            # 对需求进行正则化分解，例如 "A + B" 或 "A and B" 或 "A、B" 都会被分解为 ["A", "B"]
            pieces = [
                part.strip() 
                for part in re.split(
                    r"\s+\+\s+|(?<!\+)\+(?!\+ )(?=[A-Za-z])|、|\s+and\s+", # 正则化规则
                    original,   # 原始需求字符串
                    flags=re.I  # 忽略大小写
                )
            ]
            pieces = [part for part in pieces if part] or [original]
            
            # Do not turn 'A or B' into an AND task or split a negated conjunction.
            if re.search(
                r"\bor\b|或者|或|\b(?:not|without|no)\b|不|无", 
                original, 
                re.I
            ):
                pieces = [original] # 保持 OR 关系
                
            for piece in dict.fromkeys(pieces):
                identity = f"{kind}\0{original}\0{piece}"
                task_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
                # 将一个完整的需求，拆分为多个检索任务
                result.append(RequirementTask(
                    task_id=task_id, requirement=piece, parent_requirement=original,
                    kind=kind, query=piece,
                ))
    if len(result) > 32:
        # Preserve whole parent conditions instead of dropping hard requirements.
        result = [RequirementTask(task_id=task_identity(kind, parent, parent),
                  requirement=parent, parent_requirement=parent, kind=kind, query=parent)
                  for kind, values in groups for parent in dict.fromkeys(values) if parent.strip()]
        if len(result) > 32:
            raise ValueError("requirement_task_budget")
    if requirements.retrieval_plan:
        grouped = {}
        for item in requirements.retrieval_plan:
            grouped.setdefault((item.kind, item.parent_requirement), []).append(item)
        planned = []
        for kind, values in groups:
            for original in dict.fromkeys(values):
                items = grouped.get((kind, original))
                if not items:
                    planned.extend(task for task in result if task.kind == kind and task.parent_requirement == original)
                    continue
                for item in items:
                    planned.append(RequirementTask(
                        task_id=task_identity(kind, original, item.atomic_requirement),
                        requirement=item.atomic_requirement, parent_requirement=original, kind=kind,
                        query=original,
                        queries=list(dict.fromkeys([original, *item.queries]))[:4],
                    ))
        planned = list({task.task_id: task for task in planned}.values())
        return planned if len(planned) <= 32 else result
    return result


def decompose_requirements(requirements: UserRequirements) -> list[RequirementTask]:
    # tracing _tasks() 函数的调用
    return run_stage(
        "rag.requirement_tasks", {"must_have_count": len(requirements.must_have)},
        lambda: _tasks(requirements),
        lambda tasks: {"task_count": len(tasks), "task_ids": [task.task_id for task in tasks]},
    )
