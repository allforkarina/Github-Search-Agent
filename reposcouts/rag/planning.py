"""Validate LLM proposals against source text before they become retrieval tasks.

Exact quotes prove provenance, not semantic entailment. The unchanged parent
requirement remains the judgment target; rewrites are never new constraints.
"""
import hashlib
import re

from pydantic import TypeAdapter, ValidationError

from .schemas import RetrievalPlanItem


class RequirementValidationError(ValueError):
    """Controlled failure: the request needs clarification, not invented constraints."""


def requirement_groups(requirements):
    return {
        "must_have": requirements.must_have,
        "technology": requirements.technology_stack,
        "language": requirements.language,
        "exclusion": requirements.exclusions,
        "context": [requirements.topic, requirements.learning_goal],
    }


def protected_parent(parent):
    return bool(re.search(r"\bor\b|或者|或|\b(?:not|without|no)\b|不|无|\d+\.\d+", parent, re.I))


def validate_parsed_payload(payload, user_input):
    from reposcouts.core.schemas import UserRequirements

    if not isinstance(payload, dict):
        raise RequirementValidationError("requirements_not_an_object")
    raw_plan = payload.get("retrieval_plan", [])
    # Even an invalid plan must not bypass strict validation of the base contract.
    try:
        core = UserRequirements.model_validate({**payload, "retrieval_plan": [], "plan_warnings": []})
    except ValidationError:
        raise RequirementValidationError("requirements_schema_invalid") from None
    groups = requirement_groups(core)
    if any(len(values) > 32 or any(len(v) > 1000 for v in values) for values in groups.values()):
        raise RequirementValidationError("requirements_budget")
    known = {(kind, parent) for kind, values in groups.items() for parent in values if parent.strip()}
    if len(known) > 32:
        raise RequirementValidationError("requirements_budget")
    anchored = set()
    if isinstance(raw_plan, list) and len(raw_plan) <= 32:
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("kind"), str) or not isinstance(item.get("parent_requirement"), str):
                continue
            pair = (item["kind"], item["parent_requirement"])
            quote = item.get("source_quote")
            if pair in known and isinstance(quote, str) and quote.strip() and quote in user_input:
                anchored.add(pair)
    required = {pair for pair in known if pair[0] != "context"}
    if not required <= anchored:
        raise RequirementValidationError("unanchored_requirement_needs_clarification")
    try:
        plan = TypeAdapter(list[RetrievalPlanItem]).validate_python(raw_plan, strict=True)
        if not plan or len(plan) > 32:
            raise ValueError("empty_or_excessive_plan")
        for item in plan:
            pair = (item.kind, item.parent_requirement)
            if pair not in known or not item.source_quote.strip() or item.source_quote not in user_input:
                raise ValueError("foreign_or_unanchored_task")
            # Conjunction decomposition must not silently turn OR/negation/version into AND.
            if protected_parent(item.parent_requirement) and item.atomic_requirement != item.parent_requirement:
                raise ValueError("protected_constraint_changed")
        if not required <= {(item.kind, item.parent_requirement) for item in plan}:
            raise ValueError("missing_requirement")
    except (ValidationError, ValueError, TypeError):
        core.plan_warnings = ["retrieval_plan_invalid_rule_fallback"]
        return core
    core.retrieval_plan = plan
    return core


def task_identity(kind, parent, atomic):
    return hashlib.sha256(f"{kind}\0{parent}\0{atomic}".encode()).hexdigest()[:16]


def functional_requirement(text):
    """Conservative default: functionality needs implementation + integration.

    Explicit documentary/legal/language constraints use the appropriate artifacts.
    Ambiguous constraints retain the stricter implementation requirement.
    """
    return not bool(re.search(
        r"readme|documentation|license|licence|文档|许可证|开源协议|编程语言|written in|安装说明|"
        r"^(?:Python|Java|TypeScript|JavaScript|C\+\+|Rust|Go)(?:\s*[\d.]+)?$",
        text, re.I,
    ))


def example_request(requirements):
    text = " ".join([requirements.topic, requirements.learning_goal, *requirements.must_have])
    return bool(re.search(r"tutorial|example|教程|示例", text, re.I))
