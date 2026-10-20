"""Fail closed on missing, foreign or invented requirement citations."""
from collections import Counter
import re

from reposcouts.core.schemas import RepositoryEvaluation, RequirementCheck
from .security import sanitize_data
from .tracing import run_stage
from .planning import functional_requirement, example_request
from .context import source_kind
import ast
import textwrap


def research_summary(evidence):
    pack = evidence.research
    if pack is None:
        return {"status": "unavailable", "warnings": ["missing_research"]}
    return {
        "rag_version": pack.rag_version,
        "commit_sha": pack.commit_sha, "ingestion_status": pack.ingestion.status,
        "selected_files": pack.ingestion.selected_files, "loaded_files": pack.ingestion.loaded_files,
        "snippet_count": len(pack.snippets), "index_reused": pack.index_reused,
        "modes": sorted({item.mode for item in pack.requirements}),
        "embedding_model": pack.embedding_model, "embedding_revision": pack.embedding_revision,
        "warnings": sorted(set(pack.warnings + [warning for item in pack.requirements for warning in item.warnings])),
        "cache_stats": pack.cache_stats, "reranker_model": pack.reranker_model,
        "expansion_rounds": pack.ingestion.expansion_rounds,
        "expansion_reasons": pack.ingestion.expansion_reasons,
    }


def uncertain_evaluation(requirements, evidence, reason="insufficient_evidence"):
    return RepositoryEvaluation(
        full_name=evidence.full_name,
        purpose_summary="Available source evidence is insufficient for a supported recommendation.",
        must_have_checks=[RequirementCheck(requirement=value, status="uncertain", evidence=reason)
                          for value in dict.fromkeys(requirements.must_have)],
        requirement_match=0, learning_value=0, code_clarity=0, documentation_quality=0,
        reproduction_feasibility=0, maintenance_health=0, strengths=[], weaknesses=[],
        uncertainties=[reason], suitable_for_next_stage=False, research_summary=research_summary(evidence),
    )


def _ground(requirements, evidence, evaluation):
    pack = evidence.research
    if pack is None or not pack.snippets:
        return uncertain_evaluation(requirements, evidence)
    result = RepositoryEvaluation.model_validate(sanitize_data(evaluation.model_dump()))
    result.full_name = evidence.full_name
    result.research_summary = research_summary(evidence)
    known = {
        snippet.citation.evidence_id: snippet.citation for snippet in pack.snippets
        if snippet.citation.full_name == evidence.full_name and snippet.citation.commit_sha == pack.commit_sha
    }
    snippets = {snippet.citation.evidence_id: snippet for snippet in pack.snippets}
    counts = Counter(check.requirement for check in result.must_have_checks)
    by_requirement = {check.requirement: check for check in result.must_have_checks}
    checks = []
    for requirement in dict.fromkeys(requirements.must_have):
        check = by_requirement.get(requirement)
        if check is None or counts[requirement] != 1:
            checks.append(RequirementCheck(requirement=requirement, status="uncertain", evidence="missing_or_duplicate_judgment"))
            continue
        tasks = [item for item in pack.requirements
                 if item.task.kind == "must_have" and item.task.parent_requirement == requirement]
        allowed = {evidence_id for task in tasks for evidence_id in task.evidence_ids}
        ids = list(dict.fromkeys(check.evidence_ids))
        valid = bool(tasks and ids) and all(evidence_id in known and evidence_id in allowed for evidence_id in ids)
        covers_tasks = check.status != "satisfied" or all(set(task.evidence_ids) & set(ids) for task in tasks)
        if not valid or not covers_tasks:
            check.status = "uncertain"
            check.evidence = "引用不足或无效，当前证据无法核实该要求。"
            check.evidence_ids = []
            check.citations = []
            check.implementation_evidence_ids = []
            check.integration_evidence_ids = []
            check.contradiction_evidence_ids = []
            check.implementation_level = "unknown"
            result.uncertainties.append("citation_validation_failed: " + requirement)
        else:
            check.evidence_ids = ids
            check.citations = [known[evidence_id].model_copy(deep=True) for evidence_id in ids]
            sources = " ".join(f"{item.path}:L{item.start_line}-L{item.end_line} {item.url}" for item in check.citations)
            # All source links in the final check come from the validated source catalog.
            explanation = re.sub(r"https?://[^\s)\]]+", "", check.evidence)
            check.evidence = explanation + "\n" + sources
            impl = list(dict.fromkeys(check.implementation_evidence_ids))
            integrated = list(dict.fromkeys(check.integration_evidence_ids))
            contradiction = list(dict.fromkeys(check.contradiction_evidence_ids))
            permitted = {"source", "example"} if example_request(requirements) else {"source"}
            valid_impl = [key for key in impl if key in ids and source_kind(known[key].path) in permitted
                          and _has_implementation(snippets[key])]
            symbols = _implementation_symbols([snippets[key] for key in valid_impl])
            valid_integration = [key for key in integrated if key in ids
                                 and source_kind(known[key].path) in permitted | {"config"}
                                 and _has_integration(snippets[key], symbols)]
            check.implementation_evidence_ids = valid_impl
            check.integration_evidence_ids = valid_integration
            check.contradiction_evidence_ids = [key for key in contradiction if key in ids]
            check.implementation_level = (
                "integrated" if valid_impl and valid_integration else
                "implemented" if valid_impl else
                "dependency" if any(re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+|<dependency>|dependencies\s*[:=])",
                                              snippets[key].text) for key in ids) else "declared"
            )
            if check.status == "satisfied" and functional_requirement(requirement):
                # Every independent subtask needs both roles, not a code citation
                # from one task used to mask a missing implementation in another.
                supported = True
                for task in tasks:
                    implementations = set(task.evidence_ids) & set(valid_impl)
                    names = _implementation_symbols([snippets[key] for key in implementations])
                    wired = any(_has_integration(snippets[key], names)
                                for key in set(task.evidence_ids) & set(valid_integration))
                    supported = supported and bool(implementations) and wired
                if not supported:
                    check.status = "uncertain"
                    check.evidence += "\n尚未同时核实该功能的实现与调用/注册/配置接入。"
                    result.uncertainties.append("implementation_or_integration_missing: " + requirement)
            if check.status == "not_satisfied" and not check.contradiction_evidence_ids:
                check.status = "uncertain"
                result.uncertainties.append("explicit_contradiction_missing: " + requirement)
        checks.append(check)
    result.must_have_checks = checks
    satisfied = sum(check.status == "satisfied" for check in checks)
    result.requirement_match = round(100 * satisfied / len(checks)) if checks else 100
    result.suitable_for_next_stage = bool(
        result.suitable_for_next_stage and pack.snippets
        and all(check.status == "satisfied" for check in checks)
    )
    result.uncertainties = list(dict.fromkeys(result.uncertainties + pack.warnings))
    return result


def _has_implementation(snippet):
    if not re.sub(r"[\s{}();,:]+", "", snippet.text):
        return False
    if snippet.citation.symbol and snippet.text.strip():
        # Symbol boundaries come from AST; the evaluator still checks actual semantics.
        return True
    return bool(re.search(r"\b(?:def|class|function|interface|record)\s+\w+", snippet.text))


def _implementation_symbols(snippets):
    symbols = set()
    for snippet in snippets:
        if snippet.citation.symbol:
            symbols.add(snippet.citation.symbol.rsplit(".", 1)[-1])
        symbols.update(re.findall(r"\b(?:def|class|function|interface|record)\s+(\w+)", snippet.text))
    return symbols


def _has_integration(snippet, symbols):
    if not symbols:
        return False
    if source_kind(snippet.citation.path) == "config":
        return bool(re.search(r"[:=]", snippet.text)) and any(
            re.search(r"\b" + re.escape(name) + r"\b", snippet.text) for name in symbols)
    if snippet.citation.path.endswith(".py"):
        try:
            tree = ast.parse(textwrap.dedent(snippet.text))
            aliases = {alias.asname: alias.name.rsplit(".", 1)[-1]
                       for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                       for alias in node.names if alias.asname}
            registration = {"get", "post", "put", "delete", "patch", "route", "websocket", "register",
                            "add_node", "add_route", "add_tool", "bind_tools", "include_router"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                    if aliases.get(name, name) in symbols:
                        return True
                    if name in registration and any(isinstance(arg, ast.Name) and arg.id in symbols
                                                    for value in node.args for arg in ast.walk(value)):
                        return True
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
                    if any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") in registration
                           for d in node.decorator_list):
                        return True
            return False
        except (SyntaxError, ValueError, RecursionError):
            pass
    # Conservative lexical fallback for partial/non-Python chunks: discard
    # obvious declarations/imports; this is a structural gate, not entailment.
    lines = [line for line in snippet.text.splitlines()
             if not re.match(r"\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|import|from)\b", line)]
    text = "\n".join(lines)
    return any(re.search(r"(?<![\w$])" + re.escape(name) + r"\s*\(", text) for name in symbols)


def ground_evaluation(requirements, evidence, evaluation):
    return run_stage(
        "rag.judgment", {"repository": evidence.full_name},
        lambda: _ground(requirements, evidence, evaluation),
        lambda result: {"requirement_match": result.requirement_match, "suitable": result.suitable_for_next_stage,
                        "checks": [{"status": check.status, "evidence_ids": check.evidence_ids} for check in result.must_have_checks]},
    )
