"""Fail closed on missing, foreign or invented requirement citations."""
from collections import Counter
import re

from reposcouts.core.schemas import RepositoryEvaluation, RequirementCheck
from .security import sanitize_data
from .tracing import run_stage


def research_summary(evidence):
    pack = evidence.research
    if pack is None:
        return {"status": "unavailable", "warnings": ["missing_research"]}
    return {
        "commit_sha": pack.commit_sha, "ingestion_status": pack.ingestion.status,
        "selected_files": pack.ingestion.selected_files, "loaded_files": pack.ingestion.loaded_files,
        "snippet_count": len(pack.snippets), "index_reused": pack.index_reused,
        "modes": sorted({item.mode for item in pack.requirements}),
        "embedding_model": pack.embedding_model, "embedding_revision": pack.embedding_revision,
        "warnings": sorted(set(pack.warnings + [warning for item in pack.requirements for warning in item.warnings])),
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
        covers_tasks = all(set(task.evidence_ids) & set(ids) for task in tasks)
        if not valid or not covers_tasks:
            check.status = "uncertain"
            check.evidence = "引用不足或无效，当前证据无法核实该要求。"
            check.evidence_ids = []
            check.citations = []
            result.uncertainties.append("citation_validation_failed: " + requirement)
        else:
            check.evidence_ids = ids
            check.citations = [known[evidence_id].model_copy(deep=True) for evidence_id in ids]
            sources = " ".join(f"{item.path}:L{item.start_line}-L{item.end_line} {item.url}" for item in check.citations)
            # All source links in the final check come from the validated source catalog.
            explanation = re.sub(r"https?://[^\s)\]]+", "", check.evidence)
            check.evidence = explanation + "\n" + sources
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


def ground_evaluation(requirements, evidence, evaluation):
    return run_stage(
        "rag.judgment", {"repository": evidence.full_name},
        lambda: _ground(requirements, evidence, evaluation),
        lambda result: {"requirement_match": result.requirement_match, "suitable": result.suitable_for_next_stage,
                        "checks": [{"status": check.status, "evidence_ids": check.evidence_ids} for check in result.must_have_checks]},
    )
