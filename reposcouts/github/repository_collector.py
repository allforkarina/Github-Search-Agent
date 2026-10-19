from reposcouts.core.schemas import CandidateRepository, RepositoryEvidence, UserRequirements
from reposcouts.rag.service import research_scope

def collect_repository_evidence(
    candidate: CandidateRepository,
    requirements: UserRequirements,
) -> RepositoryEvidence:
    """
    Collect factual evidence about one candidate repository.
    
    Args:
        candidate (CandidateRepository): contains the full_name of the repo.
    """

    with research_scope() as session:
        research = session.research(candidate.full_name, requirements)
        metadata = session.metadata(candidate.full_name)
        if not metadata:
            research.warnings.append("metadata_unavailable")
        snapshot = session.snapshot(candidate.full_name)
        return RepositoryEvidence(
            full_name=candidate.full_name, metadata=metadata,
            # README is retrieved and bounded in research.snippets, just like code.
            readme="",
            structure={
                "commit_sha": snapshot.commit_sha, "selection_only": True,
                "entries": [{"path": file.path, "type": "blob"} for file in snapshot.files],
                "github_truncated": snapshot.report.tree_truncated,
            },
            research=research,
        )

