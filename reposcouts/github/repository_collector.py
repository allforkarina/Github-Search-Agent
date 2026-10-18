from reposcouts.core.schemas import (
    CandidateRepository,
    RepositoryEvidence,
)

from reposcouts.github.tools import (
    get_repository_info,
    get_repository_readme,
    get_repository_structure,
)

def collect_repository_evidence(
    candidate: CandidateRepository,
) -> RepositoryEvidence:
    """
    Collect factual evidence about one candidate repository.
    
    Args:
        candidate (CandidateRepository): contains the full_name of the repo.
    """

    full_name = candidate.full_name

    repo_info = get_repository_info.invoke(
        {
            "full_name": full_name,
        }
    )

    readme = get_repository_readme.invoke(
        {
            "full_name": full_name,
        }
    )

    structure = get_repository_structure.invoke(
        {
            "full_name": full_name,
            "branch": repo_info["default_branch"],
            "max_entries": 120,
        }
    )
    
    return RepositoryEvidence(
        full_name=full_name,
        metadata=repo_info,
        readme=readme,
        structure=structure,
    )

