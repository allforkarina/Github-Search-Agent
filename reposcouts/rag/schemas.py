from pathlib import PurePosixPath
from typing import Self, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict, PrivateAttr,
)

class EvidenceChunk(BaseModel):
    """
    Evidence chunk from a repo files with its source path.
    """
    chunk_id: str = Field(min_length=1)     # unique chunk id for each chunk block.

    full_name: str = Field(min_length=3)    # full name of the repo.
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$") # commit SHA(40 hex characters)

    path: str = Field(min_length=1)         # path of the file in the repo.
    symbol: str | None = None               # if the chunk is related to a symbol, the symbol name.

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)             # the chunk start and end line.

    text: str = Field(min_length=1)         # the chunk text content.

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """
        Validate that the path is a relative repository path.
        """
        path = PurePosixPath(value)

        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
        ):
            # path should be a relative path format.
            raise ValueError(
                "path must be a relative repository path"
            )

        return value


    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line < self.start_line:
            # start line should be less than or equal to end line.
            raise ValueError(
                "end_line must be greater than or equal to start_line"
            )

        # evidence chunk text lines.
        expected_lines = self.end_line - self.start_line + 1
        if len(self.text.splitlines()) != expected_lines:
            raise ValueError(
                f"Number of lines in text ({len(self.text.splitlines())}) "
                f"does not match the expected range ({expected_lines})"
            )

        return self


class RepositoryFile(BaseModel):
    """
    Collected file from a repo with its content and content's blob SHA.
    """
    path: str # relative path of the file in the repository
    blob_sha: str = Field(pattern=r"^[0-9a-f]{40}$") # Git ID of the file content
    content: str # content of the file
    redacted: bool = False


class IngestionReport(BaseModel):
    """
    Github repository ingestion report for RAG.(Repo source collection)
    """
    # status of repo collection in specified range.
    status: Literal["ok", "partial", "failed"] = "failed"

    tree_truncated: bool = False    # whether the repository tree is truncated due to size limits.
    discovered_files: int = 0       # total collected files count.
    eligible_files: int = 0         # eligible files count.(files that meet the criteria for ingestion)
    selected_files: int = 0         # files that are selected for ingestion.
    loaded_files: int = 0           # files that are successfully loaded into the RAG system.
    fetched_bytes: int = 0          # total bytes fetched from the repository.

    skipped_counts: dict[str, int] = Field(default_factory=dict)    # skipped files count by reason.
    failed_paths: list[str] = Field(default_factory=list)           # failed files paths that could not be processed.
    warnings: list[str] = Field(default_factory=list)
    expansion_rounds: int = 0
    expansion_reasons: list[str] = Field(default_factory=list)
    attempted_paths: list[str] = Field(default_factory=list)
    reserved_bytes: int = 0



class RepositorySnapshot(BaseModel):
    """
    Collected repo(certain commit snapshot) with its files.
    """
    _deadline: float | None = PrivateAttr(default=None)
    # repo full name, e.g., "owner/repo"
    full_name: str
    # commit SHA of the repo snapshot.
    commit_sha: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )

    # file ingestion, path and content with specific blob_sha
    files: list[RepositoryFile] = Field(default_factory=list)
    report: IngestionReport = Field(default_factory=IngestionReport)
    # Immutable tree metadata only, never file bodies. Used for bounded expansion.
    available_files: list[dict] = Field(default_factory=list, exclude=True)


class ChunkingResult(BaseModel):
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    fallback_files: list[str] = Field(default_factory=list)
    oversized_lines: dict[str, list[int]] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk: EvidenceChunk
    rrf_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    """
    Retrieval result for one requirement.
    """
    query: str
    mode: Literal["hybrid", "bm25", "unavailable"]
    hits: list[RetrievalHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reranked: bool = False


class RequirementTask(BaseModel):
    """
    Retrieval task for a specific requirement.
    """
    task_id: str
    requirement: str
    query: str
    kind: Literal["must_have", "technology", "language", "exclusion", "context"]
    parent_requirement: str
    queries: list[str] = Field(default_factory=list)


class RetrievalPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["must_have", "technology", "language", "exclusion", "context"]
    parent_requirement: str = Field(min_length=1, max_length=1000)
    source_quote: str = Field(min_length=1, max_length=2000)
    atomic_requirement: str = Field(min_length=1, max_length=1000)
    queries: list[str] = Field(min_length=1, max_length=3)

    @field_validator("queries")
    @classmethod
    def bounded_queries(cls, values):
        if any(not value.strip() or len(value) > 512 for value in values):
            raise ValueError("invalid retrieval expression")
        return list(dict.fromkeys(values))


class EvidenceCitation(BaseModel):
    evidence_id: str
    full_name: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    symbol: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    url: str


class EvidenceSnippet(BaseModel):
    """
    Evidence snippet and could be traced back to its source file and lines.
    """
    citation: EvidenceCitation
    text: str
    redacted: bool = False
    source_kind: Literal["source", "example", "test", "document", "config"] = "source"


class RequirementEvidence(BaseModel):
    task: RequirementTask
    mode: Literal["hybrid", "bm25", "unavailable"]
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RepositoryResearch(BaseModel):
    rag_version: str = "repository-research-v2"
    full_name: str
    commit_sha: str | None = None
    ingestion: IngestionReport = Field(default_factory=IngestionReport)
    requirements: list[RequirementEvidence] = Field(default_factory=list)
    snippets: list[EvidenceSnippet] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_revision: str | None = None
    index_reused: bool = False
    cache_stats: dict[str, int] = Field(default_factory=dict)
    reranker_model: str | None = None
