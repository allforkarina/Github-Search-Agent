from pathlib import PurePosixPath
from typing import Self, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

class EvidenceChunk(BaseModel):
    """
    Evidence chunk for RAG (Retrieval-Augmented Generation) model.
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
        path = PurePosixPath(value) #

        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
        ):
            raise ValueError(
                "path must be a relative repository path"
            )

        return value


    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError(
                "end_line must be greater than or equal to start_line"
            )

        expected_lines = self.end_line - self.start_line + 1
        if len(self.text.splitlines()) != expected_lines:
            raise ValueError(
                f"Number of lines in text ({len(self.text.splitlines())}) "
                f"does not match the expected range ({expected_lines})"
            )

        return self


class RepositoryFile(BaseModel):
    path: str # relative path of the file in the repository
    blob_sha: str = Field(pattern=r"^[0-9a-f]{40}$") # Git ID of the file content
    content: str # content of the file


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


class RepositorySnapshot(BaseModel):
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


class RetrievalResult(BaseModel):
    query: str
    mode: Literal["hybrid", "bm25", "unavailable"]
    hits: list[RetrievalHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RequirementTask(BaseModel):
    task_id: str
    requirement: str
    query: str
    kind: Literal["must_have", "technology", "language", "exclusion", "context"]
    parent_requirement: str


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
    citation: EvidenceCitation
    text: str
    redacted: bool = False


class RequirementEvidence(BaseModel):
    task: RequirementTask
    mode: Literal["hybrid", "bm25", "unavailable"]
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RepositoryResearch(BaseModel):
    full_name: str
    commit_sha: str | None = None
    ingestion: IngestionReport = Field(default_factory=IngestionReport)
    requirements: list[RequirementEvidence] = Field(default_factory=list)
    snippets: list[EvidenceSnippet] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_revision: str | None = None
    index_reused: bool = False
