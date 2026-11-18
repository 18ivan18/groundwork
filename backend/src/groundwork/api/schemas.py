"""Pydantic request and response models. These belong to the API layer only.

The examples use an invented company, because they appear in the public API docs.
"""

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from groundwork.ingestion.jobs import JobStatus

# Lowercase words joined by hyphens, such as `sec-10k`: collection ids appear in URLs and filters.
CollectionId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=40)]
DocTypeText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]

_EXAMPLE_PASSAGE: dict[str, JsonValue] = {
    "doc_id": "9f2c1e7a",
    "title": "ACME Corp · Form 10-K",
    "collection_id": "sec-10k",
    "doc_type": "10-K",
    "doc_date": "2025-09-27",
    "metadata": {"company": "ACME Corp", "ticker": "ACME", "form": "10-K"},
    "section_path": ["Item 1 — Business"],
    "page": None,
    "excerpt": "ACME Corp sells anvils to blacksmiths.",
    "chunk_id": "9f2c1e7a:4",
    "retrieval": {"dense_rank": 1, "sparse_rank": 3, "rrf_score": 0.0323, "rerank_rank": 1},
}


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "llm_provider": "openai_compatible",
                    "embedder": "BAAI/bge-small-en-v1.5",
                    "documents": 3,
                    "chunks": 438,
                }
            ]
        }
    )

    status: Literal["ok"] = Field(description="Always `ok`: a failing service does not answer.")
    llm_provider: str = Field(description="The configured LLM adapter.")
    embedder: str = Field(description="The embedding model behind the index.")
    documents: int = Field(description="Documents in the library.")
    chunks: int = Field(description="Passages in the index.")


class ConfigResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"upload_types": [".htm", ".html", ".md", ".txt"], "max_upload_mb": 20}]
        }
    )

    upload_types: list[str] = Field(
        description="File suffixes that uploads accept, lowercase and sorted."
    )
    max_upload_mb: int = Field(description="The largest file an upload accepts, in megabytes.")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"detail": "The LLM provider rejected the API key."}]}
    )

    detail: str = Field(description="A message that is safe to show to users.")


class AskFilters(BaseModel):
    """Limits the search. An empty list places no limit on that field."""

    collections: list[CollectionId] = Field(default=[], description="Keep these collections.")
    doc_types: list[DocTypeText] = Field(default=[], description="Keep these document types.")
    date_from: date | None = Field(
        default=None,
        description="Keep documents dated on or after this day. Undated ones never match.",
    )
    date_to: date | None = Field(
        default=None,
        description="Keep documents dated on or before this day. Undated ones never match.",
    )
    metadata: dict[str, str] = Field(
        default={}, description="Keep documents whose metadata has every one of these values."
    )


class AskRequest(BaseModel):
    # Stripping first means a question of only spaces fails the length check.
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "question": "Who buys ACME's anvils?",
                    "filters": {"collections": ["sec-10k"], "metadata": {"ticker": "ACME"}},
                    "top_k": 6,
                }
            ]
        },
    )

    question: str = Field(min_length=3, max_length=1000, description="The analyst's question.")
    filters: AskFilters | None = Field(default=None, description="Limits on what is searched.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Passages to retrieve. Defaults to TOP_K."
    )


class RetrievalInfo(BaseModel):
    dense_rank: int | None = Field(description="Rank in vector search. Null if it did not appear.")
    sparse_rank: int | None = Field(description="Rank in BM25 search. Null if it did not appear.")
    rrf_score: float = Field(description="The score after Reciprocal Rank Fusion.")
    rerank_rank: int | None = Field(
        default=None,
        description="Rank after cross-encoder re-ranking, which sets the order of the passages. "
        "Null if re-ranking is off.",
    )


class PassageResponse(BaseModel):
    doc_id: str = Field(description="The sha256 of the document's bytes.")
    title: str = Field(description="The document's title.")
    collection_id: str = Field(description="The collection the document belongs to.")
    doc_type: str = Field(description="The kind of document, such as `10-K` or `runbook`.")
    doc_date: date | None = Field(description="The date the document is about, if known.")
    metadata: dict[str, str] = Field(description="Fields only some collections have.")
    section_path: list[str] = Field(
        description="The headings the passage sits under, outermost first, such as an Item of "
        "a 10-K or a chapter. Empty when it is under none."
    )
    page: int | None = Field(description="The page number, for formats that have pages.")
    excerpt: str = Field(description="The full passage text, exactly as indexed.")
    chunk_id: str = Field(description="The passage's ID: the doc_id and its position.")
    retrieval: RetrievalInfo = Field(description="How the two searches ranked this passage.")


class SourceResponse(PassageResponse):
    id: int = Field(description="The number that cites this passage in the answer, as in [1].")


class Timings(BaseModel):
    retrieval: int = Field(description="Milliseconds spent searching.")
    generation: int = Field(description="Milliseconds spent in the LLM. 0 if it was not called.")


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "ACME Corp sells its anvils to blacksmiths [1].",
                    "insufficient_evidence": False,
                    "sources": [{"id": 1, **_EXAMPLE_PASSAGE}],
                    "retrieved": [_EXAMPLE_PASSAGE],
                    "timings_ms": {"retrieval": 21, "generation": 665},
                }
            ]
        }
    )

    answer: str = Field(description="Empty when insufficient_evidence is true.")
    insufficient_evidence: bool = Field(
        description="True when the documents do not support an answer, so none was written."
    )
    sources: list[SourceResponse] = Field(description="Only the passages the answer cites.")
    retrieved: list[PassageResponse] = Field(description="Every passage retrieved.")
    timings_ms: Timings = Field(description="Where the time went.")


class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "doc_id": "9f2c1e7a",
                    "title": "ACME Corp · Form 10-K",
                    "collection_id": "sec-10k",
                    "doc_type": "10-K",
                    "doc_date": "2025-09-27",
                    "metadata": {"company": "ACME Corp", "ticker": "ACME", "form": "10-K"},
                    "filename": "ACME-10-K-2025-09-27.txt",
                    "ingested_at": "2025-11-01T09:30:00Z",
                    "chunks": 193,
                }
            ]
        }
    )

    doc_id: str = Field(description="The sha256 of the document's bytes.")
    title: str = Field(description="The document's title.")
    collection_id: str = Field(description="The collection the document belongs to.")
    doc_type: str = Field(description="The kind of document, such as `10-K` or `runbook`.")
    doc_date: date | None = Field(description="The date the document is about, if known.")
    metadata: dict[str, str] = Field(description="Fields only some collections have.")
    filename: str = Field(description="The original file name.")
    ingested_at: datetime = Field(description="When it was indexed.")
    chunks: int = Field(description="Passages indexed from this document.")


class UploadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"job_id": "933dac8b", "doc_id": "9f2c1e7a"}]}
    )

    job_id: str = Field(description="Poll this with GET /api/jobs/{job_id}.")
    doc_id: str = Field(description="The sha256 of the file's bytes.")


class JobResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "embedding", "progress": 60, "error": None, "chunks": None}]
        }
    )

    status: JobStatus = Field(description="`indexed` and `failed` are final.")
    progress: int = Field(ge=0, le=100, description="Percentage done.")
    error: str | None = Field(description="What went wrong, when status is `failed`.")
    chunks: int | None = Field(description="Passages indexed, when status is `indexed`.")
