"""Core domain types, independent of the API and storage layers."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

# Everyone can read a document in this group. Real access control arrives in M3.
EVERYONE = "everyone"


# Keyword-only, so Document can add required fields after the defaults here.
@dataclass(frozen=True, kw_only=True)
class DocumentMeta:
    """What is known about a file before ingestion.

    `doc_type` is free text ("10-K", "runbook", "novel"), set by the collection's manifest or the
    uploader. `metadata` holds what only some collections have, such as a filing's ticker, as
    strings so a search can filter on them by equality.
    """

    title: str
    collection_id: str
    doc_type: str
    doc_date: date | None
    filename: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    access_groups: tuple[str, ...] = (EVERYONE,)
    owner_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class Document(DocumentMeta):
    """An ingested file. `doc_id` is the sha256 of its bytes, which also deduplicates uploads."""

    doc_id: str
    ingested_at: datetime


def make_doc_id(raw: bytes) -> str:
    """The sha256 of a file's bytes, so a file uploaded twice is indexed once."""
    return hashlib.sha256(raw).hexdigest()


def make_chunk_id(doc_id: str, position: int) -> str:
    """Chunk 0's id also marks a document as fully indexed: the dense index writes it last."""
    return f"{doc_id}:{position}"


@dataclass(frozen=True)
class SearchFilters:
    """Restricts a search. An empty tuple means no restriction on that field."""

    collections: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    # Inclusive. A document without a date never matches a date range.
    date_from: date | None = None
    date_to: date | None = None
    # Every key must have exactly this value in the document's metadata.
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A passage of a document. `char_start` and `char_end` index the loaded document text."""

    chunk_id: str
    doc_id: str
    text: str
    section: str | None
    page: int | None
    char_start: int
    char_end: int


def indexed_text(chunk: Chunk, document: Document) -> str:
    """The text to embed and to index for BM25, which is not the text shown to users.

    Filings say "the Company", so without this header a chunk has no link to the company that a
    question names, and the same passage from three filings looks alike to both retrievers. The
    title does the same for any document whose text doesn't name its subject.
    """
    header = " · ".join(part for part in (document.title, chunk.section) if part)
    return f"{header}\n{chunk.text}"


@dataclass(frozen=True)
class ScoredPassage:
    """A retrieved chunk. Ranks start at 1; None means that retriever did not return it, or that
    the passage was not re-ranked."""

    chunk: Chunk
    document: Document
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    dense_similarity: float | None
    rerank_rank: int | None = None
    # Raw cross-encoder output: comparable only within one search, not across questions or models.
    rerank_score: float | None = None
