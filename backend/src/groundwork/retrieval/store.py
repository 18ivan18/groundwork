"""The chunk store: where one index's chunks, vectors and documents live.

`PgVectorStore` is the store the app runs on. `MemoryStore` keeps everything in a list, for tests
that need no database; the contract tests in tests/unit/test_store.py run against both.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from groundwork.domain.models import (
    Access,
    Chunk,
    Document,
    DocumentChanges,
    DocumentMeta,
    SearchFilters,
)
from groundwork.retrieval.indexes import IndexInfo


@dataclass(frozen=True)
class DenseHit:
    chunk: Chunk
    document: Document
    similarity: float


def normalized_groups(groups: Sequence[str]) -> tuple[str, ...]:
    """Sorted and without duplicates, as the documents table stores them."""
    return tuple(sorted(set(groups)))


def matches(document: Document, filters: SearchFilters | None) -> bool:
    """Whether a document passes the filters, for stores that filter in Python."""
    if filters is None:
        return True
    if filters.collections and document.collection_id not in filters.collections:
        return False
    if filters.doc_types and document.doc_type not in filters.doc_types:
        return False
    if filters.date_from or filters.date_to:
        if document.doc_date is None:
            return False
        if filters.date_from and document.doc_date < filters.date_from:
            return False
        if filters.date_to and document.doc_date > filters.date_to:
            return False
    return all(document.metadata.get(key) == value for key, value in filters.metadata.items())


class DuplicateDocumentError(ValueError):
    """Another stored document already has these bytes, collection, groups and owner."""


class ChunkStore(Protocol):
    """One index's chunks. A document counts as indexed once its chunks are stored: `add` writes
    them all in one transaction, so there is no half-indexed document to find."""

    @property
    def index(self) -> IndexInfo: ...

    async def find_document(self, content_hash: str, meta: DocumentMeta) -> str | None:
        """The id of the document with these bytes, collection, groups and owner, if this index
        holds its chunks."""
        ...

    async def add(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        """Store a document and its chunks, which get their ids here: the chunk at position i
        becomes `<doc_id>:<i>`. A document stored before keeps its tags, and its chunks in this
        index are replaced."""
        ...

    async def search(
        self,
        query_embedding: Sequence[float],
        n: int,
        filters: SearchFilters | None = None,
        *,
        access: Access,
    ) -> list[DenseHit]:
        """The n nearest chunks that `access` may read and the filters keep. A filter that
        leaves fewer than n chunks returns all of them."""
        ...

    async def get(
        self, chunk_ids: Sequence[str], *, access: Access
    ) -> dict[str, tuple[Chunk, Document]]:
        """The chunks among `chunk_ids` that `access` may read. The ACL is checked here again,
        though the ids come from a search that checked it: defence in depth."""
        ...

    async def all_passages(self) -> list[tuple[Chunk, Document]]:
        """Every chunk with its document, in document then chunk order."""
        ...

    async def matching_ids(self, filters: SearchFilters) -> set[str]: ...

    async def documents(self, *, access: Access) -> list[tuple[Document, int]]:
        """Each document `access` may read with chunks in this index, with its chunk count, by
        title."""
        ...

    async def document(self, doc_id: str, *, access: Access) -> tuple[Document, int] | None:
        """The document, with its chunk count, if `access` may read it and this index holds it."""
        ...

    async def document_indexes(self, doc_id: str) -> list[str]:
        """The names of every index holding chunks of the document."""
        ...

    async def update_document(self, doc_id: str, changes: DocumentChanges) -> Document:
        """Change a document's tags, its chunks' copies of them included, in one transaction.
        Raises DuplicateDocumentError when the new groups make it another stored document."""
        ...

    async def delete_document(self, doc_id: str) -> None:
        """Delete the document and its chunks in every index."""
        ...

    async def storage_path_in_use(self, storage_path: str) -> bool:
        """Whether any document still names this file."""
        ...

    async def protected_collections(self) -> frozenset[str]:
        """Collections whose documents only an admin may delete."""
        ...

    async def protect_collection(self, collection_id: str) -> None: ...

    async def count_chunks(self) -> int: ...

    async def count_documents(self) -> int: ...

    async def count_document_chunks(self, doc_id: str) -> int: ...

    async def generation(self) -> int:
        """Changes when the index's chunks do, so a BM25 index built from them knows it is old."""
        ...

    async def bump_generation(self) -> int:
        """Called once an ingest job has stored all its documents."""
        ...
