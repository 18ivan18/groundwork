"""The chunk store: where one index's chunks, vectors and documents live.

`PgVectorStore` is the store the app runs on. `MemoryStore` keeps everything in a list, for tests
that need no database; the contract tests in tests/unit/test_store.py run against both.
"""

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, fields
from typing import Any, Protocol

from groundwork.domain.models import Chunk, Document, DocumentMeta, SearchFilters
from groundwork.retrieval.indexes import IndexInfo, Publication


@dataclass(frozen=True)
class DenseHit:
    chunk: Chunk
    document: Document
    similarity: float


def normalized_groups(groups: Sequence[str]) -> tuple[str, ...]:
    """Sorted and without duplicates, as the documents table stores them."""
    return tuple(sorted(set(groups)))


def _tags(meta: DocumentMeta) -> dict[str, Any]:
    values = {field.name: getattr(meta, field.name) for field in fields(DocumentMeta)}
    values["metadata"] = dict(meta.metadata)
    values["access_groups"] = normalized_groups(meta.access_groups)
    return values


@dataclass(frozen=True)
class CorpusDocument:
    """A document stored by a corpus sync. Its collection and key identify it."""

    document: Document
    key: str
    content_hash: str
    # Whether this index holds its chunks: another index may hold them instead.
    in_index: bool

    def is_current(self, content_hash: str, meta: DocumentMeta) -> bool:
        """Whether it was stored from these bytes and tags, access groups included."""
        return self.content_hash == content_hash and _tags(self.document) == _tags(meta)


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


class ChunkStore(Protocol):
    """One index's chunks. A document counts as indexed once its chunks are stored: `add` writes
    them all in one transaction, so there is no half-indexed document to find."""

    @property
    def index(self) -> IndexInfo: ...

    async def find_document(self, content_hash: str, meta: DocumentMeta) -> str | None:
        """The id of the document with these bytes, collection, groups and owner, if this index
        holds its chunks: an upload if there is one, else a corpus document."""
        ...

    async def find_upload(self, content_hash: str, meta: DocumentMeta) -> str | None:
        """As `find_document`, but never a corpus document."""
        ...

    async def add(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        publish: bool = False,
    ) -> Document:
        """Store an upload and its chunks, which get their ids here: the chunk at position i
        becomes `<doc_id>:<i>`. One stored before keeps its tags, and its chunks in this index are
        replaced. The same transaction records that the index has writes to publish, or with
        `publish` publishes them."""
        ...

    async def add_corpus_document(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        manifest_key: str,
    ) -> Document | None:
        """As `add`, but the corpus document of this collection and key. One stored from other
        bytes or tags is removed from every index in the same transaction. An upload never becomes
        a corpus document, nor the other way round.

        None if this index holds an upload with the same bytes, collection, groups and owner. The
        upload stands in: nothing is stored, and the corpus document is removed as a changed one
        is if it has chunks in this index or was stored from other bytes or tags."""
        ...

    async def corpus_documents(self) -> list[CorpusDocument]:
        """Every corpus document, including those only other indexes hold chunks of."""
        ...

    async def remove_corpus_documents(self, doc_ids: Sequence[str]) -> None:
        """Remove corpus documents from every index, in one transaction that also records that
        each index which held one has writes to publish. Ids of uploads are ignored."""
        ...

    def syncing(self) -> AbstractAsyncContextManager[None]:
        """Held for the length of a corpus sync: another one waits, whichever index it is for."""
        ...

    async def search(
        self, query_embedding: Sequence[float], n: int, filters: SearchFilters | None = None
    ) -> list[DenseHit]: ...

    async def get(self, chunk_ids: Sequence[str]) -> dict[str, tuple[Chunk, Document]]: ...

    async def all_passages(self) -> list[tuple[Chunk, Document]]:
        """Every chunk with its document, in document then chunk order."""
        ...

    async def matching_ids(self, filters: SearchFilters) -> set[str]: ...

    async def documents(self) -> list[tuple[Document, int]]:
        """Each document with chunks in this index, with its chunk count, by title."""
        ...

    async def count_chunks(self) -> int: ...

    async def count_documents(self) -> int: ...

    async def count_document_chunks(self, doc_id: str) -> int: ...

    async def publication(self) -> Publication:
        """The last publication: what a saved BM25 index must match to be loaded."""
        ...

    async def publish(self) -> int | None:
        """Move the generation, so that BM25 finds what was written to the index since the last
        publication, whichever process wrote it. Returns the new generation, or None if there
        was nothing to publish."""
        ...

    async def publish_all(self) -> dict[str, int]:
        """As `publish`, for every index with writes to publish, not only this one. The new
        generation of each, by index name."""
        ...
