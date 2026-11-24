"""A chunk store in memory: exact cosine search over a list. For tests that need no database."""

import asyncio
import itertools
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import numpy as np

from groundwork.domain.models import (
    Chunk,
    Document,
    DocumentMeta,
    SearchFilters,
    make_chunk_id,
)
from groundwork.retrieval.indexes import IndexInfo, IndexSpec, Publication
from groundwork.retrieval.store import CorpusDocument, DenseHit, matches, normalized_groups


def memory_index(dimension: int, name: str = "memory") -> IndexInfo:
    spec = IndexSpec(name=name, chunker="window", embedder_model="fake", dimension=dimension)
    return IndexInfo(id=1, spec=spec, status="ready", generation=0, is_active=True)


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


@dataclass(frozen=True)
class _Stored:
    document: Document
    content_hash: str
    # None for an upload.
    manifest_key: str | None


@dataclass
class MemoryData:
    """What a MemoryStore holds. Two stores given the same data behave like two processes on one
    database."""

    ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    documents: dict[str, _Stored] = field(default_factory=dict)
    # doc_id -> its chunks and their unit vectors, in order.
    chunks: dict[str, list[tuple[Chunk, np.ndarray]]] = field(default_factory=dict)
    generation: int = 0
    token: str = field(default_factory=lambda: str(uuid.uuid4()))
    unpublished: bool = False
    sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MemoryStore:
    def __init__(self, index: IndexInfo, data: MemoryData | None = None) -> None:
        self._index = index
        self._data = data if data is not None else MemoryData()

    @property
    def index(self) -> IndexInfo:
        return self._index

    def _identical(self, content_hash: str, meta: DocumentMeta) -> list[_Stored]:
        """The documents with these bytes, collection, groups and owner, uploads first."""
        groups = normalized_groups(meta.access_groups)
        identity = (content_hash, meta.collection_id, groups, meta.owner_id)
        found = [
            stored
            for stored in self._data.documents.values()
            if identity
            == (
                stored.content_hash,
                stored.document.collection_id,
                stored.document.access_groups,
                stored.document.owner_id,
            )
        ]
        return sorted(found, key=lambda stored: stored.manifest_key is not None)

    async def find_document(self, content_hash: str, meta: DocumentMeta) -> str | None:
        for stored in self._identical(content_hash, meta):
            if self._data.chunks.get(stored.document.doc_id):
                return stored.document.doc_id
        return None

    async def find_upload(self, content_hash: str, meta: DocumentMeta) -> str | None:
        for stored in self._identical(content_hash, meta):
            if stored.manifest_key is None and self._data.chunks.get(stored.document.doc_id):
                return stored.document.doc_id
        return None

    async def add(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        publish: bool = False,
    ) -> Document:
        uploads = [s for s in self._identical(content_hash, meta) if s.manifest_key is None]
        existing = uploads[0].document if uploads else None
        document = self._store(meta, content_hash, chunks, embeddings, existing, None)
        if publish:
            self._publish()
        return document

    async def add_corpus_document(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        manifest_key: str,
    ) -> Document | None:
        uploaded = await self.find_upload(content_hash, meta) is not None
        existing: Document | None = None
        for item in await self.corpus_documents():
            if (item.document.collection_id, item.key) == (meta.collection_id, manifest_key):
                current = item.is_current(content_hash, meta)
                if not current or (uploaded and item.in_index):
                    await self.remove_corpus_documents([item.document.doc_id])
                else:
                    existing = item.document
        if uploaded:
            return None
        return self._store(meta, content_hash, chunks, embeddings, existing, manifest_key)

    def _store(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        document: Document | None,
        manifest_key: str | None,
    ) -> Document:
        """Store the chunks of `document`, or of a new one."""
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk needs one embedding")
        if document is None:
            document = Document(
                title=meta.title,
                collection_id=meta.collection_id,
                doc_type=meta.doc_type,
                doc_date=meta.doc_date,
                filename=meta.filename,
                metadata=dict(meta.metadata),
                access_groups=normalized_groups(meta.access_groups),
                owner_id=meta.owner_id,
                doc_id=str(next(self._data.ids)),
                ingested_at=datetime.now(UTC),
            )
            self._data.documents[document.doc_id] = _Stored(document, content_hash, manifest_key)
        self._data.chunks[document.doc_id] = [
            (
                replace(chunk, chunk_id=make_chunk_id(document.doc_id, i), doc_id=document.doc_id),
                _unit(embedding),
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        self._data.unpublished = True
        return document

    async def corpus_documents(self) -> list[CorpusDocument]:
        return [
            CorpusDocument(
                stored.document,
                key=stored.manifest_key,
                content_hash=stored.content_hash,
                in_index=bool(self._data.chunks.get(doc_id)),
            )
            for doc_id, stored in self._data.documents.items()
            if stored.manifest_key is not None
        ]

    async def remove_corpus_documents(self, doc_ids: Sequence[str]) -> None:
        for doc_id in doc_ids:
            if doc_id in self._data.documents and self._data.documents[doc_id].manifest_key:
                del self._data.documents[doc_id]
                if self._data.chunks.pop(doc_id, []):
                    self._data.unpublished = True

    @asynccontextmanager
    async def syncing(self) -> AsyncIterator[None]:
        async with self._data.sync_lock:
            yield

    def _by_id(self) -> dict[str, Document]:
        return {doc_id: stored.document for doc_id, stored in self._data.documents.items()}

    def _passages(
        self, filters: SearchFilters | None = None
    ) -> list[tuple[Chunk, Document, np.ndarray]]:
        documents = self._by_id()
        return [
            (chunk, documents[doc_id], vector)
            for doc_id, items in sorted(self._data.chunks.items(), key=lambda item: int(item[0]))
            if matches(documents[doc_id], filters)
            for chunk, vector in items
        ]

    async def search(
        self, query_embedding: Sequence[float], n: int, filters: SearchFilters | None = None
    ) -> list[DenseHit]:
        query = _unit(query_embedding)
        scored = [
            DenseHit(chunk, document, similarity=float(vector @ query))
            for chunk, document, vector in self._passages(filters)
        ]
        # sorted() is stable, so ties keep document then chunk order.
        return sorted(scored, key=lambda hit: -hit.similarity)[:n]

    async def get(self, chunk_ids: Sequence[str]) -> dict[str, tuple[Chunk, Document]]:
        wanted = set(chunk_ids)
        return {
            chunk.chunk_id: (chunk, document)
            for chunk, document, _ in self._passages()
            if chunk.chunk_id in wanted
        }

    async def all_passages(self) -> list[tuple[Chunk, Document]]:
        return [(chunk, document) for chunk, document, _ in self._passages()]

    async def matching_ids(self, filters: SearchFilters) -> set[str]:
        return {chunk.chunk_id for chunk, _, _ in self._passages(filters)}

    async def documents(self) -> list[tuple[Document, int]]:
        documents = self._by_id()
        return sorted(
            (
                (documents[doc_id], len(items))
                for doc_id, items in self._data.chunks.items()
                if items
            ),
            key=lambda item: (item[0].title, int(item[0].doc_id)),
        )

    async def count_chunks(self) -> int:
        return sum(len(items) for items in self._data.chunks.values())

    async def count_documents(self) -> int:
        return sum(bool(items) for items in self._data.chunks.values())

    async def count_document_chunks(self, doc_id: str) -> int:
        return len(self._data.chunks.get(doc_id, []))

    async def publication(self) -> Publication:
        return Publication(self._data.generation, self._data.token, self._data.unpublished)

    async def publish(self) -> int | None:
        return self._publish()

    async def publish_all(self) -> dict[str, int]:
        generation = self._publish()
        return {} if generation is None else {self._index.name: generation}

    async def vacuum(self) -> None:
        """Nothing to do: a removed chunk leaves nothing behind."""

    def _publish(self) -> int | None:
        if not self._data.unpublished:
            return None
        self._data.generation += 1
        self._data.token = str(uuid.uuid4())
        self._data.unpublished = False
        return self._data.generation
