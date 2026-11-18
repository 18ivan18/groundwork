"""A chunk store in memory: exact cosine search over a list. For tests that need no database."""

import itertools
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from groundwork.domain.models import (
    Chunk,
    Document,
    DocumentMeta,
    SearchFilters,
    make_chunk_id,
)
from groundwork.retrieval.indexes import IndexInfo, IndexSpec
from groundwork.retrieval.store import DenseHit, matches, normalized_groups

_Identity = tuple[str, str, tuple[str, ...], str | None]


def memory_index(dimension: int, name: str = "memory") -> IndexInfo:
    spec = IndexSpec(name=name, chunker="window", embedder_model="fake", dimension=dimension)
    return IndexInfo(id=1, spec=spec, status="ready", generation=0, is_active=True)


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array


class MemoryStore:
    def __init__(self, index: IndexInfo) -> None:
        self._index = index
        self._ids = itertools.count(1)
        self._documents: dict[_Identity, Document] = {}
        # doc_id -> its chunks and their unit vectors, in order.
        self._chunks: dict[str, list[tuple[Chunk, np.ndarray]]] = {}

    @property
    def index(self) -> IndexInfo:
        return self._index

    @staticmethod
    def _identity(content_hash: str, meta: DocumentMeta) -> _Identity:
        groups = normalized_groups(meta.access_groups)
        return (content_hash, meta.collection_id, groups, meta.owner_id)

    async def find_document(self, content_hash: str, meta: DocumentMeta) -> str | None:
        document = self._documents.get(self._identity(content_hash, meta))
        return document.doc_id if document and self._chunks.get(document.doc_id) else None

    async def add(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk needs one embedding")
        identity = self._identity(content_hash, meta)
        document = self._documents.get(identity)
        if document is None:
            document = Document(
                title=meta.title,
                collection_id=meta.collection_id,
                doc_type=meta.doc_type,
                doc_date=meta.doc_date,
                filename=meta.filename,
                metadata=dict(meta.metadata),
                access_groups=identity[2],
                owner_id=meta.owner_id,
                doc_id=str(next(self._ids)),
                ingested_at=datetime.now(UTC),
            )
            self._documents[identity] = document
        self._chunks[document.doc_id] = [
            (
                replace(chunk, chunk_id=make_chunk_id(document.doc_id, i), doc_id=document.doc_id),
                _unit(embedding),
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        return document

    def _by_id(self) -> dict[str, Document]:
        return {document.doc_id: document for document in self._documents.values()}

    def _passages(
        self, filters: SearchFilters | None = None
    ) -> list[tuple[Chunk, Document, np.ndarray]]:
        documents = self._by_id()
        return [
            (chunk, documents[doc_id], vector)
            for doc_id, items in sorted(self._chunks.items(), key=lambda item: int(item[0]))
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
            ((documents[doc_id], len(items)) for doc_id, items in self._chunks.items() if items),
            key=lambda item: (item[0].title, int(item[0].doc_id)),
        )

    async def count_chunks(self) -> int:
        return sum(len(items) for items in self._chunks.values())

    async def count_documents(self) -> int:
        return sum(bool(items) for items in self._chunks.values())

    async def count_document_chunks(self, doc_id: str) -> int:
        return len(self._chunks.get(doc_id, []))
