"""BM25 keyword search, derived from one index's chunks in the store.

bm25s cannot filter, but it can weight: `weight_mask` multiplies every chunk's score before the
top n are taken. Arrays aligned with bm25s's positions (each chunk's document, and each document's
collection, type, date, metadata and groups) turn search filters into that mask, so a filter never
leaves fewer than n results behind.

`SparseCache` keeps one SparseIndex per publication of an index on disk, under
`<root>/<index id>/<generation>-<token>/`, and each API worker reloads it when that changes.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import bm25s
import numpy as np
import Stemmer

from groundwork.domain.models import Chunk, Document, SearchFilters, indexed_text
from groundwork.retrieval.indexes import Publication
from groundwork.retrieval.store import ChunkStore, matches

logger = logging.getLogger(__name__)

_SIDE_FILE = "positions.json"
_STEMMER = Stemmer.Stemmer("english")


def _tokenize(texts: list[str]) -> bm25s.tokenization.Tokenized:
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=False)


def _document_json(document: Document) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "collection_id": document.collection_id,
        "doc_type": document.doc_type,
        "doc_date": document.doc_date.isoformat() if document.doc_date else None,
        "filename": document.filename,
        "metadata": dict(document.metadata),
        "access_groups": list(document.access_groups),
        "owner_id": document.owner_id,
        "ingested_at": document.ingested_at.isoformat(),
    }


def _document_from_json(data: dict[str, Any]) -> Document:
    return Document(
        doc_id=data["doc_id"],
        title=data["title"],
        collection_id=data["collection_id"],
        doc_type=data["doc_type"],
        doc_date=date.fromisoformat(data["doc_date"]) if data["doc_date"] else None,
        filename=data["filename"],
        metadata=data["metadata"],
        access_groups=tuple(data["access_groups"]),
        owner_id=data["owner_id"],
        ingested_at=datetime.fromisoformat(data["ingested_at"]),
    )


@dataclass(frozen=True)
class Positions:
    """What is known about each bm25s position, for filtering before the top n."""

    chunk_ids: list[str]
    documents: list[Document]
    # For each position, the index of its document in `documents`.
    document_of: np.ndarray
    # Group name -> bit, built from the groups this index holds: M3 adds a `private:<user>`
    # group per uploader, so the set is not fixed. Bit i of `group_bits[p]` is set when the
    # chunk at position p is readable by group i.
    group_bit: dict[str, int] = field(default_factory=dict)
    group_bits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))

    @classmethod
    def from_passages(cls, passages: Sequence[tuple[Chunk, Document]]) -> "Positions":
        documents: dict[str, int] = {}
        ordered: list[Document] = []
        document_of = []
        for _, document in passages:
            if document.doc_id not in documents:
                documents[document.doc_id] = len(ordered)
                ordered.append(document)
            document_of.append(documents[document.doc_id])
        return cls.create([chunk.chunk_id for chunk, _ in passages], ordered, document_of)

    @classmethod
    def create(
        cls, chunk_ids: list[str], documents: list[Document], document_of: Sequence[int]
    ) -> "Positions":
        groups = sorted({group for document in documents for group in document.access_groups})
        if len(groups) > 64:
            raise ValueError(f"{len(groups)} access groups do not fit a 64-bit mask")
        group_bit = {group: bit for bit, group in enumerate(groups)}
        document_bits = np.array(
            [sum(1 << group_bit[g] for g in document.access_groups) for document in documents],
            dtype=np.uint64,
        )
        positions = np.asarray(document_of, dtype=np.int64)
        return cls(
            chunk_ids=chunk_ids,
            documents=documents,
            document_of=positions,
            group_bit=group_bit,
            group_bits=document_bits[positions] if documents else np.zeros(0, np.uint64),
        )

    def mask(self, filters: SearchFilters | None) -> np.ndarray | None:
        """1.0 where a chunk passes the filters, else 0.0; None when nothing is filtered."""
        if filters is None or filters == SearchFilters():
            return None
        allowed = np.array([matches(d, filters) for d in self.documents], dtype=np.float32)
        result: np.ndarray = allowed[self.document_of]
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_ids": self.chunk_ids,
            "documents": [_document_json(document) for document in self.documents],
            "document_of": self.document_of.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Positions":
        documents = [_document_from_json(item) for item in data["documents"]]
        return cls.create(data["chunk_ids"], documents, data["document_of"])


class SparseIndex:
    def __init__(self, retriever: bm25s.BM25 | None, positions: Positions) -> None:
        self._retriever = retriever
        self._positions = positions

    @classmethod
    def build(cls, passages: Sequence[tuple[Chunk, Document]]) -> "SparseIndex":
        positions = Positions.from_passages(passages)
        if not passages:
            return cls(None, positions)
        retriever = bm25s.BM25()
        texts = [indexed_text(chunk, document) for chunk, document in passages]
        retriever.index(_tokenize(texts), show_progress=False)
        return cls(retriever, positions)

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        positions = Positions.from_json(json.loads((path / _SIDE_FILE).read_text()))
        if not positions.chunk_ids:
            return cls(None, positions)
        return cls(bm25s.BM25.load(path, show_progress=False), positions)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._retriever is not None:
            self._retriever.save(path, show_progress=False)
        (path / _SIDE_FILE).write_text(json.dumps(self._positions.to_json()))

    def __len__(self) -> int:
        return len(self._positions.chunk_ids)

    def search(
        self, query: str, n: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        """Up to n (chunk_id, score) pairs, best first. Chunks the filters exclude score 0."""
        if self._retriever is None or n <= 0:
            return []
        results = self._retriever.retrieve(
            _tokenize([query]),
            k=min(n, len(self)),
            show_progress=False,
            weight_mask=self._positions.mask(filters),
        )
        hits = []
        for position, score in zip(results.documents[0], results.scores[0], strict=True):
            # bm25s also returns chunks that share no term with the query, with a score of 0.
            if score <= 0:
                break
            hits.append((self._positions.chunk_ids[int(position)], float(score)))
        return hits


class SparseSource(Protocol):
    async def current(self) -> SparseIndex: ...


class FixedSparse:
    """A SparseIndex that never changes, for the evaluation and tests."""

    def __init__(self, index: SparseIndex) -> None:
        self._index = index

    async def current(self) -> SparseIndex:
        return self._index


async def build_sparse(store: ChunkStore) -> SparseIndex:
    passages = await store.all_passages()
    return await asyncio.to_thread(SparseIndex.build, passages)


class SparseCache:
    """The BM25 index for the store's last publication.

    The publication is read at most once per `check_seconds`. When its generation or its token has
    changed, the index for the new one is loaded from disk, or built from the store and saved
    there for the other workers, unless the store holds writes not yet published. The token keeps
    a recreated or restored database from loading what was saved for another one that had reached
    the same generation. Two workers building the same publication at once both succeed: each
    writes to its own temporary folder, and whichever renames second finds the folder there and
    keeps it.
    """

    def __init__(
        self,
        store: ChunkStore,
        root: Path,
        *,
        check_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._root = root / str(store.index.id)
        self._check_seconds = check_seconds
        self._clock = clock
        self._index: SparseIndex | None = None
        # What `_index` was built from.
        self._publication: Publication | None = None
        self._checked_at = 0.0
        self._lock = asyncio.Lock()

    async def current(self) -> SparseIndex:
        if self._index is not None and self._clock() - self._checked_at < self._check_seconds:
            return self._index
        async with self._lock:
            publication = await self._store.publication()
            self._checked_at = self._clock()
            if self._index is None or publication != self._publication:
                self._index = await self._load_or_build(publication)
                self._publication = publication
            return self._index

    async def _load_or_build(self, publication: Publication) -> SparseIndex:
        path = self._root / publication.folder_name
        if path.exists():
            return await asyncio.to_thread(SparseIndex.load, path)
        index = await build_sparse(self._store)
        # Read after the build: if nothing was written since the publication even now, the index
        # holds exactly what it published. Otherwise the index is this worker's only, because a
        # backup could bring the publication back without those writes.
        after = await self._store.publication()
        if after == publication and not after.written_since:
            await asyncio.to_thread(self._save, index, publication)
        logger.info(
            "Built BM25 for index %s generation %s",
            self._store.index.name,
            publication.generation,
        )
        return index

    def _save(self, index: SparseIndex, publication: Publication) -> None:
        path = self._root / publication.folder_name
        self._root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self._root, prefix=".building-"))
        try:
            index.save(staging)
            os.rename(staging, path)
        except OSError:
            # Another worker saved this generation first; its copy is the same.
            if not path.exists():
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        # Keep the previous generation for a worker still reading it; older ones go. So does this
        # generation with another token: it is from before the database was recreated or restored.
        current = publication.generation
        for old in self._root.iterdir():
            generation = Publication.generation_in(old.name)
            if generation is None or old == path:
                continue
            if generation < current - 1 or generation == current:
                shutil.rmtree(old, ignore_errors=True)
