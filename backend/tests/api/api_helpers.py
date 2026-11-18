"""Shared by the API tests. Fixtures live in conftest.py, which importlib mode cannot import."""

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from groundwork.domain.models import Chunk, Document, DocumentMeta, ScoredPassage, SearchFilters
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.hybrid import SearchResult
from groundwork.retrieval.store import ChunkStore

DOCUMENT = Document(
    title="ACME Corp · Form 10-K",
    collection_id="sec-10k",
    doc_type="10-K",
    doc_date=date(2025, 9, 27),
    metadata={"company": "ACME Corp", "ticker": "ACME"},
    filename="acme.txt",
    doc_id="doc",
    ingested_at=datetime(2025, 11, 1, tzinfo=UTC),
)


def make_passage(number: int, similarity: float = 0.9) -> ScoredPassage:
    chunk = Chunk(
        f"doc:{number}", "doc", f"Anvil fact {number}.", ("Item 1 — Business",), None, 0, 1
    )
    return ScoredPassage(
        chunk,
        DOCUMENT,
        number,
        number + 1,
        0.0323,
        similarity,
        rerank_rank=number,
        rerank_score=1.0,
    )


class StubRetriever:
    def __init__(self, passages: list[ScoredPassage]) -> None:
        self.passages = passages
        self.calls: list[tuple[int, SearchFilters | None]] = []

    async def retrieve(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> SearchResult:
        self.calls.append((k, filters))
        similarities = [p.dense_similarity for p in self.passages if p.dense_similarity]
        return SearchResult(self.passages[:k], max(similarities, default=None))


ClientFactory = Callable[..., TestClient]


async def add_document(store: ChunkStore, title: str, chunks: int, **meta: Any) -> Document:
    """Store a document of `chunks` identical chunks, tagged like DOCUMENT unless `meta` says
    otherwise."""
    fields = {
        "title": title,
        "collection_id": DOCUMENT.collection_id,
        "doc_type": DOCUMENT.doc_type,
        "doc_date": DOCUMENT.doc_date,
        "metadata": DOCUMENT.metadata,
        "filename": DOCUMENT.filename,
        **meta,
    }
    items = [Chunk("x", "x", "anvils", (), None, 0, 6) for _ in range(chunks)]
    vectors = FakeEmbedder().embed_documents(["anvils"] * chunks)
    return await store.add(DocumentMeta(**fields), f"hash-{title}", items, vectors)


def as_user(app: FastAPI, user: str = "alice", **kwargs: Any) -> TestClient:
    """A client whose every request sends `X-User: user`."""
    return TestClient(app, headers={"X-User": user}, **kwargs)
