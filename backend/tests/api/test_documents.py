import asyncio

from api_helpers import add_document, as_user
from fastapi import FastAPI

from groundwork.retrieval.memory_store import MemoryStore


def add(store: MemoryStore, title: str, chunks: int) -> str:
    return asyncio.run(add_document(store, title, chunks)).doc_id


def test_empty_library(app: FastAPI) -> None:
    assert as_user(app).get("/api/documents").json() == []


def test_documents_are_listed_with_chunk_counts_by_title(app: FastAPI, store: MemoryStore) -> None:
    add(store, "Beta report", 2)
    alpha = add(store, "Alpha report", 3)

    body = as_user(app).get("/api/documents").json()

    assert [(d["title"], d["chunks"]) for d in body] == [("Alpha report", 3), ("Beta report", 2)]
    assert body[0].pop("ingested_at")
    assert body[0] == {
        "doc_id": alpha,
        "title": "Alpha report",
        "collection_id": "sec-10k",
        "doc_type": "10-K",
        "doc_date": "2025-09-27",
        "metadata": {"company": "ACME Corp", "ticker": "ACME"},
        "filename": "acme.txt",
        "access_groups": ["everyone"],
        "owner_id": None,
        "chunks": 3,
    }
