import asyncio

from api_helpers import add_document
from fastapi import FastAPI
from fastapi.testclient import TestClient

from groundwork.retrieval.memory_store import MemoryStore


def test_health_reports_configuration(app: FastAPI) -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_provider": "openai_compatible",
        "embedder": "BAAI/bge-small-en-v1.5",
        "documents": 0,
        "chunks": 0,
    }


def test_health_counts_what_is_indexed(app: FastAPI, store: MemoryStore) -> None:
    asyncio.run(add_document(store, "ACME Corp · Form 10-K", 2))

    body = TestClient(app).get("/api/health").json()

    assert body["documents"] == 1
    assert body["chunks"] == 2
