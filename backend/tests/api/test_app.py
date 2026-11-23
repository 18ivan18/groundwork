import asyncio
from pathlib import Path

import pytest
from api_helpers import add_document
from fastapi import FastAPI
from fastapi.testclient import TestClient

from groundwork import container
from groundwork.config import Settings
from groundwork.main import create_app
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.memory_store import MemoryStore

ORIGIN = "http://localhost:5173"


def preflight(client: TestClient, origin: str) -> str | None:
    response = client.options(
        "/api/ask",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )
    return response.headers.get("access-control-allow-origin")


def test_cors_allows_the_configured_origin(app: FastAPI) -> None:
    assert preflight(TestClient(app), ORIGIN) == ORIGIN


def test_cors_rejects_other_origins(app: FastAPI) -> None:
    assert preflight(TestClient(app), "http://evil.test") is None


def test_docs_are_served_under_api(app: FastAPI) -> None:
    client = TestClient(app)

    assert client.get("/api/docs").status_code == 200
    assert client.get("/docs").status_code == 404


def test_openapi_lists_the_routes_and_the_502(app: FastAPI) -> None:
    schema = TestClient(app).get("/api/openapi.json").json()

    assert set(schema["paths"]) == {
        "/api/health",
        "/api/config",
        "/api/ask",
        "/api/documents",
        "/api/jobs/{job_id}",
    }
    assert "502" in schema["paths"]["/api/ask"]["post"]["responses"]


def test_every_route_is_documented(app: FastAPI) -> None:
    schema = TestClient(app).get("/api/openapi.json").json()

    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            where = f"{method} {path}"
            assert operation.get("summary"), where
            assert operation.get("description"), where
            assert "500" in operation["responses"], where

    for name, model in schema["components"]["schemas"].items():
        if name not in {"HTTPValidationError", "ValidationError"}:
            for field, details in model.get("properties", {}).items():
                assert "description" in details, f"{name}.{field}"


def test_startup_fails_fast_without_an_llm_model(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_dir=tmp_path))

    with pytest.raises(ValueError, match="requires LLM_MODEL"), TestClient(app):
        pass


def test_startup_publishes_what_an_earlier_process_left(
    tmp_path: Path, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container, "build_embedder", lambda settings: FakeEmbedder())
    settings = Settings(_env_file=None, data_dir=tmp_path, llm_model="any", reranker_model="")
    asyncio.run(add_document(store, "killed ingest", 1))

    with TestClient(create_app(settings, store=store)):
        assert asyncio.run(store.generation()) == 1

    assert asyncio.run(store.publish()) is None
