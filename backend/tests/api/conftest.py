"""An app wired to fakes: no network, no API keys, no model download."""

from pathlib import Path

import pytest
from api_helpers import ClientFactory, StubRetriever, as_user, make_passage
from fastapi import FastAPI
from fastapi.testclient import TestClient
from upload_harness import Harness

from groundwork.api.deps import get_qa_service
from groundwork.config import Settings
from groundwork.llm.fake import FakeLLM
from groundwork.main import create_app
from groundwork.qa.service import QAService
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.memory_store import MemoryStore, memory_index


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(memory_index(FakeEmbedder().dimension))


@pytest.fixture
def app(tmp_path: Path, store: MemoryStore) -> FastAPI:
    return create_app(Settings(_env_file=None, data_dir=tmp_path), store=store)


@pytest.fixture
def retriever() -> StubRetriever:
    return StubRetriever([make_passage(1), make_passage(2), make_passage(3)])


@pytest.fixture
def make_client(app: FastAPI, retriever: StubRetriever) -> ClientFactory:
    """Return a client whose LLM replies with the given script."""

    def factory(*replies: str | Exception) -> TestClient:
        service = QAService(
            retriever, FakeLLM(list(replies)), top_k=6, min_relevance=0.5, max_tokens=100
        )
        app.dependency_overrides[get_qa_service] = lambda: service
        # False, so a crash shows up as the 500 a real client would get.
        return as_user(app, raise_server_exceptions=False)

    return factory


@pytest.fixture
def harness(app: FastAPI) -> Harness:
    return Harness(app, FakeEmbedder())
