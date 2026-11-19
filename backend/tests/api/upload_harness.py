"""Drives uploads through the API with the ingestion worker run in-process."""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from api_helpers import as_user
from fastapi import FastAPI
from httpx import Response

from groundwork.api.deps import get_upload_service
from groundwork.container import Container
from groundwork.domain.models import UNRESTRICTED, Access, ScoredPassage, SearchFilters
from groundwork.domain.users import load_users
from groundwork.ingestion.uploads import UploadService
from groundwork.ingestion.worker import IngestWorker
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.hybrid import HybridRetriever, SearchMode
from groundwork.retrieval.sparse import SparseCache

USERS = load_users(Path(__file__).parents[2] / "conf" / "users.yaml")
ALICE = USERS["alice"]
MAX_BYTES = 5000
TEXT = (
    "Anvils are heavy steel blocks that ACME Corp sells to blacksmiths everywhere. " * 8
).encode()
FORM = {"doc_type": "notes"}


class Harness:
    """Drives uploads through the API, then runs the ingestion worker until the queue is empty,
    retrying a failure at once."""

    def __init__(self, app: FastAPI, embedder: Embedder, heartbeat_seconds: float = 5) -> None:
        container: Container = app.state.container
        self.store = container.store
        self.jobs = container.jobs
        data_dir = container.settings.data_dir
        self.uploads_dir = data_dir / "uploads"
        sparse = SparseCache(self.store, data_dir / "bm25", check_seconds=0)
        self.retriever = HybridRetriever(embedder, self.store, sparse, candidates=20, rrf_k=60)
        self.service = UploadService(self.store, self.jobs, data_dir=data_dir, max_bytes=MAX_BYTES)
        self.worker = IngestWorker(
            self.jobs,
            self.store,
            embedder,
            data_dir=data_dir,
            chunk_tokens=40,
            chunk_overlap=5,
            heartbeat_seconds=heartbeat_seconds,
            backoff=lambda _: 0,
        )
        app.dependency_overrides[get_upload_service] = lambda: self.service
        self.app = app
        self.client = as_user(app)

    def search(
        self,
        query: str,
        k: int,
        filters: SearchFilters | None = None,
        mode: SearchMode = "reranked",
        *,
        access: Access = UNRESTRICTED,
    ) -> list[ScoredPassage]:
        return asyncio.run(self.retriever.search(query, k, filters, mode, access=access))

    def count_chunks(self) -> int:
        return asyncio.run(self.store.count_chunks())

    def generation(self) -> int:
        return asyncio.run(self.store.generation())

    def upload(
        self,
        content: bytes = TEXT,
        name: str = "notes.txt",
        *,
        user: str = "alice",
        groups: Sequence[str] = (),
        **form: str,
    ) -> Response:
        data: dict[str, Any] = {**FORM, **form}
        if groups:
            data["access_groups"] = list(groups)
        response = as_user(self.app, user).post(
            "/api/documents", files={"file": (name, content)}, data=data
        )
        self.work()
        return response

    def work(self) -> None:
        asyncio.run(self.worker.drain())

    def documents(self, user: str) -> list[dict[str, Any]]:
        body: list[dict[str, Any]] = as_user(self.app, user).get("/api/documents").json()
        return body

    def job(self, job_id: str, user: str = "alice") -> dict[str, Any]:
        response = as_user(self.app, user).get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body
