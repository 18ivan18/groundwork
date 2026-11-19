"""The ingestion worker (#42), on the in-memory queue and store and on Postgres."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import empty_database

from groundwork.domain.models import UNRESTRICTED, DocumentMeta, content_hash
from groundwork.ingestion.jobs import INGEST_DOCUMENT, STALE_ERROR, JobQueue, MemoryJobQueue
from groundwork.ingestion.pg_jobs import PgJobQueue
from groundwork.ingestion.worker import (
    MISSING_FILE_ERROR,
    TOO_SHORT_ERROR,
    IngestRequest,
    IngestWorker,
    ingest_payload,
)
from groundwork.retrieval.embedder import Embedder, FakeEmbedder
from groundwork.retrieval.indexes import IndexSpec, ensure_index
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.store import ChunkStore

TEXT = "Anvils are heavy steel blocks that ACME Corp sells to blacksmiths everywhere. " * 8


def meta(name: str = "notes.txt") -> DocumentMeta:
    return DocumentMeta(
        title=name,
        collection_id="uploads",
        doc_type="notes",
        doc_date=date(2025, 6, 30),
        filename=name,
        metadata={"team": "ops"},
        access_groups=("everyone", "hr"),
    )


@dataclass
class Setup:
    store: ChunkStore
    queue: JobQueue
    data_dir: Path

    def worker(self, embedder: Embedder | None = None, **options: object) -> IngestWorker:
        return IngestWorker(
            self.queue,
            self.store,
            embedder or FakeEmbedder(),
            data_dir=self.data_dir,
            chunk_tokens=40,
            chunk_overlap=5,
            backoff=lambda _: 0,
            **options,  # type: ignore[arg-type]
        )

    async def enqueue(self, text: str = TEXT, name: str = "notes.txt") -> str:
        raw = text.encode()
        digest = content_hash(raw)
        (self.data_dir / f"{digest}.txt").write_bytes(raw)
        job = await self.queue.enqueue(
            INGEST_DOCUMENT, ingest_payload(f"{digest}.txt", digest, meta(name)), owner_id=None
        )
        return job.job_id


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
async def setup(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Setup]:
    dimension = FakeEmbedder().dimension
    if request.param == "memory":
        yield Setup(MemoryStore(memory_index(dimension)), MemoryJobQueue(), tmp_path)
        return
    async with empty_database(request.getfixturevalue("database_url")) as pool:
        async with pool.connection() as connection:
            await connection.execute("TRUNCATE jobs")
        spec = IndexSpec(name="test", chunker="window", embedder_model="fake", dimension=dimension)
        store = PgVectorStore(pool, await ensure_index(pool, spec))
        yield Setup(store, PgJobQueue(pool), tmp_path)


def test_a_payload_keeps_every_field() -> None:
    original = meta()

    request = IngestRequest.from_payload(ingest_payload("a/b.txt", "abc", original))

    assert (request.path, request.digest, request.meta) == ("a/b.txt", "abc", original)


async def test_a_job_is_indexed_and_published(setup: Setup) -> None:
    job_id = await setup.enqueue()

    await setup.worker().drain()

    job = await setup.queue.get(job_id)
    assert job is not None and job.status == "indexed" and job.doc_id is not None
    assert job.chunks == await setup.store.count_chunks() > 0
    [(document, _)] = await setup.store.documents(access=UNRESTRICTED)
    assert (document.doc_date, dict(document.metadata)) == (date(2025, 6, 30), {"team": "ops"})
    assert document.access_groups == ("everyone", "hr")
    assert await setup.store.generation() == 1


async def test_the_generation_moves_once_for_a_batch(setup: Setup) -> None:
    for i in range(3):
        await setup.enqueue(f"{TEXT} Batch {i}.", f"notes-{i}.txt")

    await setup.worker().drain()

    assert len(await setup.store.documents(access=UNRESTRICTED)) == 3
    assert await setup.store.generation() == 1


async def test_a_busy_worker_still_publishes(setup: Setup) -> None:
    for i in range(3):
        await setup.enqueue(f"{TEXT} Batch {i}.", f"notes-{i}.txt")
    worker = setup.worker(publish_seconds=0)

    assert await worker.run_once()

    assert await setup.store.generation() == 1


async def test_a_job_survives_its_worker_dying(setup: Setup) -> None:
    """A worker claimed the job and died mid-way, say in an API or worker restart: once its
    heartbeat is stale, another worker takes the job back and finishes it."""
    job_id = await setup.enqueue()
    await setup.queue.claim("died-mid-job")

    await setup.worker(stale_after=0).drain()

    job = await setup.queue.get(job_id)
    assert job is not None and (job.status, job.attempts) == ("indexed", 2)
    assert job.error is None
    assert await setup.store.count_chunks() > 0


async def test_a_stale_job_records_why_it_was_retried(setup: Setup) -> None:
    job_id = await setup.enqueue()
    await setup.queue.claim("died-mid-job")

    await setup.queue.reap_stale(stale_after=0)

    job = await setup.queue.get(job_id)
    assert job is not None and (job.status, job.error) == ("queued", STALE_ERROR)


async def test_cancelling_during_embedding_stores_nothing(setup: Setup) -> None:
    job_ids: list[str] = []
    loop = asyncio.get_running_loop()

    class CancellingEmbedder(FakeEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            # The user presses cancel while this batch embeds.
            asyncio.run_coroutine_threadsafe(setup.queue.cancel(job_ids[0]), loop).result()
            return super().embed_documents(texts)

    job_ids.append(await setup.enqueue())

    await setup.worker(CancellingEmbedder()).drain()

    job = await setup.queue.get(job_ids[0])
    assert job is not None and job.status == "cancelled"
    assert await setup.store.count_chunks() == 0
    assert await setup.store.generation() == 0


async def test_a_cancel_that_lands_while_storing_is_undone(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = await setup.enqueue()
    add = setup.store.add

    async def add_then_cancel(*args: Any, **kwargs: Any) -> Any:
        document = await add(*args, **kwargs)
        # The document is deleted in the Library just as the worker stores it.
        await setup.queue.cancel(job_id)
        return document

    monkeypatch.setattr(setup.store, "add", add_then_cancel)

    await setup.worker().drain()

    job = await setup.queue.get(job_id)
    assert job is not None and job.status == "cancelled"
    assert await setup.store.count_chunks() == 0


async def test_a_file_too_short_fails_without_retrying(setup: Setup) -> None:
    job_id = await setup.enqueue("Too short.")

    await setup.worker().drain()

    job = await setup.queue.get(job_id)
    assert job is not None and (job.status, job.attempts, job.error) == (
        "failed",
        1,
        TOO_SHORT_ERROR,
    )


async def test_a_missing_file_fails_without_retrying(setup: Setup) -> None:
    job_id = await setup.enqueue()
    for path in setup.data_dir.iterdir():
        path.unlink()

    await setup.worker().drain()

    job = await setup.queue.get(job_id)
    assert job is not None and (job.status, job.error) == ("failed", MISSING_FILE_ERROR)


async def test_a_file_already_stored_is_not_embedded_again(setup: Setup) -> None:
    first, second = await setup.enqueue(), await setup.enqueue()
    embedded: list[int] = []

    class CountingEmbedder(FakeEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            embedded.append(len(texts))
            return super().embed_documents(texts)

    await setup.worker(CountingEmbedder()).drain()

    jobs = [await setup.queue.get(job_id) for job_id in (first, second)]
    assert [job.status for job in jobs if job] == ["indexed", "indexed"]
    assert jobs[0] is not None and jobs[1] is not None and jobs[0].doc_id == jobs[1].doc_id
    assert sum(embedded) == await setup.store.count_chunks()


async def test_run_stops_when_asked(setup: Setup) -> None:
    await setup.enqueue()
    stop = asyncio.Event()
    worker = setup.worker(poll_seconds=0.01)

    running = asyncio.create_task(worker.run(stop))
    while not await setup.store.count_chunks():
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(running, 5)

    assert await setup.store.generation() == 1
