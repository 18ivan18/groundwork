"""The JobQueue contract (#42), run against the in-memory queue and against Postgres."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from conftest import empty_database

from groundwork.ingestion.jobs import INGEST_DOCUMENT, STALE_ERROR, JobQueue, MemoryJobQueue
from groundwork.ingestion.pg_jobs import PgJobQueue

DOC_IDS = (1, 3, 7)


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
async def queue(request: pytest.FixtureRequest) -> AsyncIterator[JobQueue]:
    if request.param == "memory":
        yield MemoryJobQueue()
        return
    async with empty_database(request.getfixturevalue("database_url")) as pool:
        async with pool.connection() as connection:
            await connection.execute("TRUNCATE jobs")
            # A job's document must exist: these are the ids the tests use.
            await connection.execute("INSERT INTO collections (id, name) VALUES ('c', 'C')")
            for doc_id in DOC_IDS:
                await connection.execute(
                    "INSERT INTO documents (id, content_hash, collection_id, title, filename, "
                    "doc_type, access_groups) VALUES (%s, %s, 'c', 't', 'f', 'note', '{everyone}')",
                    (doc_id, f"hash-{doc_id}"),
                )
        yield PgJobQueue(pool)


async def enqueue(queue: JobQueue, name: str = "a", owner: str | None = "alice") -> str:
    job = await queue.enqueue(INGEST_DOCUMENT, {"name": name}, owner_id=owner, max_attempts=2)
    return job.job_id


async def test_a_new_job_is_queued(queue: JobQueue) -> None:
    job_id = await enqueue(queue)

    job = await queue.get(job_id)

    assert job is not None
    assert (job.status, job.progress, job.attempts, job.owner_id) == ("queued", 0, 0, "alice")
    assert job.payload == {"name": "a"}


async def test_an_unknown_job_is_none(queue: JobQueue) -> None:
    assert await queue.get("not-a-uuid") is None
    assert await queue.get("00000000-0000-0000-0000-000000000000") is None


async def test_jobs_are_claimed_oldest_first_and_once(queue: JobQueue) -> None:
    first, second = await enqueue(queue, "a"), await enqueue(queue, "b")

    claimed = await queue.claim("w1")
    other = await queue.claim("w2")

    assert claimed is not None and other is not None
    assert (claimed.job_id, other.job_id) == (first, second)
    assert (claimed.status, claimed.attempts) == ("parsing", 1)
    assert await queue.claim("w3") is None


async def test_concurrent_claims_never_share_a_job(queue: JobQueue) -> None:
    await enqueue(queue)

    results = await asyncio.gather(*(queue.claim(f"w{i}") for i in range(5)))

    assert sum(job is not None for job in results) == 1


async def test_a_heartbeat_records_progress(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    await queue.claim("w1")

    assert await queue.heartbeat(job_id, "w1", "embedding", 55)

    job = await queue.get(job_id)
    assert job is not None and (job.status, job.progress) == ("embedding", 55)


async def test_success_records_the_document(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    await queue.claim("w1")

    await queue.succeed(job_id, "w1", doc_id="7", chunks=12)

    job = await queue.get(job_id)
    assert job is not None
    assert (job.status, job.progress, job.doc_id, job.chunks) == ("indexed", 100, "7", 12)


async def test_a_failure_is_retried_then_dead(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    await queue.claim("w1")

    first = await queue.fail(job_id, "w1", "boom", retry_in=0)
    await queue.claim("w1")
    second = await queue.fail(job_id, "w1", "boom again", retry_in=0)

    assert first is not None and (first.status, first.error) == ("queued", "boom")
    assert second is not None and (second.status, second.attempts) == ("dead", 2)
    assert await queue.claim("w1") is None


async def test_a_retry_waits_for_its_backoff(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    await queue.claim("w1")

    await queue.fail(job_id, "w1", "boom", retry_in=3600)

    assert await queue.claim("w1") is None


async def test_a_permanent_failure_is_not_retried(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    await queue.claim("w1")

    failed = await queue.fail(job_id, "w1", "No text.", retry_in=None)

    assert failed is not None and (failed.status, failed.attempts) == ("failed", 1)
    assert await queue.claim("w1") is None


async def test_a_job_whose_worker_stopped_is_taken_back(queue: JobQueue) -> None:
    """The worker claimed it and died: its heartbeat stops, another worker finishes the job."""
    job_id = await enqueue(queue)
    await queue.claim("dead-worker")

    assert await queue.reap_stale(stale_after=0) == 1
    claimed = await queue.claim("w2")

    assert claimed is not None and (claimed.job_id, claimed.attempts) == (job_id, 2)
    assert claimed.error == STALE_ERROR
    # The first worker, if it was only slow, can no longer touch the job.
    assert not await queue.heartbeat(job_id, "dead-worker", "embedding", 50)
    await queue.succeed(job_id, "dead-worker", doc_id="1", chunks=1)
    job = await queue.get(job_id)
    assert job is not None and job.status == "parsing"


async def test_a_stale_job_out_of_attempts_is_dead(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    for worker in ("w1", "w2"):
        await queue.claim(worker)
        await queue.reap_stale(stale_after=0)

    job = await queue.get(job_id)
    assert job is not None and job.status == "dead"


async def test_a_running_job_is_not_reaped_while_its_heartbeat_is_fresh(queue: JobQueue) -> None:
    await enqueue(queue)
    await queue.claim("w1")

    assert await queue.reap_stale(stale_after=3600) == 0


async def test_cancelling_a_queued_job_is_immediate(queue: JobQueue) -> None:
    job_id = await enqueue(queue)

    cancelled = await queue.cancel(job_id)

    assert cancelled is not None and cancelled.status == "cancelled"
    assert await queue.claim("w1") is None


async def test_cancelling_a_running_job_asks_its_worker_to_stop(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    await queue.claim("w1")

    asked = await queue.cancel(job_id)
    keep_going = await queue.heartbeat(job_id, "w1", "embedding", 40)
    await queue.mark_cancelled(job_id, "w1")

    assert asked is not None and asked.cancel_requested and asked.status == "parsing"
    assert not keep_going
    job = await queue.get(job_id)
    assert job is not None and job.status == "cancelled"


async def test_a_dead_job_can_be_retried(queue: JobQueue) -> None:
    job_id = await enqueue(queue)
    for _ in range(2):
        await queue.claim("w1")
        await queue.fail(job_id, "w1", "boom", retry_in=0)

    retried = await queue.retry(job_id)

    assert retried is not None and (retried.status, retried.attempts) == ("queued", 0)
    assert retried.error is None
    assert await queue.claim("w1") is not None


async def test_only_failed_or_dead_jobs_can_be_retried(queue: JobQueue) -> None:
    job_id = await enqueue(queue)

    assert await queue.retry(job_id) is None


async def test_a_done_job_is_recorded_without_work(queue: JobQueue) -> None:
    job = await queue.record_done(INGEST_DOCUMENT, owner_id="alice", doc_id="3", chunks=4)

    assert (job.status, job.progress, job.doc_id, job.chunks) == ("indexed", 100, "3", 4)
    assert await queue.claim("w1") is None


async def test_jobs_are_listed_newest_first_by_owner_and_status(queue: JobQueue) -> None:
    mine = await enqueue(queue, "a", "alice")
    await enqueue(queue, "b", "hana")
    done = await queue.record_done(INGEST_DOCUMENT, owner_id="alice", doc_id="3", chunks=1)

    alices = await queue.list(owner_id="alice")
    everyones = await queue.list()
    queued = await queue.list(owner_id="alice", statuses=["queued"])
    for_document = await queue.list(doc_id="3")

    assert [job.job_id for job in alices] == [done.job_id, mine]
    assert len(everyones) == 3
    assert [job.job_id for job in queued] == [mine]
    assert [job.job_id for job in for_document] == [done.job_id]
