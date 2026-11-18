"""The ingestion worker: claims `ingest_document` jobs from the queue and indexes their files.

Runs as its own service (`python -m groundwork.ingestion.worker`), so an upload keeps going while
the API restarts, and any number of workers can share the queue. A job names a file under
DATA_DIR and the metadata to store it with; the API (uploads) and scripts/ingest.py (the corpus)
enqueue them.

While a job runs, a heartbeat records its progress every few seconds. The heartbeat is also how a
cancel arrives: the embedding loop checks for it between batches and stops. Storing is
idempotent (the store upserts a document and its chunks), so two workers that index the same
file, or a worker that was taken for dead but was only slow, store it once.

The index's generation, which makes every API worker reload BM25, moves when the worker runs
out of work, or 30 seconds after it stored something, rather than after every file: a corpus
ingest of 200 files rebuilds BM25 a handful of times, not 200.
"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from groundwork.domain.models import DocumentMeta
from groundwork.ingestion.jobs import INGEST_DOCUMENT, Job, JobQueue, JobStatus
from groundwork.ingestion.loaders import NoTextError, loader_for
from groundwork.ingestion.pipeline import prepare_document
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.store import ChunkStore

logger = logging.getLogger(__name__)

# Embedding is the slow step, so its batches drive the progress bar from 20 to 90.
_EMBEDDING_START = 20
_EMBEDDING_END = 90

FAILED_ERROR = "Indexing failed. Please try again."
TOO_SHORT_ERROR = "This file has too little text to index."
MISSING_FILE_ERROR = "The uploaded file is no longer stored."


def ingest_payload(path: str, digest: str, meta: DocumentMeta) -> dict[str, Any]:
    """An `ingest_document` job's payload. `path` is relative to DATA_DIR, so the API and the
    worker may mount it in different places."""
    fields = asdict(meta)
    fields["doc_date"] = meta.doc_date.isoformat() if meta.doc_date else None
    fields["access_groups"] = list(meta.access_groups)
    fields["metadata"] = dict(meta.metadata)
    return {"path": path, "digest": digest, "meta": fields}


@dataclass(frozen=True)
class IngestRequest:
    path: str
    digest: str
    meta: DocumentMeta

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IngestRequest":
        fields = dict(payload["meta"])
        fields["doc_date"] = date.fromisoformat(fields["doc_date"]) if fields["doc_date"] else None
        fields["access_groups"] = tuple(fields["access_groups"])
        return cls(payload["path"], payload["digest"], DocumentMeta(**fields))


class Cancelled(Exception):
    """The job was cancelled, or taken back from this worker."""


class PermanentFailure(Exception):
    """A failure no retry can fix. Its message is shown to the user."""


@dataclass
class _Progress:
    status: JobStatus = "parsing"
    progress: int = 10
    stop: bool = False


def retry_delay(attempt: int) -> float:
    """Seconds before retry number `attempt`: 10, 40, 90..., capped at ten minutes."""
    return float(min(10 * attempt**2, 600))


class IngestWorker:
    def __init__(
        self,
        queue: JobQueue,
        store: ChunkStore,
        embedder: Embedder,
        *,
        data_dir: Path,
        chunk_tokens: int,
        chunk_overlap: int,
        worker_id: str | None = None,
        heartbeat_seconds: float = 5.0,
        stale_after: float = 60.0,
        poll_seconds: float = 1.0,
        backoff: Callable[[int], float] = retry_delay,
        publish_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._queue = queue
        self._store = store
        self._embedder = embedder
        self._data_dir = data_dir
        self._chunk_tokens = chunk_tokens
        self._chunk_overlap = chunk_overlap
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._heartbeat_seconds = heartbeat_seconds
        self._stale_after = stale_after
        self._poll_seconds = poll_seconds
        self._backoff = backoff
        # Documents stored since the generation last moved, and when the first of them was.
        self._unpublished = False
        self._unpublished_since = 0.0
        self._publish_seconds = publish_seconds
        self._clock = clock

    async def run(self, stop: asyncio.Event) -> None:
        """Work until `stop` is set; a job in progress is finished first."""
        logger.info("Worker %s started", self.worker_id)
        while not stop.is_set():
            if not await self.run_once():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), self._poll_seconds)
        await self.publish()

    async def drain(self) -> None:
        """Run every due job, then publish. For tests and one-off runs."""
        while await self.run_once():
            pass

    async def run_once(self) -> bool:
        """Take back stale jobs, then run one job. False when there was none to run, after
        publishing what earlier jobs stored."""
        await self._queue.reap_stale(self._stale_after)
        job = await self._queue.claim(self.worker_id)
        if job is None:
            await self.publish()
            return False
        await self._run(job)
        # A queue that never empties still publishes, at most this long after a store.
        if self._clock() - self._unpublished_since >= self._publish_seconds:
            await self.publish()
        return True

    async def publish(self) -> None:
        if self._unpublished:
            generation = await self._store.bump_generation()
            self._unpublished = False
            logger.info("Generation %s", generation)

    async def _run(self, job: Job) -> None:
        state = _Progress()
        beating = asyncio.create_task(self._beat(job.job_id, state))
        try:
            if job.kind != INGEST_DOCUMENT:
                raise PermanentFailure(f"Unknown job kind {job.kind}.")
            doc_id, chunks = await self._ingest(
                job.job_id, IngestRequest.from_payload(job.payload), state
            )
        except Cancelled:
            await self._queue.mark_cancelled(job.job_id, self.worker_id)
            logger.info("Job %s cancelled", job.job_id)
        except PermanentFailure as error:
            await self._queue.fail(job.job_id, self.worker_id, str(error), retry_in=None)
        except Exception:
            # The exception stays in the log: job errors are shown to users.
            logger.exception("Job %s failed (attempt %s)", job.job_id, job.attempts)
            await self._queue.fail(
                job.job_id, self.worker_id, FAILED_ERROR, retry_in=self._backoff(job.attempts)
            )
        else:
            await self._queue.succeed(job.job_id, self.worker_id, doc_id=doc_id, chunks=chunks)
        finally:
            beating.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beating

    async def _beat(self, job_id: str, state: _Progress) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            if not await self._queue.heartbeat(
                job_id, self.worker_id, state.status, state.progress
            ):
                state.stop = True
                return

    async def _check(self, job_id: str, state: _Progress) -> None:
        """A heartbeat now: between steps, so a cancel is seen before the next one starts."""
        if not await self._queue.heartbeat(job_id, self.worker_id, state.status, state.progress):
            raise Cancelled

    async def _ingest(
        self, job_id: str, request: IngestRequest, state: _Progress
    ) -> tuple[str, int]:
        meta, digest = request.meta, request.digest
        if doc_id := await self._store.find_document(digest, meta):
            return doc_id, await self._store.count_document_chunks(doc_id)
        path = self._data_dir / request.path
        try:
            raw = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as error:
            raise PermanentFailure(MISSING_FILE_ERROR) from error
        try:
            loaded = await asyncio.to_thread(loader_for(meta.filename), raw)
        except NoTextError as error:
            raise PermanentFailure(str(error)) from error

        def report(share: float) -> None:
            # In the embedding thread: the heartbeat sends this, and its answer comes back here.
            if state.stop:
                raise Cancelled
            state.status = "embedding"
            state.progress = _EMBEDDING_START + round((_EMBEDDING_END - _EMBEDDING_START) * share)

        prepared = await asyncio.to_thread(
            prepare_document,
            loaded,
            meta,
            digest,
            self._embedder,
            chunk_tokens=self._chunk_tokens,
            chunk_overlap=self._chunk_overlap,
            on_progress=report,
        )
        if not prepared.chunks:
            raise PermanentFailure(TOO_SHORT_ERROR)
        # Asked now, not at the next heartbeat: a cancelled job must store nothing.
        await self._check(job_id, state)
        document = await self._store.add(meta, digest, prepared.chunks, prepared.embeddings)
        if not self._unpublished:
            self._unpublished, self._unpublished_since = True, self._clock()
        return document.doc_id, len(prepared.chunks)


async def main() -> None:
    # Imported here: the container imports the upload service, which imports this module.
    from groundwork.config import Settings
    from groundwork.container import app_index, build_embedder, chunking
    from groundwork.db.pool import open_pool
    from groundwork.ingestion.pg_jobs import PgJobQueue
    from groundwork.retrieval.pgvector_store import PgVectorStore

    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    embedder = await asyncio.to_thread(build_embedder, settings)
    pool = await open_pool(settings.database_url)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    try:
        store = PgVectorStore(pool, await app_index(pool, settings, embedder.dimension))
        chunk_tokens, chunk_overlap = chunking(store.index, settings)
        worker = IngestWorker(
            PgJobQueue(pool),
            store,
            embedder,
            data_dir=settings.data_dir,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
        )
        await worker.run(stop)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
