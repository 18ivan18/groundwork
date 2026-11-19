"""The job queue (D21): work the API hands to the worker service, persisted so it survives
restarts of either.

A job is claimed by one worker at a time. While it runs, the worker's heartbeat moves
`locked_at`; a job whose heartbeat stops is taken back and retried, up to `max_attempts`, and
then is `dead`. A failure the job reports as permanent (a file with no text) is `failed` at once.
Retrying a failed or dead job, or cancelling a queued or running one, is the user's choice.

`PgJobQueue` (ingestion/pg_jobs.py) is the queue the app runs on; `MemoryJobQueue` is for tests
that need no database. tests/unit/test_jobs.py runs the same contract against both.
"""

import asyncio
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

JobStatus = Literal["queued", "parsing", "embedding", "indexed", "failed", "dead", "cancelled"]
RUNNING: tuple[JobStatus, ...] = ("parsing", "embedding")
FINISHED: tuple[JobStatus, ...] = ("indexed", "failed", "dead", "cancelled")
RETRYABLE: tuple[JobStatus, ...] = ("failed", "dead")

INGEST_DOCUMENT = "ingest_document"
STALE_ERROR = "The worker stopped responding."


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    # Who started it: only they, and admins, see it. None for the corpus ingest.
    owner_id: str | None
    payload: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime
    status: JobStatus = "queued"
    progress: int = 0
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = None
    # The document the job stored: known at once for a file already indexed, else once it is.
    doc_id: str | None = None
    chunks: int | None = None
    cancel_requested: bool = False


class JobQueue(Protocol):
    async def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        owner_id: str | None,
        max_attempts: int = 3,
    ) -> Job: ...

    async def record_done(
        self, kind: str, *, owner_id: str | None, doc_id: str, chunks: int
    ) -> Job:
        """A job that is already `indexed`: the file was stored before."""
        ...

    async def get(self, job_id: str) -> Job | None: ...

    async def list(
        self,
        *,
        owner_id: str | None = None,
        statuses: Sequence[JobStatus] = (),
        doc_id: str | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """Newest first. `owner_id` None lists everyone's."""
        ...

    async def reap_stale(self, stale_after: float) -> int:
        """Take back running jobs whose heartbeat is older than `stale_after` seconds: queued
        again, or dead once out of attempts. Returns how many."""
        ...

    async def claim(self, worker_id: str) -> Job | None:
        """The oldest queued job that is due, now `parsing` and locked to this worker, with one
        more attempt counted. Two workers never claim the same job."""
        ...

    async def heartbeat(
        self, job_id: str, worker_id: str, status: JobStatus, progress: int
    ) -> bool:
        """Record progress. False when the worker should stop: the job was cancelled, or another
        worker took it back."""
        ...

    async def succeed(self, job_id: str, worker_id: str, *, doc_id: str, chunks: int) -> None: ...

    async def fail(
        self, job_id: str, worker_id: str, error: str, *, retry_in: float | None
    ) -> Job | None:
        """A failed attempt. With `retry_in` and attempts left, queued again after that many
        seconds; with attempts used up, dead; without `retry_in`, failed for good."""
        ...

    async def mark_cancelled(self, job_id: str, worker_id: str) -> None: ...

    async def cancel(self, job_id: str) -> Job | None:
        """A queued job is cancelled at once; a running one is asked to stop."""
        ...

    async def retry(self, job_id: str) -> Job | None:
        """A failed or dead job, queued again with its attempts reset. None for any other."""
        ...


@dataclass
class _Entry:
    job: Job
    run_after: datetime
    sequence: int
    locked_by: str | None = None
    locked_at: datetime | None = None


@dataclass
class MemoryJobQueue:
    """The queue in a dict, for tests. `clock` stands in for the database's now()."""

    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _new(
        self, kind: str, payload: Mapping[str, Any], owner_id: str | None, **fields: Any
    ) -> Job:
        now = self.clock()
        job = Job(
            job_id=str(uuid.uuid4()),
            kind=kind,
            owner_id=owner_id,
            payload=dict(payload),
            created_at=now,
            updated_at=now,
            **fields,
        )
        self._entries[job.job_id] = _Entry(job, now, sequence=len(self._entries))
        return job

    def _update(self, job_id: str, **fields: Any) -> Job:
        entry = self._entries[job_id]
        entry.job = replace(entry.job, updated_at=self.clock(), **fields)
        return entry.job

    def _owns(self, job_id: str, worker_id: str) -> bool:
        entry = self._entries.get(job_id)
        return entry is not None and entry.locked_by == worker_id

    async def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        owner_id: str | None,
        max_attempts: int = 3,
    ) -> Job:
        async with self._lock:
            return self._new(kind, payload, owner_id, max_attempts=max_attempts)

    async def record_done(
        self, kind: str, *, owner_id: str | None, doc_id: str, chunks: int
    ) -> Job:
        async with self._lock:
            return self._new(
                kind, {}, owner_id, status="indexed", progress=100, doc_id=doc_id, chunks=chunks
            )

    async def get(self, job_id: str) -> Job | None:
        entry = self._entries.get(job_id)
        return entry.job if entry else None

    async def list(
        self,
        *,
        owner_id: str | None = None,
        statuses: Sequence[JobStatus] = (),
        doc_id: str | None = None,
        limit: int = 100,
    ) -> list[Job]:
        entries = sorted(self._entries.values(), key=lambda e: e.sequence, reverse=True)
        return [
            e.job
            for e in entries
            if (owner_id is None or e.job.owner_id == owner_id)
            and (not statuses or e.job.status in statuses)
            and (doc_id is None or e.job.doc_id == doc_id)
        ][:limit]

    async def reap_stale(self, stale_after: float) -> int:
        async with self._lock:
            cutoff = self.clock() - timedelta(seconds=stale_after)
            stale = [
                e
                for e in self._entries.values()
                if e.job.status in RUNNING and e.locked_at is not None and e.locked_at <= cutoff
            ]
            for entry in stale:
                spent = entry.job.attempts >= entry.job.max_attempts
                # run_after is kept, so the job goes back to its place in the queue, not the end.
                entry.locked_by = entry.locked_at = None
                self._update(
                    entry.job.job_id, status="dead" if spent else "queued", error=STALE_ERROR
                )
            return len(stale)

    async def claim(self, worker_id: str) -> Job | None:
        async with self._lock:
            now = self.clock()
            due = [
                e for e in self._entries.values() if e.job.status == "queued" and e.run_after <= now
            ]
            if not due:
                return None
            entry = min(due, key=lambda e: (e.run_after, e.sequence))
            entry.locked_by, entry.locked_at = worker_id, now
            return self._update(
                entry.job.job_id, status="parsing", progress=0, attempts=entry.job.attempts + 1
            )

    async def heartbeat(
        self, job_id: str, worker_id: str, status: JobStatus, progress: int
    ) -> bool:
        async with self._lock:
            if not self._owns(job_id, worker_id):
                return False
            self._entries[job_id].locked_at = self.clock()
            job = self._update(job_id, status=status, progress=progress)
            return not job.cancel_requested

    async def succeed(self, job_id: str, worker_id: str, *, doc_id: str, chunks: int) -> None:
        async with self._lock:
            if self._owns(job_id, worker_id):
                self._entries[job_id].locked_by = None
                self._update(
                    job_id, status="indexed", progress=100, doc_id=doc_id, chunks=chunks, error=None
                )

    async def fail(
        self, job_id: str, worker_id: str, error: str, *, retry_in: float | None
    ) -> Job | None:
        async with self._lock:
            if not self._owns(job_id, worker_id):
                return None
            entry = self._entries[job_id]
            entry.locked_by = None
            if retry_in is None:
                return self._update(job_id, status="failed", progress=0, error=error)
            if entry.job.attempts >= entry.job.max_attempts:
                return self._update(job_id, status="dead", progress=0, error=error)
            entry.run_after = self.clock() + timedelta(seconds=retry_in)
            return self._update(job_id, status="queued", progress=0, error=error)

    async def mark_cancelled(self, job_id: str, worker_id: str) -> None:
        async with self._lock:
            if self._owns(job_id, worker_id):
                self._entries[job_id].locked_by = None
                self._update(job_id, status="cancelled", progress=0)

    async def cancel(self, job_id: str) -> Job | None:
        async with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return None
            if entry.job.status == "queued":
                return self._update(job_id, status="cancelled")
            if entry.job.status in RUNNING:
                return self._update(job_id, cancel_requested=True)
            return entry.job

    async def retry(self, job_id: str) -> Job | None:
        async with self._lock:
            entry = self._entries.get(job_id)
            if entry is None or entry.job.status not in RETRYABLE:
                return None
            entry.run_after = self.clock()
            return self._update(
                job_id, status="queued", progress=0, attempts=0, error=None, cancel_requested=False
            )
