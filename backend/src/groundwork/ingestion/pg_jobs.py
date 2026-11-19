"""The job queue on Postgres: `FOR UPDATE SKIP LOCKED` lets any number of workers claim from one
table without claiming the same job twice. See ingestion/jobs.py for the contract."""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from groundwork.db.pool import Pool
from groundwork.ingestion.jobs import RETRYABLE, RUNNING, STALE_ERROR, Job, JobStatus

_COLUMNS = (
    "id, kind, created_by, payload, created_at, updated_at, status, progress, attempts, "
    "max_attempts, error, document_id, result, cancel_requested"
)
# Only the worker holding the lock may finish a job: one whose heartbeat stopped may have been
# taken back and claimed by another.
_LOCKED = "id = %s AND locked_by = %s"


def _job(row: TupleRow) -> Job:
    (
        job_id,
        kind,
        owner_id,
        payload,
        created_at,
        updated_at,
        status,
        progress,
        attempts,
        max_attempts,
        error,
        document_id,
        result,
        cancel_requested,
    ) = row
    return Job(
        job_id=str(job_id),
        kind=kind,
        owner_id=owner_id,
        payload=payload,
        created_at=created_at,
        updated_at=updated_at,
        status=status,
        progress=progress,
        attempts=attempts,
        max_attempts=max_attempts,
        error=error,
        doc_id=None if document_id is None else str(document_id),
        chunks=result.get("chunks"),
        cancel_requested=cancel_requested,
    )


class PgJobQueue:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def _one(self, sql: str, params: Sequence[Any] = ()) -> Job | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(sql, params)
            row = await cursor.fetchone()
        return _job(row) if row else None

    async def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        owner_id: str | None,
        max_attempts: int = 3,
    ) -> Job:
        job = await self._one(
            f"INSERT INTO jobs (id, kind, payload, status, created_by, max_attempts) "
            f"VALUES (%s, %s, %s, 'queued', %s, %s) RETURNING {_COLUMNS}",
            (uuid.uuid4(), kind, Jsonb(dict(payload)), owner_id, max_attempts),
        )
        assert job is not None
        return job

    async def record_done(
        self, kind: str, *, owner_id: str | None, doc_id: str, chunks: int
    ) -> Job:
        job = await self._one(
            f"INSERT INTO jobs (id, kind, status, progress, created_by, document_id, result) "
            f"VALUES (%s, %s, 'indexed', 100, %s, %s, %s) RETURNING {_COLUMNS}",
            (uuid.uuid4(), kind, owner_id, int(doc_id), Jsonb({"chunks": chunks})),
        )
        assert job is not None
        return job

    async def get(self, job_id: str) -> Job | None:
        try:
            key = uuid.UUID(job_id)
        except ValueError:
            return None
        return await self._one(f"SELECT {_COLUMNS} FROM jobs WHERE id = %s", (key,))

    async def list(
        self,
        *,
        owner_id: str | None = None,
        statuses: Sequence[JobStatus] = (),
        doc_id: str | None = None,
        limit: int = 100,
    ) -> list[Job]:
        clauses: list[str] = ["true"]
        params: list[Any] = []
        if owner_id is not None:
            clauses.append("created_by = %s")
            params.append(owner_id)
        if statuses:
            clauses.append("status = ANY(%s)")
            params.append(list(statuses))
        if doc_id is not None:
            clauses.append("document_id = %s")
            params.append(int(doc_id))
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE {' AND '.join(clauses)} "
                f"ORDER BY created_at DESC, id LIMIT {int(limit)}",
                params,
            )
            rows = await cursor.fetchall()
        return [_job(row) for row in rows]

    async def reap_stale(self, stale_after: float) -> int:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE jobs SET "
                "status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END, "
                # run_after is kept, so the job goes back to its place in the queue, not the end.
                "locked_by = NULL, locked_at = NULL, error = %s, "
                "updated_at = now() "
                "WHERE status = ANY(%s) AND locked_at <= now() - make_interval(secs => %s)",
                (STALE_ERROR, list(RUNNING), stale_after),
            )
            return cursor.rowcount

    async def claim(self, worker_id: str) -> Job | None:
        return await self._one(
            f"UPDATE jobs SET status = 'parsing', progress = 0, attempts = attempts + 1, "
            f"locked_by = %s, locked_at = now(), updated_at = now() "
            f"WHERE id = (SELECT id FROM jobs WHERE status = 'queued' AND run_after <= now() "
            f"ORDER BY run_after, created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
            f"RETURNING {_COLUMNS}",
            (worker_id,),
        )

    async def heartbeat(
        self, job_id: str, worker_id: str, status: JobStatus, progress: int
    ) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"UPDATE jobs SET status = %s, progress = %s, locked_at = now(), "
                f"updated_at = now() WHERE {_LOCKED} RETURNING cancel_requested",
                (status, progress, uuid.UUID(job_id), worker_id),
            )
            row = await cursor.fetchone()
        return row is not None and not row[0]

    async def succeed(self, job_id: str, worker_id: str, *, doc_id: str, chunks: int) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                f"UPDATE jobs SET status = 'indexed', progress = 100, document_id = %s, "
                f"result = %s, error = NULL, locked_by = NULL, updated_at = now() "
                f"WHERE {_LOCKED}",
                (int(doc_id), Jsonb({"chunks": chunks}), uuid.UUID(job_id), worker_id),
            )

    async def fail(
        self, job_id: str, worker_id: str, error: str, *, retry_in: float | None
    ) -> Job | None:
        if retry_in is None:
            status_sql, params = "'failed'", [error]
        else:
            status_sql = "CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END"
            params = [error]
        return await self._one(
            f"UPDATE jobs SET status = {status_sql}, progress = 0, error = %s, "
            f"locked_by = NULL, run_after = now() + make_interval(secs => %s), "
            f"updated_at = now() WHERE {_LOCKED} RETURNING {_COLUMNS}",
            (*params, retry_in or 0, uuid.UUID(job_id), worker_id),
        )

    async def mark_cancelled(self, job_id: str, worker_id: str) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                f"UPDATE jobs SET status = 'cancelled', progress = 0, locked_by = NULL, "
                f"updated_at = now() WHERE {_LOCKED}",
                (uuid.UUID(job_id), worker_id),
            )

    async def cancel(self, job_id: str) -> Job | None:
        job = await self.get(job_id)
        if job is None:
            return None
        return await self._one(
            f"UPDATE jobs SET "
            f"status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END, "
            f"cancel_requested = cancel_requested OR status = ANY(%s), updated_at = now() "
            f"WHERE id = %s RETURNING {_COLUMNS}",
            (list(RUNNING), uuid.UUID(job.job_id)),
        )

    async def retry(self, job_id: str) -> Job | None:
        if await self.get(job_id) is None:
            return None
        return await self._one(
            f"UPDATE jobs SET status = 'queued', progress = 0, attempts = 0, error = NULL, "
            f"cancel_requested = false, run_after = now(), updated_at = now() "
            f"WHERE id = %s AND status = ANY(%s) RETURNING {_COLUMNS}",
            (uuid.UUID(job_id), list(RETRYABLE)),
        )
