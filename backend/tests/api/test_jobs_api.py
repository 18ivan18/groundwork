"""The Jobs API (#46): own jobs only, everyone's for an admin, retry and cancel by status."""

import asyncio
from datetime import date

import pytest
from api_helpers import as_user
from fastapi import FastAPI

from groundwork.container import Container
from groundwork.domain.models import DocumentMeta
from groundwork.ingestion.jobs import INGEST_DOCUMENT, JobQueue
from groundwork.ingestion.worker import ingest_payload


def queue_of(app: FastAPI) -> JobQueue:
    container: Container = app.state.container
    return container.jobs


def enqueue(app: FastAPI, owner: str, filename: str = "notes.txt") -> str:
    meta = DocumentMeta(
        title=filename,
        collection_id="uploads",
        doc_type="notes",
        doc_date=date(2025, 1, 1),
        filename=filename,
        access_groups=(f"private:{owner}",),
        owner_id=owner,
    )
    job = asyncio.run(
        queue_of(app).enqueue(
            INGEST_DOCUMENT, ingest_payload(f"uploads/{filename}", "abc", meta), owner_id=owner
        )
    )
    return job.job_id


def kill(app: FastAPI, job_id: str) -> None:
    """Fail the job until it is dead. It must be the only queued job: claims take the oldest."""

    async def run() -> None:
        queue = queue_of(app)
        for _ in range(3):
            await queue.claim("w")
            await queue.fail(job_id, "w", "boom", retry_in=0)

    asyncio.run(run())


def test_a_user_lists_only_their_own_jobs(app: FastAPI) -> None:
    mine = enqueue(app, "alice")
    enqueue(app, "hana", "salaries.txt")

    jobs = as_user(app, "alice").get("/api/jobs").json()

    assert [job["job_id"] for job in jobs] == [mine]
    assert jobs[0]["filename"] == "notes.txt"
    assert jobs[0]["status"] == "queued"


def test_an_admin_lists_every_job_without_others_files(app: FastAPI) -> None:
    enqueue(app, "alice")
    enqueue(app, "hana", "salaries.txt")
    done = asyncio.run(
        queue_of(app).record_done(INGEST_DOCUMENT, owner_id="hana", doc_id="7", chunks=3)
    )

    jobs = as_user(app, "admin").get("/api/jobs").json()

    assert {job["owner_id"] for job in jobs} == {"alice", "hana"}
    assert all(job["filename"] is None and job["doc_id"] is None for job in jobs)
    [hanas_done] = [job for job in jobs if job["job_id"] == done.job_id]
    assert hanas_done["status"] == "indexed"


def test_jobs_filter_by_status(app: FastAPI) -> None:
    dead = enqueue(app, "alice", "other.txt")
    kill(app, dead)
    queued = enqueue(app, "alice")

    response = as_user(app).get("/api/jobs", params={"status": ["dead", "failed"]})

    assert [job["job_id"] for job in response.json()] == [dead]
    assert queued not in {job["job_id"] for job in response.json()}


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", ""), ("post", "/retry"), ("post", "/cancel")],
)
def test_another_users_job_is_not_found(app: FastAPI, method: str, path: str) -> None:
    job_id = enqueue(app, "hana")

    response = as_user(app, "alice").request(method, f"/api/jobs/{job_id}{path}")

    assert response.status_code == 404
    assert response.json() == {"detail": "No such job."}


def test_an_unknown_job_is_not_found(app: FastAPI) -> None:
    assert as_user(app).get("/api/jobs/not-a-job").status_code == 404


def test_a_dead_job_is_retried(app: FastAPI) -> None:
    job_id = enqueue(app, "alice")
    kill(app, job_id)

    response = as_user(app).post(f"/api/jobs/{job_id}/retry")

    assert response.status_code == 200
    assert (response.json()["status"], response.json()["attempts"]) == ("queued", 0)


def test_a_queued_job_cannot_be_retried(app: FastAPI) -> None:
    job_id = enqueue(app, "alice")

    response = as_user(app).post(f"/api/jobs/{job_id}/retry")

    assert response.status_code == 409


def test_a_queued_job_is_cancelled(app: FastAPI) -> None:
    job_id = enqueue(app, "alice")

    response = as_user(app).post(f"/api/jobs/{job_id}/cancel")

    assert (response.status_code, response.json()["status"]) == (200, "cancelled")


def test_a_running_job_is_asked_to_stop(app: FastAPI) -> None:
    job_id = enqueue(app, "alice")
    asyncio.run(queue_of(app).claim("w"))

    body = as_user(app).post(f"/api/jobs/{job_id}/cancel").json()

    assert (body["status"], body["cancel_requested"]) == ("parsing", True)


def test_a_finished_job_cannot_be_cancelled(app: FastAPI) -> None:
    job_id = enqueue(app, "alice")
    kill(app, job_id)

    assert as_user(app).post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_an_admin_can_cancel_anyones_job(app: FastAPI) -> None:
    job_id = enqueue(app, "hana")

    body = as_user(app, "admin").post(f"/api/jobs/{job_id}/cancel").json()

    assert body["status"] == "cancelled"
    assert body["filename"] is None
