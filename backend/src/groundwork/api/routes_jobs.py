"""Ingestion jobs: list, status, retry and cancel (#46).

A user sees their own jobs; an admin sees everyone's. A job the user may not see is 404, as if it
did not exist. Another user's job never shows the file or the document it stored: an admin can
manage the queue without learning what a private upload was.
"""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi import Path as PathParam

from groundwork.api.deps import ContainerDep, CurrentUser
from groundwork.api.schemas import ErrorResponse, JobResponse
from groundwork.domain.users import User
from groundwork.ingestion.jobs import FINISHED, RETRYABLE, Job, JobQueue, JobStatus

router = APIRouter(tags=["jobs"])

JobId = Annotated[str, PathParam(description="The `job_id` from an upload or the job list.")]
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "No such job of this user's."}
}


def job_response(job: Job, user: User) -> JobResponse:
    own = job.owner_id == user.user_id
    return JobResponse(
        job_id=job.job_id,
        kind=job.kind,
        owner_id=job.owner_id,
        filename=job.payload.get("meta", {}).get("filename") if own else None,
        doc_id=job.doc_id if own else None,
        status=job.status,
        progress=job.progress,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error=job.error,
        chunks=job.chunks,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


async def visible_job(jobs: JobQueue, job_id: str, user: User) -> Job:
    job = await jobs.get(job_id)
    if job is None or not (user.is_admin or job.owner_id == user.user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")
    return job


@router.get(
    "/jobs",
    summary="List ingestion jobs",
    description="The user's jobs, newest first; every user's for an admin.",
)
async def list_jobs(
    container: ContainerDep,
    user: CurrentUser,
    job_status: Annotated[
        list[JobStatus] | None, Query(alias="status", description="Only jobs in these states.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[JobResponse]:
    jobs = await container.jobs.list(
        owner_id=None if user.is_admin else user.user_id,
        statuses=job_status or (),
        limit=limit,
    )
    return [job_response(job, user) for job in jobs]


@router.get(
    "/jobs/{job_id}",
    summary="Check a job's progress",
    description="Only the user who started the job, or an admin, sees it.",
    responses=NOT_FOUND,
)
async def get_job(job_id: JobId, container: ContainerDep, user: CurrentUser) -> JobResponse:
    return job_response(await visible_job(container.jobs, job_id, user), user)


@router.post(
    "/jobs/{job_id}/retry",
    summary="Retry a failed or dead job",
    description="Queues it again with its attempts reset.",
    responses={
        **NOT_FOUND,
        409: {"model": ErrorResponse, "description": "The job is not failed or dead."},
    },
)
async def retry_job(job_id: JobId, container: ContainerDep, user: CurrentUser) -> JobResponse:
    job = await visible_job(container.jobs, job_id, user)
    if job.status not in RETRYABLE or (retried := await container.jobs.retry(job_id)) is None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A {job.status} job cannot be retried.")
    return job_response(retried, user)


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel a queued or running job",
    description=(
        "A queued job is cancelled at once. A running one stops before its next embedding "
        "batch, and stores nothing: `cancel_requested` is true until it has."
    ),
    responses={
        **NOT_FOUND,
        409: {"model": ErrorResponse, "description": "The job has already finished."},
    },
)
async def cancel_job(job_id: JobId, container: ContainerDep, user: CurrentUser) -> JobResponse:
    job = await visible_job(container.jobs, job_id, user)
    if job.status in FINISHED or (cancelled := await container.jobs.cancel(job_id)) is None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A {job.status} job cannot be cancelled.")
    return job_response(cancelled, user)
