"""Document library, upload and job status endpoints."""

from dataclasses import asdict
from datetime import date
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi import Path as PathParam

from groundwork.api.deps import ContainerDep, UploadServiceDep
from groundwork.api.schemas import (
    CollectionId,
    DocTypeText,
    DocumentResponse,
    ErrorResponse,
    JobResponse,
    UploadResponse,
)
from groundwork.ingestion.loaders import NoTextError, UnsupportedFormatError
from groundwork.ingestion.uploads import FileTooLargeError

router = APIRouter(tags=["documents"])

# Uploads go here unless the uploader names another collection.
UPLOADS_COLLECTION = "uploads"


@router.get(
    "/documents",
    summary="List the document library",
    description="Returns every indexed document, in title order.",
)
async def list_documents(container: ContainerDep) -> list[DocumentResponse]:
    return [
        DocumentResponse(**asdict(document), chunks=chunks)
        for document, chunks in await container.store.documents()
    ]


@router.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document",
    description=(
        "Accepts one file of a type that `GET /api/config` lists, up to its size limit. "
        "Indexing runs in the background: poll the returned job "
        "for its progress. A file already in the library gets a job that is already `indexed`, "
        "and its original tags are kept."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "The file is too large."},
        415: {"model": ErrorResponse, "description": "The file type is not supported."},
        422: {"description": "A field is invalid, or the file has no text (`detail` is a string)."},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="The document to index.")],
    doc_type: Annotated[DocTypeText, Form(description="The kind of document, such as `runbook`.")],
    service: UploadServiceDep,
    background: BackgroundTasks,
    collection: Annotated[
        CollectionId, Form(description="The collection to add it to.")
    ] = UPLOADS_COLLECTION,
    doc_date: Annotated[
        date | None, Form(description="The date the document is about, if known.")
    ] = None,
) -> UploadResponse:
    try:
        job = await service.accept(
            file.filename or "",
            await file.read(),
            collection_id=collection,
            doc_type=doc_type,
            doc_date=doc_date,
            schedule=background.add_task,
        )
    except FileTooLargeError as error:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(error)) from error
    except UnsupportedFormatError as error:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(error)) from error
    except NoTextError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return UploadResponse(job_id=job.job_id, doc_id=job.doc_id)


@router.get(
    "/jobs/{job_id}",
    summary="Check an upload's progress",
    description="Jobs live in memory, so a job from before a server restart returns 404.",
    responses={404: {"model": ErrorResponse, "description": "No such job."}},
)
async def get_job(
    job_id: Annotated[str, PathParam(description="The `job_id` from the upload.")],
    service: UploadServiceDep,
) -> JobResponse:
    job = service.job(job_id)
    if job is None:
        # Jobs live in memory, so this also covers a job from before a restart.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")
    return JobResponse(status=job.status, progress=job.progress, error=job.error, chunks=job.chunks)
