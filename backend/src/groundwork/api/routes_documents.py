"""Document library, upload and job status endpoints."""

from dataclasses import asdict
from datetime import date
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from groundwork.api.deps import ContainerDep, CurrentUser, UploadServiceDep
from groundwork.api.schemas import (
    CollectionId,
    DocTypeText,
    DocumentResponse,
    ErrorResponse,
    UploadResponse,
)
from groundwork.domain.models import Access
from groundwork.ingestion.loaders import NoTextError, UnsupportedFormatError
from groundwork.ingestion.uploads import FileTooLargeError, ForbiddenGroupsError

router = APIRouter(tags=["documents"])

# Uploads go here unless the uploader names another collection.
UPLOADS_COLLECTION = "uploads"


@router.get(
    "/documents",
    summary="List the document library",
    description="Returns every indexed document the user may read, in title order.",
)
async def list_documents(container: ContainerDep, user: CurrentUser) -> list[DocumentResponse]:
    documents = await container.store.documents(access=Access.of(user.groups))
    return [DocumentResponse(**asdict(document), chunks=chunks) for document, chunks in documents]


@router.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document",
    description=(
        "Accepts one file of a type that `GET /api/config` lists, up to its size limit. "
        "The ingestion worker indexes it: poll the returned job for its progress. "
        "The uploader owns the document; by default only they can read it. "
        "A file this uploader already stored, in the same collection for the same groups, gets "
        "a job that is already `indexed`, and its original tags are kept."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "The file is too large."},
        415: {"model": ErrorResponse, "description": "The file type is not supported."},
        403: {"model": ErrorResponse, "description": "A group the uploader is not in."},
        422: {"description": "A field is invalid, or the file has no text (`detail` is a string)."},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="The document to index.")],
    doc_type: Annotated[DocTypeText, Form(description="The kind of document, such as `runbook`.")],
    service: UploadServiceDep,
    user: CurrentUser,
    collection: Annotated[
        CollectionId, Form(description="The collection to add it to.")
    ] = UPLOADS_COLLECTION,
    doc_date: Annotated[
        date | None, Form(description="The date the document is about, if known.")
    ] = None,
    access_groups: Annotated[
        list[str] | None,
        Form(
            description=(
                "Groups that may read the document, from the uploader's own (`GET /api/me`). "
                "Repeat the field for several. None: only the uploader."
            )
        ),
    ] = None,
) -> UploadResponse:
    try:
        job = await service.accept(
            file.filename or "",
            await file.read(),
            owner=user,
            collection_id=collection,
            doc_type=doc_type,
            doc_date=doc_date,
            access_groups=access_groups or (),
        )
    except ForbiddenGroupsError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    except FileTooLargeError as error:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(error)) from error
    except UnsupportedFormatError as error:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(error)) from error
    except NoTextError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return UploadResponse(job_id=job.job_id, doc_id=job.doc_id)
