"""Document library, upload and job status endpoints."""

from dataclasses import asdict
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from groundwork.api.deps import CurrentUser, LibraryDep, UploadServiceDep
from groundwork.api.routes_jobs import job_response
from groundwork.api.schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    CollectionId,
    DocTypeText,
    DocumentDetailResponse,
    DocumentPatch,
    DocumentResponse,
    ErrorResponse,
    UploadResponse,
)
from groundwork.domain.models import Document, DocumentChanges
from groundwork.ingestion.loaders import NoTextError, UnsupportedFormatError
from groundwork.ingestion.uploads import FileTooLargeError, ForbiddenGroupsError
from groundwork.library import (
    DocumentNotFoundError,
    LibraryQuery,
    ProtectedCollectionError,
    Sort,
)
from groundwork.retrieval.store import DuplicateDocumentError

router = APIRouter(tags=["documents"])

# Uploads go here unless the uploader names another collection.
UPLOADS_COLLECTION = "uploads"


DocId = Annotated[str, PathParam(description="The document's `doc_id`.")]
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "No such document this user may see."}
}


def document_response(document: Document, chunks: int) -> DocumentResponse:
    return DocumentResponse(**asdict(document), chunks=chunks)


@router.get(
    "/documents",
    summary="List the document library",
    description=(
        "The indexed documents the user may read, filtered, sorted and paged. The "
        "`X-Total-Count` header gives how many match before paging."
    ),
)
async def list_documents(
    library: LibraryDep,
    user: CurrentUser,
    response: Response,
    collection: Annotated[list[str] | None, Query(description="Only these collections.")] = None,
    doc_type: Annotated[list[str] | None, Query(description="Only these types.")] = None,
    mine: Annotated[bool, Query(description="Only the user's own uploads.")] = False,
    q: Annotated[str, Query(max_length=200, description="Text in the title or file name.")] = "",
    sort: Annotated[Sort, Query(description="A field; `-` first for descending.")] = "title",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 500,
) -> list[DocumentResponse]:
    query = LibraryQuery(
        collections=tuple(collection or ()),
        doc_types=tuple(doc_type or ()),
        mine=mine,
        text=q.strip(),
        sort=sort,
        offset=offset,
        limit=limit,
    )
    page, total = await library.page(user, query)
    response.headers["X-Total-Count"] = str(total)
    return [document_response(document, chunks) for document, chunks in page]


@router.get(
    "/documents/{doc_id}",
    summary="A document's details",
    description="Its tags, the indexes holding it, its ingestion jobs and what the user may do.",
    responses=NOT_FOUND,
)
async def get_document(
    doc_id: DocId, library: LibraryDep, user: CurrentUser
) -> DocumentDetailResponse:
    try:
        detail = await library.detail(user, doc_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.") from error
    return DocumentDetailResponse(
        **document_response(detail.document, detail.chunks).model_dump(),
        indexes=detail.indexes,
        jobs=[job_response(job, user) for job in detail.jobs],
        can_edit=detail.can_edit,
        can_delete=detail.can_delete,
        downloadable=detail.document.storage_path is not None,
    )


@router.get(
    "/documents/{doc_id}/file",
    summary="Download the original file",
    description="The file as it was uploaded or committed, under its original name.",
    response_class=FileResponse,
    responses={**NOT_FOUND, 200: {"description": "The file as it was uploaded."}},
)
async def download_document(doc_id: DocId, library: LibraryDep, user: CurrentUser) -> FileResponse:
    try:
        path, document = await library.file(user, doc_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such file.") from error
    return FileResponse(path, filename=document.filename)


@router.patch(
    "/documents/{doc_id}",
    summary="Edit a document's tags and groups",
    description=(
        "For its owner or an admin. Groups are limited to the editor's own. The change applies "
        "to every search from the next one on, in every API worker."
    ),
    responses={
        **NOT_FOUND,
        403: {"model": ErrorResponse, "description": "A group the editor is not in."},
        409: {"model": ErrorResponse, "description": "The same file is stored for those groups."},
    },
)
async def update_document(
    doc_id: DocId, patch: DocumentPatch, library: LibraryDep, user: CurrentUser
) -> DocumentResponse:
    changes = DocumentChanges(
        title=patch.title,
        doc_type=patch.doc_type,
        doc_date=patch.doc_date,
        access_groups=None if patch.access_groups is None else tuple(patch.access_groups),
    )
    try:
        document = await library.update(user, doc_id, changes)
    except DocumentNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.") from error
    except ForbiddenGroupsError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    except DuplicateDocumentError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    return document_response(document, await library.chunk_count(doc_id))


DELETE_ERRORS: dict[int | str, dict[str, Any]] = {
    403: {"model": ErrorResponse, "description": "A protected collection, and not an admin."}
}


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description=(
        "For its owner or an admin; anyone else gets 404. Removes it from every index, cancels "
        "its unfinished ingestion and deletes the stored file."
    ),
    responses={**NOT_FOUND, **DELETE_ERRORS},
)
async def delete_document(doc_id: DocId, library: LibraryDep, user: CurrentUser) -> None:
    try:
        deleted, _ = await library.delete(user, [doc_id])
    except ProtectedCollectionError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")


@router.post(
    "/documents/delete",
    summary="Delete several documents",
    description=(
        "Deletes those the user may and lists the rest as not found. If any is in a protected "
        "collection and the user is not an admin, nothing is deleted."
    ),
    responses=DELETE_ERRORS,
)
async def delete_documents(
    request: BulkDeleteRequest, library: LibraryDep, user: CurrentUser
) -> BulkDeleteResponse:
    try:
        deleted, not_found = await library.delete(user, request.doc_ids)
    except ProtectedCollectionError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    return BulkDeleteResponse(deleted=deleted, not_found=not_found)


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
