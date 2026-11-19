"""The document library (#44, #45): who may see, download, edit and delete which document.

- Seeing is reading: a document is visible to the groups it is shared with, and to admins, who
  read everything. Anything a user may not see is "not found", never "forbidden", so its id
  reveals nothing.
- Editing and deleting are for the document's owner, or an admin. Groups can be set only to the
  editor's own (or to the owner alone, when none are given).
- Collections marked protected (the demo corpora) need an admin to delete from.

A change moves the index's generation, so every API worker rebuilds BM25 and stops returning the
old document or its old groups. A delete also cancels the document's unfinished ingestion and
removes the stored file once no document names it.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from groundwork.domain.models import Document, DocumentChanges
from groundwork.domain.users import User
from groundwork.ingestion.jobs import RUNNING, Job, JobQueue
from groundwork.ingestion.uploads import UPLOADS_FOLDER, upload_groups
from groundwork.retrieval.store import ChunkStore

logger = logging.getLogger(__name__)

Sort = Literal["title", "-title", "ingested_at", "-ingested_at", "doc_date", "-doc_date"]


class DocumentNotFoundError(LookupError):
    """No such document, or one this user may not see or change."""


class ProtectedCollectionError(PermissionError):
    """Only an admin may delete from this collection."""


@dataclass(frozen=True)
class LibraryQuery:
    collections: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    # Only the documents this user uploaded.
    mine: bool = False
    # Case-insensitive, anywhere in the title or file name.
    text: str = ""
    sort: Sort = "title"
    offset: int = 0
    limit: int = 500


@dataclass(frozen=True)
class DocumentDetail:
    document: Document
    chunks: int
    indexes: list[str]
    jobs: list[Job]
    can_edit: bool
    can_delete: bool


def _sort_key(sort: Sort, item: tuple[Document, int]) -> tuple[object, ...]:
    document = item[0]
    field = sort.lstrip("-")
    value: object
    if field == "title":
        value = document.title.casefold()
    elif field == "ingested_at":
        value = document.ingested_at
    else:
        # Undated documents sort last either way round.
        value = (document.doc_date is not None, document.doc_date)
    return (value, int(document.doc_id))


class LibraryService:
    def __init__(self, store: ChunkStore, jobs: JobQueue, *, data_dir: Path) -> None:
        self._store = store
        self._jobs = jobs
        self._data_dir = data_dir

    async def page(self, user: User, query: LibraryQuery) -> tuple[list[tuple[Document, int]], int]:
        """One page of the documents the user may see, and how many there are in all.

        Filtered in Python over the user's whole list: the library holds hundreds of documents,
        not millions. Past a few thousand this belongs in SQL."""
        documents = await self._store.documents(access=user.access)
        text = query.text.casefold()
        matching = [
            (document, chunks)
            for document, chunks in documents
            if (not query.collections or document.collection_id in query.collections)
            and (not query.doc_types or document.doc_type in query.doc_types)
            and (not query.mine or document.owner_id == user.user_id)
            and (text in document.title.casefold() or text in document.filename.casefold())
        ]
        matching.sort(
            key=lambda item: _sort_key(query.sort, item), reverse=query.sort.startswith("-")
        )
        return matching[query.offset : query.offset + query.limit], len(matching)

    async def chunk_count(self, doc_id: str) -> int:
        return await self._store.count_document_chunks(doc_id)

    async def _visible(self, user: User, doc_id: str) -> tuple[Document, int]:
        found = await self._store.document(doc_id, access=user.access)
        if found is None:
            raise DocumentNotFoundError(doc_id)
        return found

    @staticmethod
    def _may_change(user: User, document: Document) -> bool:
        return user.is_admin or (
            document.owner_id is not None and document.owner_id == user.user_id
        )

    async def detail(self, user: User, doc_id: str) -> DocumentDetail:
        document, chunks = await self._visible(user, doc_id)
        protected = await self._store.protected_collections()
        may_change = self._may_change(user, document)
        return DocumentDetail(
            document=document,
            chunks=chunks,
            indexes=await self._store.document_indexes(doc_id),
            jobs=await self._jobs.list(doc_id=doc_id, limit=20),
            can_edit=may_change,
            can_delete=may_change and (user.is_admin or document.collection_id not in protected),
        )

    async def file(self, user: User, doc_id: str) -> tuple[Path, Document]:
        """The stored original. DocumentNotFoundError when it was not kept or is gone."""
        document, _ = await self._visible(user, doc_id)
        if document.storage_path is None:
            raise DocumentNotFoundError(doc_id)
        root = self._data_dir.resolve()
        path = (root / document.storage_path).resolve()
        # storage_path is written by this code, but a path out of DATA_DIR is never served.
        if not path.is_relative_to(root) or not await asyncio.to_thread(path.is_file):
            raise DocumentNotFoundError(doc_id)
        return path, document

    async def update(self, user: User, doc_id: str, changes: DocumentChanges) -> Document:
        """Raises ForbiddenGroupsError for a group the editor is not in, and
        DuplicateDocumentError when the new groups would make it a copy of another document."""
        document, _ = await self._visible(user, doc_id)
        if not self._may_change(user, document):
            raise DocumentNotFoundError(doc_id)
        if changes.access_groups is not None:
            changes = replace(changes, access_groups=upload_groups(user, changes.access_groups))
        updated = await self._store.update_document(doc_id, changes)
        await self._store.bump_generation()
        return updated

    async def delete(self, user: User, doc_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        """Delete what the user may; returns (deleted, not found). Raises
        ProtectedCollectionError before deleting anything if one is in a protected collection."""
        protected = await self._store.protected_collections()
        targets: list[Document] = []
        missing: list[str] = []
        for doc_id in dict.fromkeys(doc_ids):
            try:
                document, _ = await self._visible(user, doc_id)
            except DocumentNotFoundError:
                missing.append(doc_id)
                continue
            if not self._may_change(user, document):
                missing.append(doc_id)
            elif document.collection_id in protected and not user.is_admin:
                raise ProtectedCollectionError(
                    f"Only an admin can delete from the {document.collection_id} collection."
                )
            else:
                targets.append(document)
        for document in targets:
            await self._cancel_ingestion(document)
            await self._store.delete_document(document.doc_id)
            await self._remove_file(document)
        if targets:
            await self._store.bump_generation()
        return [document.doc_id for document in targets], missing

    async def _cancel_ingestion(self, document: Document) -> None:
        """Unfinished jobs for the same file and owner: re-ingesting it must not bring it back."""
        active = await self._jobs.list(
            owner_id=document.owner_id, statuses=("queued", *RUNNING), limit=10_000
        )
        for job in active:
            if job.payload.get("path") == document.storage_path:
                await self._jobs.cancel(job.job_id)

    async def _remove_file(self, document: Document) -> None:
        path = document.storage_path
        # Only uploads: the corpora under raw/ are the repository's files.
        if path is None or not path.startswith(f"{UPLOADS_FOLDER}/"):
            return
        if not await self._store.storage_path_in_use(path):
            (self._data_dir / path).unlink(missing_ok=True)
            logger.info("Removed %s", path)
