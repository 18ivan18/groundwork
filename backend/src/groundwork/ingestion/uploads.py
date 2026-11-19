"""Accept an uploaded file and queue it for the ingestion worker (ingestion/worker.py)."""

import asyncio
from collections.abc import Sequence
from datetime import date
from pathlib import Path, PureWindowsPath

from groundwork.domain.models import DocumentMeta, content_hash
from groundwork.domain.users import User, private_group
from groundwork.ingestion.jobs import INGEST_DOCUMENT, Job, JobQueue
from groundwork.ingestion.loaders import NoTextError, loader_for
from groundwork.ingestion.worker import ingest_payload
from groundwork.retrieval.store import ChunkStore

UPLOADS_FOLDER = "uploads"


class FileTooLargeError(ValueError):
    """The file is over the upload limit."""


class ForbiddenGroupsError(ValueError):
    """The uploader asked to share with a group they are not in."""


def upload_groups(user: User, requested: Sequence[str]) -> tuple[str, ...]:
    """The groups an upload is shared with: the uploader's choice from their own groups, or only
    themselves when they choose none."""
    if not requested:
        return (private_group(user.user_id),)
    foreign = sorted(set(requested) - user.groups)
    if foreign:
        raise ForbiddenGroupsError(
            f"You can share only with your own groups, not {', '.join(foreign)}."
        )
    return tuple(sorted(set(requested)))


class UploadService:
    def __init__(
        self, store: ChunkStore, jobs: JobQueue, *, data_dir: Path, max_bytes: int
    ) -> None:
        self._store = store
        self._jobs = jobs
        self._data_dir = data_dir
        self._max_bytes = max_bytes

    async def accept(
        self,
        filename: str,
        raw: bytes,
        *,
        owner: User,
        collection_id: str,
        doc_type: str,
        doc_date: date | None,
        access_groups: Sequence[str] = (),
    ) -> Job:
        """Validate a file and queue its indexing, owned by `owner` and readable by
        `access_groups` (only the owner when empty).

        Loading happens here as well as in the worker, so a file with no text gets an immediate
        error. The same bytes already stored by this owner, in this collection, for these groups
        get a finished job and no new work. Anyone else's copy is never matched: deduplication
        must not reveal that a document exists.
        """
        groups = upload_groups(owner, access_groups)
        # PureWindowsPath splits on both kinds of slash. The name is only ever metadata: the file
        # on disk is named by its hash, so a name like "../../x.txt" cannot choose where it lands.
        name = PureWindowsPath(filename).name
        load = loader_for(name)
        if len(raw) > self._max_bytes:
            limit_mb = self._max_bytes // (1024 * 1024)
            raise FileTooLargeError(f"The file is larger than the {limit_mb} MB limit.")

        digest = content_hash(raw)
        relative = f"{UPLOADS_FOLDER}/{digest}{Path(name).suffix.lower()}"
        meta = DocumentMeta(
            title=name,
            collection_id=collection_id,
            doc_type=doc_type,
            doc_date=doc_date,
            filename=name,
            access_groups=groups,
            owner_id=owner.user_id,
            storage_path=relative,
        )
        if doc_id := await self._store.find_document(digest, meta):
            chunks = await self._store.count_document_chunks(doc_id)
            return await self._jobs.record_done(
                INGEST_DOCUMENT, owner_id=owner.user_id, doc_id=doc_id, chunks=chunks
            )

        # Loaded from memory, so a file is written to disk only once it has passed every check.
        loaded = await asyncio.to_thread(load, raw)
        if not loaded.text.strip():
            raise NoTextError("No text was found in this file.")
        path = self._data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, raw)
        return await self._jobs.enqueue(
            INGEST_DOCUMENT, ingest_payload(relative, digest, meta), owner_id=owner.user_id
        )
