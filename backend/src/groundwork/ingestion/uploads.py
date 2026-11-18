"""Accept an uploaded file, then index it in the background."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from functools import partial
from pathlib import Path, PureWindowsPath

from groundwork.domain.models import DocumentMeta, content_hash
from groundwork.domain.users import User, private_group
from groundwork.ingestion.jobs import Job, JobRegistry
from groundwork.ingestion.loaders import LoadedDocument, NoTextError, loader_for
from groundwork.ingestion.pipeline import prepare_document
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.store import ChunkStore

logger = logging.getLogger(__name__)

# Embedding is the slow step, so its batches drive the progress bar from 20 to 90.
_EMBEDDING_START = 20
_EMBEDDING_END = 90

# Runs a coroutine function after the response is sent, such as FastAPI's
# `BackgroundTasks.add_task`.
Schedule = Callable[[Callable[[], Awaitable[None]]], None]


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
        self,
        embedder: Embedder,
        store: ChunkStore,
        *,
        uploads_dir: Path,
        max_bytes: int,
        chunk_tokens: int,
        chunk_overlap: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._uploads_dir = uploads_dir
        self._max_bytes = max_bytes
        self._chunk_tokens = chunk_tokens
        self._chunk_overlap = chunk_overlap
        self._jobs = JobRegistry()
        # One upload stored at a time in this worker, so two copies of one file racing through
        # the pipeline are stored once.
        self._write_lock = asyncio.Lock()

    def job(self, job_id: str, user: User) -> Job | None:
        """The job, if this user started it. Another user's job is None, as if it did not exist,
        so its id reveals nothing."""
        job = self._jobs.get(job_id)
        return job if job is not None and job.owner_id == user.user_id else None

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
        schedule: Schedule,
    ) -> Job:
        """Validate a file and schedule its indexing, owned by `owner` and readable by
        `access_groups` (only the owner when empty).

        Loading happens here rather than in the job, so a file with no text gets an immediate
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
        meta = DocumentMeta(
            title=name,
            collection_id=collection_id,
            doc_type=doc_type,
            doc_date=doc_date,
            filename=name,
            access_groups=groups,
            owner_id=owner.user_id,
        )
        if doc_id := await self._store.find_document(digest, meta):
            job = self._jobs.create(owner.user_id, doc_id)
            chunks = await self._store.count_document_chunks(doc_id)
            return self._jobs.update(job.job_id, "indexed", 100, chunks=chunks)

        # Loaded from memory, so a file is written to disk only once it has passed every check.
        loaded = await asyncio.to_thread(load, raw)
        if not loaded.text.strip():
            raise NoTextError("No text was found in this file.")
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        (self._uploads_dir / f"{digest}{Path(name).suffix.lower()}").write_bytes(raw)

        job = self._jobs.create(owner.user_id)
        schedule(partial(self._index, job.job_id, digest, loaded, meta))
        return job

    async def _index(
        self, job_id: str, digest: str, loaded: LoadedDocument, meta: DocumentMeta
    ) -> None:
        try:
            await self._chunk_embed_store(job_id, digest, loaded, meta)
        except Exception:
            # The exception stays in the log: job errors are shown to users.
            logger.exception("Indexing failed for job %s", job_id)
            self._jobs.update(job_id, "failed", 0, error="Indexing failed. Please try again.")

    async def _chunk_embed_store(
        self, job_id: str, digest: str, loaded: LoadedDocument, meta: DocumentMeta
    ) -> None:
        self._jobs.update(job_id, "parsing", 10)

        def report(share: float) -> None:
            progress = _EMBEDDING_START + round((_EMBEDDING_END - _EMBEDDING_START) * share)
            self._jobs.update(job_id, "embedding", progress)

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
            self._jobs.update(job_id, "failed", 0, error="This file has too little text to index.")
            return

        async with self._write_lock:
            # Checked again: the same file may have been stored while this one was embedding.
            doc_id = await self._store.find_document(digest, meta)
            if doc_id is None:
                document = await self._store.add(meta, digest, prepared.chunks, prepared.embeddings)
                doc_id = document.doc_id
                # One upload is one job, so the generation moves once per file.
                await self._store.bump_generation()
        self._jobs.update(job_id, "indexed", 100, chunks=len(prepared.chunks), doc_id=doc_id)
