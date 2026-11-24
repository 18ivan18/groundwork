"""Corpus sync: make the index hold exactly the corpus documents the manifests describe, beside
the users' uploads, which a sync never changes."""

import asyncio
import os
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import groundwork
from groundwork.domain.models import content_hash
from groundwork.ingestion.corpus import CorpusEntry
from groundwork.ingestion.loaders import loader_for
from groundwork.ingestion.pipeline import PreparedDocument, prepare_document
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.store import ChunkStore, CorpusDocument, normalized_groups

# "removed" is also the outcome of a document that nothing replaces: its new text is too little to
# index, or an upload stands in for it.
SyncOutcome = Literal["added", "replaced", "removed", "skipped", "too little text", "failed"]
# What became of the stored copy of a document that failed. It is removed, from every index, when
# the document's access groups changed. "not removed" is when that failed too: the text can still
# be read by its old groups.
OldCopy = Literal["kept", "removed", "not removed"]

_PACKAGE = Path(groundwork.__file__).parent


@dataclass(frozen=True)
class SyncResult:
    # `<collection>/<key>`
    name: str
    outcome: SyncOutcome
    # Stored by this sync.
    chunks: int = 0
    seconds: float = 0.0
    # Of a failed document: what went wrong, without any of the document's text.
    error: str | None = None
    # Of a failed document that was stored before.
    old_copy: OldCopy | None = None


@dataclass
class SyncReport:
    """What became of each document, in the order it happened."""

    results: list[SyncResult] = field(default_factory=list)
    # The new generation of each index that was published, by index name.
    published: dict[str, int] = field(default_factory=dict)

    def names(self, outcome: SyncOutcome) -> list[str]:
        return [result.name for result in self.results if result.outcome == outcome]

    @property
    def failures(self) -> list[SyncResult]:
        return [result for result in self.results if result.outcome == "failed"]


# Receives what became of each document as it happens.
ResultCallback = Callable[[SyncResult], None]


def _name(collection: str, key: str) -> str:
    return f"{collection}/{key}"


async def sync_corpus(
    entries: Sequence[CorpusEntry],
    embedder: Embedder,
    store: ChunkStore,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
    on_result: ResultCallback | None = None,
) -> SyncReport:
    """Make the index match `entries`, which must be the whole corpus: a corpus document they
    don't list is removed. Raises ValueError for no entries, so that a missing directory can't
    empty the library.

    Documents that left the corpus go first, from every index. A changed one is replaced in one
    transaction, after its new text is embedded; the other indexes lose it too, and get the new
    one at their own next sync.

    A document that can't be read, loaded, chunked, embedded or stored fails, and the sync goes on
    with the others: the caller must look at `failures`. It keeps its old copy, unless its access
    groups changed. Then the old copy is removed from every index, so that no one reads it through
    the groups it had. Anything but an Exception, such as cancellation, stops the sync.

    Every index with writes to publish is published once at the end: also after a failure or a
    stop, and also what an earlier process wrote and never published. After that, if it published
    anything, it vacuums.

    One sync runs at a time on a database. Uploads may be stored beside it."""
    if not entries:
        raise ValueError("There are no corpus documents to sync.")
    names = Counter(_name(entry.meta.collection_id, entry.key) for entry in entries)
    if repeated := sorted(name for name, count in names.items() if count > 1):
        raise ValueError(f"Corpus documents listed twice: {', '.join(repeated)}")

    report = SyncReport()

    def add_result(result: SyncResult) -> None:
        report.results.append(result)
        if on_result:
            on_result(result)

    try:
        async with store.syncing():
            try:
                stored = {
                    _name(item.document.collection_id, item.key): item
                    for item in await store.corpus_documents()
                }
                gone = {name: item for name, item in stored.items() if name not in names}
                if gone:
                    await store.remove_corpus_documents([i.document.doc_id for i in gone.values()])
                    for name in gone:
                        add_result(SyncResult(name, "removed"))
                for entry in entries:
                    name = _name(entry.meta.collection_id, entry.key)
                    started = time.perf_counter()
                    try:
                        outcome, chunks = await _sync_document(
                            entry, stored.get(name), embedder, store, chunk_tokens, chunk_overlap
                        )
                        result = SyncResult(name, outcome, chunks, time.perf_counter() - started)
                    except Exception as error:
                        result = await _failed(name, error, entry, stored.get(name), store)
                    add_result(result)
            finally:
                report.published = await store.publish_all()
    finally:
        # After the lock and its transaction, and not on an unchanged rerun. What was published
        # rather than what this sync did, so that it includes what a killed one removed.
        if report.published:
            await store.vacuum()
    return report


async def _failed(
    name: str,
    error: Exception,
    entry: CorpusEntry,
    stored: CorpusDocument | None,
    store: ChunkStore,
) -> SyncResult:
    """The result of a document that failed, after removing its old copy if its groups changed."""
    if stored is None:
        return SyncResult(name, "failed", error=_describe(error))
    if stored.document.access_groups == normalized_groups(entry.meta.access_groups):
        return SyncResult(name, "failed", error=_describe(error), old_copy="kept")
    try:
        await store.remove_corpus_documents([stored.document.doc_id])
    except Exception as removal:
        both = f"{_describe(error)}; removing the old copy: {_describe(removal)}"
        return SyncResult(name, "failed", error=both, old_copy="not removed")
    return SyncResult(name, "failed", error=_describe(error), old_copy="removed")


def _describe(error: Exception) -> str:
    """The error's type and where in this package it came from. Not its message, which may quote
    the document: an embedder's its input, a database's the row. For Python's own operating system
    errors, the text of their number is added."""
    kind = type(error)
    description = kind.__qualname__
    if isinstance(error, OSError) and kind.__module__ == "builtins" and error.errno is not None:
        description += f" ({os.strerror(error.errno)})"
    frames = [
        frame.f_code
        for frame, _ in traceback.walk_tb(error.__traceback__)
        if Path(frame.f_code.co_filename).is_relative_to(_PACKAGE)
    ]
    if frames:
        description += f" in {Path(frames[-1].co_filename).name}, {frames[-1].co_name}"
    return description


async def _sync_document(
    entry: CorpusEntry,
    stored: CorpusDocument | None,
    embedder: Embedder,
    store: ChunkStore,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[SyncOutcome, int]:
    """Make the index hold this document as its manifest describes it. The outcome, and how many
    chunks were stored."""
    raw = await asyncio.to_thread(entry.path.read_bytes)
    digest = content_hash(raw)
    current = stored is not None and stored.is_current(digest, entry.meta)
    # An upload that says the same to the same readers stands in for the corpus document, in the
    # indexes that hold the upload's chunks. This is what becomes of the corpus document then: one
    # that only other indexes hold stays, because the upload does not stand in there.
    stood_in: SyncOutcome = (
        "removed" if stored is not None and (stored.in_index or not current) else "skipped"
    )
    if await store.find_upload(digest, entry.meta) is not None:
        if stored is not None and stood_in == "removed":
            await store.remove_corpus_documents([stored.document.doc_id])
        return stood_in, 0
    if stored is not None and current and stored.in_index:
        return "skipped", 0

    def prepare() -> PreparedDocument:
        return prepare_document(
            loader_for(entry.path.name)(raw),
            entry.meta,
            digest,
            embedder,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
        )

    prepared = await asyncio.to_thread(prepare)
    if not prepared.chunks:
        if stored is not None and not current:
            await store.remove_corpus_documents([stored.document.doc_id])
            return "removed", 0
        return "too little text", 0
    document = await store.add_corpus_document(
        entry.meta, digest, prepared.chunks, prepared.embeddings, manifest_key=entry.key
    )
    if document is None:
        # The upload was stored while this was embedding.
        return stood_in, 0
    return ("added" if stored is None or current else "replaced"), len(prepared.chunks)
