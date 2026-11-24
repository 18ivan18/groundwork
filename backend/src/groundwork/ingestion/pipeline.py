"""Ingestion steps from a file to the index, shared by corpus sync, uploads and the evals."""

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from groundwork.domain.models import (
    Chunk,
    Document,
    DocumentMeta,
    content_hash,
    indexed_text,
)
from groundwork.ingestion.chunking import TokenCounter, chunk_document
from groundwork.ingestion.loaders import LoadedDocument, load_file, loader_for
from groundwork.ingestion.structure import StructureDetector, detector_for
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.store import ChunkStore

# Chunks per embedding call. Only progress reporting needs batches: the embedder batches again
# inside each call.
EMBED_BATCH = 64

# Receives the share of chunks embedded so far, from 0.0 to 1.0.
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class IngestResult:
    # None when the file had too little text to index.
    doc_id: str | None
    chunks: int
    skipped: bool


@dataclass(frozen=True)
class PreparedDocument:
    """A document chunked and embedded, ready to store. `chunks` is empty if it had too little
    text to index. Ids in `document` and `chunks` are placeholders until the store assigns its
    own."""

    document: Document
    chunks: list[Chunk]
    embeddings: list[list[float]]


def chunk_loaded(
    loaded: LoadedDocument,
    doc_id: str,
    counter: TokenCounter,
    detector: StructureDetector,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[Chunk]:
    return chunk_document(
        doc_id,
        loaded.text,
        detector(loaded),
        counter,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        page_spans=loaded.page_spans,
    )


def chunk_file(
    path: Path,
    meta: DocumentMeta,
    doc_id: str,
    counter: TokenCounter,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[Chunk]:
    return chunk_loaded(
        load_file(path),
        doc_id,
        counter,
        detector_for(meta.doc_type, meta.filename),
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )


def prepare_document(
    loaded: LoadedDocument,
    meta: DocumentMeta,
    doc_id: str,
    embedder: Embedder,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
    on_progress: ProgressCallback | None = None,
) -> PreparedDocument:
    """Chunk and embed a loaded file: every step before it is stored.

    `on_progress` is called with 0.0 once chunking is done, then after each embedding batch.
    """
    chunks = chunk_loaded(
        loaded,
        doc_id,
        embedder,
        detector_for(meta.doc_type, meta.filename),
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
    document = Document(**asdict(meta), doc_id=doc_id, ingested_at=datetime.now(UTC))
    if not chunks:
        return PreparedDocument(document, [], [])

    texts = [indexed_text(chunk, document) for chunk in chunks]
    embeddings: list[list[float]] = []
    if on_progress:
        on_progress(0.0)
    for start in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(embedder.embed_documents(texts[start : start + EMBED_BATCH]))
        if on_progress:
            on_progress(len(embeddings) / len(texts))
    return PreparedDocument(document, chunks, embeddings)


async def ingest_file(
    path: Path,
    meta: DocumentMeta,
    embedder: Embedder,
    store: ChunkStore,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
) -> IngestResult:
    """Chunk, embed and store a file, unless the store already holds it. It gets no manifest key,
    so a corpus sync leaves it alone. Does not publish: the caller does that once, after its last
    file."""
    raw = path.read_bytes()
    digest = content_hash(raw)
    if doc_id := await store.find_document(digest, meta):
        return IngestResult(doc_id=doc_id, chunks=0, skipped=True)

    loaded = loader_for(path.name)(raw)
    prepared = await asyncio.to_thread(
        prepare_document,
        loaded,
        meta,
        digest,
        embedder,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
    if not prepared.chunks:
        return IngestResult(doc_id=None, chunks=0, skipped=False)
    document = await store.add(meta, digest, prepared.chunks, prepared.embeddings)
    return IngestResult(doc_id=document.doc_id, chunks=len(prepared.chunks), skipped=False)
