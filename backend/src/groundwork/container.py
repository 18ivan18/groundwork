"""Wires settings to the application's long-lived objects."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from groundwork.config import Settings
from groundwork.db.pool import Pool, open_pool
from groundwork.domain.users import User
from groundwork.ingestion.uploads import UploadService
from groundwork.llm.base import LLMClient
from groundwork.llm.factory import build_llm
from groundwork.qa.service import QAService
from groundwork.retrieval.embedder import (
    Embedder,
    FastEmbedEmbedder,
    SentenceTransformerEmbedder,
)
from groundwork.retrieval.hybrid import HybridRetriever
from groundwork.retrieval.indexes import (
    IndexInfo,
    IndexSpec,
    activate,
    active_index,
    ensure_index,
    window_spec,
)
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.reranker import FastEmbedReranker, Reranker
from groundwork.retrieval.sparse import SparseCache, SparseSource
from groundwork.retrieval.store import ChunkStore


class EmbeddingModelMismatchError(ValueError):
    """The index was built with another model. Mixing models gives meaningless similarities."""


@dataclass(frozen=True)
class Container:
    settings: Settings
    store: ChunkStore
    # Loaded at startup so a broken file stops the server at once. Enforced from M3.
    users: Mapping[str, User]


@dataclass(frozen=True)
class Services:
    """The objects that need the embedding model and the LLM, built once a server starts."""

    qa: QAService
    uploads: UploadService


def build_embedder(settings: Settings) -> Embedder:
    match settings.embedding_provider:
        case "fastembed":
            return FastEmbedEmbedder(settings.embedding_model)
        case "sentence_transformers":
            return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)


def settings_index_spec(settings: Settings, dimension: int) -> IndexSpec:
    """The window index the chunking and embedding settings describe."""
    return window_spec(
        settings.chunk_tokens, settings.chunk_overlap, settings.embedding_model, dimension
    )


async def app_index(pool: Pool, settings: Settings, dimension: int) -> IndexInfo:
    """The active index. With none active yet, the window index the settings describe, created
    and activated. Raises EmbeddingModelMismatchError if the active index was embedded with a
    model other than the configured one."""
    index = await active_index(pool)
    if index is None:
        created = await ensure_index(pool, settings_index_spec(settings, dimension))
        await activate(pool, created.id)
        index = await active_index(pool)
        assert index is not None
    if index.spec.embedder_model != settings.embedding_model:
        raise EmbeddingModelMismatchError(
            f"The active index {index.name} was built with {index.spec.embedder_model}, not "
            f"{settings.embedding_model}. Set EMBEDDING_MODEL to match, or activate another index."
        )
    return index


def chunking(index: IndexInfo, settings: Settings) -> tuple[int, int]:
    """(chunk_tokens, chunk_overlap) the index was built with, so uploads are chunked alike."""
    params = index.spec.chunker_params
    return (
        int(params.get("chunk_tokens", settings.chunk_tokens)),
        int(params.get("chunk_overlap", settings.chunk_overlap)),
    )


def build_reranker(settings: Settings) -> Reranker | None:
    return FastEmbedReranker(settings.reranker_model) if settings.reranker_model else None


def bm25_root(settings: Settings) -> Path:
    return settings.data_dir / "bm25"


def make_retriever(
    settings: Settings,
    embedder: Embedder,
    store: ChunkStore,
    sparse: SparseSource,
    reranker: Reranker | None,
) -> HybridRetriever:
    """The retriever from parts already built. The evaluation uses this too, so it always
    measures the settings the app runs with."""
    return HybridRetriever(
        embedder,
        store,
        sparse,
        candidates=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
    )


def build_services(
    settings: Settings,
    store: ChunkStore,
    embedder: Embedder | None = None,
    *,
    llm: LLMClient | None = None,
) -> Services:
    # The LLM comes first: a missing setting should fail before the embedder loads.
    llm = llm or build_llm(settings)
    embedder = embedder or build_embedder(settings)
    sparse = SparseCache(store, bm25_root(settings))
    retriever = make_retriever(settings, embedder, store, sparse, build_reranker(settings))
    chunk_tokens, chunk_overlap = chunking(store.index, settings)
    return Services(
        qa=QAService(
            retriever,
            llm,
            top_k=settings.top_k,
            min_relevance=settings.min_relevance,
            max_tokens=settings.llm_max_tokens,
        ),
        uploads=UploadService(
            embedder,
            store,
            uploads_dir=settings.data_dir / "uploads",
            max_bytes=settings.max_upload_mb * 1024 * 1024,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
        ),
    )


@dataclass(frozen=True)
class AppStore:
    store: PgVectorStore
    embedder: Embedder


@asynccontextmanager
async def open_app_store(settings: Settings) -> AsyncIterator[AppStore]:
    """For scripts: the embedder and the active index's store, with the pool closed after."""
    embedder = await asyncio.to_thread(build_embedder, settings)
    pool = await open_pool(settings.database_url)
    try:
        index = await app_index(pool, settings, embedder.dimension)
        yield AppStore(PgVectorStore(pool, index), embedder)
    finally:
        await pool.close()
