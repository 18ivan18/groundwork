"""Index data/raw/ into the active index on Postgres. Safe to re-run: documents already in the
index are skipped.

With no active index yet, it creates and activates the window index the chunking settings
describe. It publishes once, at the end, so each API worker rebuilds BM25 once: also after a
failure, and also what an earlier ingest stored but did not publish.

First makes data/raw/ hold the corpora version locked in conf/corpora.lock.json, downloading it
if needed. GROUNDWORK_CORPORA=local skips that and uses data/raw/ as it is, unchecked. Fails if it
holds no corpus documents, so an empty library never looks like success.

Run from backend/: uv run python scripts/ingest.py
"""

import asyncio
import sys
from pathlib import Path

from groundwork.config import Settings
from groundwork.container import app_index, build_embedder, chunking
from groundwork.db.pool import open_pool
from groundwork.ingestion.corpora_bundle import ensure_corpora
from groundwork.ingestion.corpus import load_corpus
from groundwork.ingestion.pipeline import IngestResult, ingest_corpus
from groundwork.retrieval.pgvector_store import PgVectorStore


def report(path: Path, result: IngestResult, seconds: float) -> None:
    if result.skipped:
        print(f"{path.name}: already indexed")
    elif result.doc_id is None:
        print(f"{path.name}: too little text to index")
    else:
        print(f"{path.name}: {result.chunks} chunks in {seconds:.0f}s")


async def main() -> None:
    settings = Settings()
    raw_dir = ensure_corpora(settings)
    corpus = load_corpus(raw_dir)
    if not corpus:
        sys.exit(f"No corpus documents in {raw_dir}, so nothing was indexed.")
    embedder = build_embedder(settings)
    pool = await open_pool(settings.database_url)
    try:
        store = PgVectorStore(pool, await app_index(pool, settings, embedder.dimension))
        chunk_tokens, chunk_overlap = chunking(store.index, settings)
        print(f"Index: {store.index.name}")
        generation = await ingest_corpus(
            corpus,
            embedder,
            store,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            on_result=report,
        )
        if generation is not None:
            print(f"Generation {generation}")
        documents, chunks = await store.count_documents(), await store.count_chunks()
        print(f"Index: {documents} documents, {chunks} chunks")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
