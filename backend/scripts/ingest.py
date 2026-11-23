"""Index data/raw/ into the active index on Postgres. Safe to re-run: documents already in the
index are skipped.

With no active index yet, it creates and activates the window index the chunking settings
describe. The index's generation moves once, at the end, so each API worker rebuilds BM25 once.

First makes data/raw/ hold the corpora version locked in conf/corpora.lock.json, downloading it
if needed. GROUNDWORK_CORPORA=local skips that and uses data/raw/ as it is, unchecked.

Run from backend/: uv run python scripts/ingest.py
"""

import asyncio
import time

from groundwork.config import Settings
from groundwork.container import app_index, build_embedder, chunking
from groundwork.db.pool import open_pool
from groundwork.ingestion.corpora_bundle import ensure_corpora
from groundwork.ingestion.corpus import load_corpus
from groundwork.ingestion.pipeline import ingest_file
from groundwork.retrieval.pgvector_store import PgVectorStore


async def main() -> None:
    settings = Settings()
    raw_dir = ensure_corpora(settings)
    embedder = build_embedder(settings)
    pool = await open_pool(settings.database_url)
    try:
        store = PgVectorStore(pool, await app_index(pool, settings, embedder.dimension))
        chunk_tokens, chunk_overlap = chunking(store.index, settings)
        print(f"Index: {store.index.name}")
        added = 0
        for path, meta in load_corpus(raw_dir):
            started = time.perf_counter()
            result = await ingest_file(
                path,
                meta,
                embedder,
                store,
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
            )
            if result.skipped:
                print(f"{path.name}: already indexed")
            elif result.doc_id is None:
                print(f"{path.name}: too little text to index")
            else:
                added += 1
                elapsed = time.perf_counter() - started
                print(f"{path.name}: {result.chunks} chunks in {elapsed:.0f}s")
        if added:
            generation = await store.bump_generation()
            print(f"Generation {generation}")
        documents, chunks = await store.count_documents(), await store.count_chunks()
        print(f"Index: {documents} documents, {chunks} chunks")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
