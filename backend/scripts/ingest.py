"""Queue data/raw/ for the ingestion worker: one `ingest_document` job per file the active index
does not hold yet. Safe to re-run: files already indexed, or already queued, are skipped.

With no active index yet, it creates and activates the window index the chunking settings
describe. The worker service (`python -m groundwork.ingestion.worker`) does the indexing; with
`--work`, this script does it too and returns once the queue is empty.

Run from backend/: uv run python scripts/ingest.py [--work]
"""

import argparse
import asyncio
import time
from dataclasses import replace

from groundwork.config import Settings
from groundwork.container import app_index, build_embedder, chunking
from groundwork.db.pool import open_pool
from groundwork.domain.models import content_hash
from groundwork.ingestion.corpus import load_corpus
from groundwork.ingestion.jobs import INGEST_DOCUMENT, RUNNING
from groundwork.ingestion.pg_jobs import PgJobQueue
from groundwork.ingestion.worker import IngestWorker, ingest_payload
from groundwork.retrieval.pgvector_store import PgVectorStore


async def main(work: bool) -> None:
    settings = Settings()
    embedder = build_embedder(settings)
    pool = await open_pool(settings.database_url)
    try:
        store = PgVectorStore(pool, await app_index(pool, settings, embedder.dimension))
        queue = PgJobQueue(pool)
        print(f"Index: {store.index.name}")
        pending = await queue.list(statuses=("queued", *RUNNING), limit=100_000)
        queued_paths = {job.payload.get("path") for job in pending}
        added = 0
        protected: set[str] = set()
        for path, corpus_meta in load_corpus(settings.data_dir / "raw"):
            relative = path.relative_to(settings.data_dir).as_posix()
            meta = replace(corpus_meta, storage_path=relative)
            if meta.collection_id not in protected:
                # The demo corpora: only an admin may delete from them.
                await store.protect_collection(meta.collection_id)
                protected.add(meta.collection_id)
            digest = content_hash(path.read_bytes())
            if relative in queued_paths or await store.find_document(digest, meta):
                continue
            await queue.enqueue(
                INGEST_DOCUMENT, ingest_payload(relative, digest, meta), owner_id=None
            )
            added += 1
        print(f"Queued {added} files ({len(queued_paths)} already queued)")

        if work:
            chunk_tokens, chunk_overlap = chunking(store.index, settings)
            worker = IngestWorker(
                queue,
                store,
                embedder,
                data_dir=settings.data_dir,
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
            )
            started = time.perf_counter()
            await worker.drain()
            print(f"Worked through the queue in {time.perf_counter() - started:.0f}s")
        documents, chunks = await store.count_documents(), await store.count_chunks()
        print(f"Index: {documents} documents, {chunks} chunks")
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--work", action="store_true", help="index the queue here, without the worker service"
    )
    asyncio.run(main(parser.parse_args().work))
