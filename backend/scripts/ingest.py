"""Make the active index on Postgres hold exactly the corpus documents in data/raw/, beside the
users' uploads, which it never changes. Documents that left the corpus are removed, changed ones
are replaced, and unchanged ones are skipped without embedding, so it is safe and fast to re-run.

With no active index yet, it creates and activates the window index the chunking settings
describe. It publishes once, at the end, so each API worker rebuilds BM25 once: also after a
failure, and also what an earlier ingest stored but did not publish. A document removed or
replaced is gone from the database's other indexes too, which are published as well. After
publishing anything it vacuums.

A document that fails doesn't stop the others: the script names each one and its error at the end,
without the document's text, and exits with status 1. A failed document keeps its old copy, unless
its access groups changed: then the old copy is removed.

First makes data/raw/ hold the corpora version locked in conf/corpora.lock.json, downloading it
if needed. GROUNDWORK_CORPORA=local skips that and uses data/raw/ as it is, unchecked. Fails if it
holds no corpus documents, so an empty library never looks like success.

Run from backend/: uv run python scripts/ingest.py
"""

import asyncio
import sys
from typing import get_args

from groundwork.config import Settings
from groundwork.container import app_index, build_embedder, chunking
from groundwork.db.pool import open_pool
from groundwork.ingestion.corpora_bundle import ensure_corpora
from groundwork.ingestion.corpus import corpus_entries
from groundwork.ingestion.sync import SyncOutcome, SyncResult, sync_corpus
from groundwork.retrieval.pgvector_store import PgVectorStore


def print_result(result: SyncResult) -> None:
    if result.outcome == "failed":
        print(failure(result), file=sys.stderr)
    elif result.chunks:
        print(f"{result.name}: {result.outcome}, {result.chunks} chunks in {result.seconds:.0f}s")
    elif result.outcome != "skipped":
        print(f"{result.name}: {result.outcome}")


def failure(result: SyncResult) -> str:
    line = f"{result.name}: {result.error}"
    match result.old_copy:
        case "kept":
            return f"{line}. Its old copy was kept."
        case "removed":
            return f"{line}. Its access groups changed, so its old copy was removed."
        case "not removed":
            return (
                f"{line}.\n  WARNING: {result.name} IS STILL READABLE BY ITS OLD ACCESS GROUPS. "
                "They changed, but its old copy could not be removed. Run ingest again."
            )
        case None:
            return f"{line}. It is not indexed."


async def main() -> None:
    settings = Settings()
    raw_dir = ensure_corpora(settings)
    entries = corpus_entries(raw_dir)
    if not entries:
        sys.exit(f"No corpus documents in {raw_dir}, so nothing was indexed.")
    embedder = build_embedder(settings)
    pool = await open_pool(settings.database_url)
    try:
        store = PgVectorStore(pool, await app_index(pool, settings, embedder.dimension))
        chunk_tokens, chunk_overlap = chunking(store.index, settings)
        print(f"Index: {store.index.name}")
        report = await sync_corpus(
            entries,
            embedder,
            store,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            on_result=print_result,
        )
        outcomes = get_args(SyncOutcome)
        print(", ".join(f"{len(report.names(outcome))} {outcome}" for outcome in outcomes))
        for name, generation in report.published.items():
            print(f"Published {name}: generation {generation}")
        documents, chunks = await store.count_documents(), await store.count_chunks()
        print(f"Index: {documents} documents, {chunks} chunks")
        if report.failures:
            count = len(report.failures)
            lines = [f"{count} {'document' if count == 1 else 'documents'} failed:"]
            sys.exit("\n".join(lines + [failure(result) for result in report.failures]))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
