"""Build the indexes from data/raw/. Safe to re-run: files already indexed are skipped.

First makes data/raw/ hold the corpora version locked in conf/corpora.lock.json, downloading it
if needed; without a lock file data/raw/ is used as it is.

Run from backend/: uv run python scripts/ingest.py
"""

import time
from pathlib import Path

from groundwork.config import Settings
from groundwork.container import build_dense_index, build_embedder, rebuild_sparse_index
from groundwork.ingestion.corpora_bundle import LOCK_FILE, ensure_locked
from groundwork.ingestion.corpus import load_corpus
from groundwork.ingestion.pipeline import ingest_file

BACKEND_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = Settings()
    raw_dir = settings.data_dir / "raw"
    print(ensure_locked(raw_dir, BACKEND_DIR / LOCK_FILE))
    embedder = build_embedder(settings)
    index = build_dense_index(settings)

    for path, meta in load_corpus(raw_dir):
        started = time.perf_counter()
        result = ingest_file(
            path,
            meta,
            embedder,
            index,
            chunk_tokens=settings.chunk_tokens,
            chunk_overlap=settings.chunk_overlap,
        )
        if result.skipped:
            print(f"{path.name}: already indexed")
        else:
            elapsed = time.perf_counter() - started
            print(f"{path.name}: {result.chunks} chunks in {elapsed:.0f}s")

    sparse = rebuild_sparse_index(settings, index)
    print(f"Index: {index.count_documents()} documents, {len(sparse)} chunks")


if __name__ == "__main__":
    main()
