from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from groundwork.domain.models import DocumentMeta, content_hash, make_chunk_id, parse_chunk_id
from groundwork.ingestion import pipeline
from groundwork.ingestion.loaders import LoadedDocument
from groundwork.ingestion.pipeline import ingest_corpus, ingest_file, prepare_document
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.memory_store import MemoryData, MemoryStore, memory_index
from groundwork.retrieval.sparse import SparseCache

META = DocumentMeta(
    title="ACME Corp · Form 10-K",
    collection_id="sec-10k",
    doc_type="10-K",
    doc_date=None,
    metadata={"company": "ACME Corp", "ticker": "ACME"},
    filename="acme.txt",
)
TEXT = "Item 1. Business\n\n" + "ACME Corp sells anvils to coyotes in the desert. " * 40


class CountingEmbedder(FakeEmbedder):
    calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        self.calls += 1
        return super().embed_documents(texts)


class RecordingEmbedder(FakeEmbedder):
    texts: Sequence[str] = ()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts = texts
        return super().embed_documents(texts)


async def ingest(
    path: Path, embedder: FakeEmbedder, store: MemoryStore
) -> tuple[str | None, int, bool]:
    result = await ingest_file(path, META, embedder, store, chunk_tokens=100, chunk_overlap=10)
    return result.doc_id, result.chunks, result.skipped


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(memory_index(FakeEmbedder().dimension))


@pytest.fixture
def filing(tmp_path: Path) -> Path:
    path = tmp_path / "acme.txt"
    path.write_text(TEXT)
    return path


async def test_ingest_indexes_a_new_file(filing: Path, store: MemoryStore) -> None:
    doc_id, chunks, skipped = await ingest(filing, FakeEmbedder(), store)

    assert not skipped
    assert chunks > 1
    assert await store.count_chunks() == chunks
    assert await store.find_document(content_hash(TEXT.encode()), META) == doc_id


async def test_ingest_is_idempotent_and_does_not_embed_twice(
    filing: Path, store: MemoryStore
) -> None:
    embedder = CountingEmbedder()
    first, _, _ = await ingest(filing, embedder, store)
    stored = await store.count_chunks()

    again, _, skipped = await ingest(filing, embedder, store)

    assert skipped
    assert again == first
    assert embedder.calls == 1
    assert await store.count_chunks() == stored


async def test_changed_content_is_a_new_document(filing: Path, store: MemoryStore) -> None:
    await ingest(filing, FakeEmbedder(), store)
    filing.write_text(TEXT + "New sentence about ACME.")

    _, _, skipped = await ingest(filing, FakeEmbedder(), store)

    assert not skipped
    assert await store.count_documents() == 2


async def test_too_little_text_stores_nothing(tmp_path: Path, store: MemoryStore) -> None:
    path = tmp_path / "short.txt"
    path.write_text("Too short.")

    assert await ingest(path, FakeEmbedder(), store) == (None, 0, False)
    assert await store.count_documents() == 0


async def test_header_is_embedded_but_stored_text_stays_verbatim(
    filing: Path, store: MemoryStore
) -> None:
    embedder = RecordingEmbedder()

    await ingest(filing, embedder, store)

    assert embedder.texts[0].startswith("ACME Corp · Form 10-K · Item 1 — Business\n")
    chunk, _ = (await store.all_passages())[0]
    assert "Form 10-K" not in chunk.text


def test_content_hash_is_the_sha256_and_chunk_ids_round_trip() -> None:
    assert content_hash(b"content") == (
        "ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
    )
    assert make_chunk_id("42", 7) == "42:7"
    assert parse_chunk_id("42:7") == ("42", 7)


def test_prepare_reports_progress_per_embedding_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "EMBED_BATCH", 2)
    shares: list[float] = []

    prepared = prepare_document(
        LoadedDocument(TEXT),
        META,
        "doc",
        FakeEmbedder(),
        chunk_tokens=100,
        chunk_overlap=10,
        on_progress=shares.append,
    )

    assert len(prepared.embeddings) == len(prepared.chunks) > 2
    assert shares[0] == 0.0
    assert shares[-1] == 1.0
    assert shares == sorted(shares)
    assert len(shares) == 1 + -(-len(prepared.chunks) // 2)


def test_prepare_returns_no_chunks_for_too_little_text() -> None:
    progress: list[float] = []

    prepared = prepare_document(
        LoadedDocument("Too short."),
        META,
        "doc",
        FakeEmbedder(),
        chunk_tokens=100,
        chunk_overlap=10,
        on_progress=progress.append,
    )

    assert (prepared.chunks, prepared.embeddings, progress) == ([], [], [])


class BrokenOnRoadrunners(FakeEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any("roadrunners" in text for text in texts):
            raise RuntimeError("out of memory")
        return super().embed_documents(texts)


@pytest.fixture
def corpus(tmp_path: Path) -> list[tuple[Path, DocumentMeta]]:
    entries = []
    for name in ("anvils", "roadrunners"):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"Item 1. Business\n\n{f'ACME Corp knows all about {name}. ' * 40}")
        entries.append((path, replace(META, title=name, filename=path.name)))
    return entries


async def ingest_all(
    corpus: list[tuple[Path, DocumentMeta]], embedder: FakeEmbedder, store: MemoryStore
) -> int | None:
    return await ingest_corpus(corpus, embedder, store, chunk_tokens=100, chunk_overlap=10)


async def found_by_bm25(store: MemoryStore, root: Path, query: str) -> bool:
    index = await SparseCache(store, root / "bm25", check_seconds=0).current()
    return bool(index.search(query, 1))


async def test_a_bulk_ingest_publishes_once(
    corpus: list[tuple[Path, DocumentMeta]], store: MemoryStore
) -> None:
    assert await ingest_all(corpus, FakeEmbedder(), store) == 1

    assert await store.generation() == 1


async def test_a_rerun_with_nothing_new_publishes_nothing(
    corpus: list[tuple[Path, DocumentMeta]], store: MemoryStore
) -> None:
    await ingest_all(corpus, FakeEmbedder(), store)

    assert await ingest_all(corpus, FakeEmbedder(), store) is None
    assert await store.generation() == 1


async def test_after_an_ingest_fails_partway_and_is_rerun_bm25_finds_every_document(
    corpus: list[tuple[Path, DocumentMeta]], store: MemoryStore, tmp_path: Path
) -> None:
    assert not await found_by_bm25(store, tmp_path, "anvils")

    with pytest.raises(RuntimeError, match="out of memory"):
        await ingest_all(corpus, BrokenOnRoadrunners(), store)

    assert await found_by_bm25(store, tmp_path, "anvils")
    assert await store.generation() == 1

    await ingest_all(corpus, FakeEmbedder(), store)

    assert await found_by_bm25(store, tmp_path, "anvils")
    assert await found_by_bm25(store, tmp_path, "roadrunners")
    assert await store.generation() == 2


async def test_a_killed_ingest_is_published_by_the_next_one(
    corpus: list[tuple[Path, DocumentMeta]], tmp_path: Path
) -> None:
    data = MemoryData()
    killed = MemoryStore(memory_index(FakeEmbedder().dimension), data)
    # Saves an empty BM25 index, as an API running before the ingest does.
    assert not await found_by_bm25(killed, tmp_path, "anvils")
    for path, meta in corpus:
        await ingest_file(path, meta, FakeEmbedder(), killed, chunk_tokens=100, chunk_overlap=10)

    rerun = MemoryStore(memory_index(FakeEmbedder().dimension), data)
    assert await ingest_all(corpus, FakeEmbedder(), rerun) == 1

    assert await found_by_bm25(rerun, tmp_path, "anvils")
    assert await found_by_bm25(rerun, tmp_path, "roadrunners")
