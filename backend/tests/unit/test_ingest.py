from collections.abc import Sequence
from pathlib import Path

import pytest

from groundwork.domain.models import DocumentMeta, content_hash, make_chunk_id
from groundwork.ingestion import pipeline
from groundwork.ingestion.loaders import LoadedDocument
from groundwork.ingestion.pipeline import ingest_file, prepare_document
from groundwork.retrieval.dense import DenseIndex
from groundwork.retrieval.embedder import FakeEmbedder

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


def ingest(path: Path, embedder: FakeEmbedder, index: DenseIndex) -> tuple[str, int, bool]:
    result = ingest_file(path, META, embedder, index, chunk_tokens=100, chunk_overlap=10)
    return result.doc_id, result.chunks, result.skipped


@pytest.fixture
def filing(tmp_path: Path) -> Path:
    path = tmp_path / "acme.txt"
    path.write_text(TEXT)
    return path


def test_ingest_indexes_a_new_file(filing: Path, tmp_path: Path) -> None:
    index = DenseIndex(tmp_path / "index", "fake-model")

    doc_id, chunks, skipped = ingest(filing, FakeEmbedder(), index)

    assert not skipped
    assert chunks > 1
    assert index.count_chunks() == chunks
    assert index.has_document(doc_id)


def test_ingest_is_idempotent_and_does_not_embed_twice(filing: Path, tmp_path: Path) -> None:
    index = DenseIndex(tmp_path / "index", "fake-model")
    embedder = CountingEmbedder()
    ingest(filing, embedder, index)
    stored = index.count_chunks()

    _, _, skipped = ingest(filing, embedder, index)

    assert skipped
    assert embedder.calls == 1
    assert index.count_chunks() == stored


def test_changed_content_is_a_new_document(filing: Path, tmp_path: Path) -> None:
    index = DenseIndex(tmp_path / "index", "fake-model")
    ingest(filing, FakeEmbedder(), index)
    filing.write_text(TEXT + "New sentence about ACME.")

    _, _, skipped = ingest(filing, FakeEmbedder(), index)

    assert not skipped
    assert index.count_documents() == 2


def test_header_is_embedded_but_stored_text_stays_verbatim(filing: Path, tmp_path: Path) -> None:
    index = DenseIndex(tmp_path / "index", "fake-model")
    embedder = RecordingEmbedder()

    ingest(filing, embedder, index)

    assert embedder.texts[0].startswith("ACME Corp · Form 10-K · Item 1 — Business\n")
    chunk, _ = index.all_passages()[0]
    assert "Form 10-K" not in chunk.text


def test_doc_id_is_the_sha256_of_the_bytes_and_chunk_0_marks_a_document() -> None:
    doc_id = content_hash(b"content")

    assert doc_id == "ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
    assert make_chunk_id(doc_id, 0) == f"{doc_id}:0"


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
