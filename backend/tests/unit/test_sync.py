"""Corpus sync, through manifests on disk and the real retriever: what a search finds after it."""

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from collection_builder import doc, write_collection
from conftest import empty_database

from groundwork.db.pool import Pool
from groundwork.domain.models import DocumentMeta, ScoredPassage, SearchFilters, content_hash
from groundwork.ingestion.corpus import corpus_entries
from groundwork.ingestion.loaders import loader_for
from groundwork.ingestion.pipeline import ingest_file, prepare_document
from groundwork.ingestion.sync import SyncReport, sync_corpus
from groundwork.ingestion.uploads import UploadService
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.hybrid import HybridRetriever
from groundwork.retrieval.indexes import IndexSpec, ensure_index
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.sparse import SparseCache
from groundwork.retrieval.store import ChunkStore

DIMENSION = FakeEmbedder().dimension


def about(subject: str) -> str:
    return f"The runbook explains {subject} to whoever is on call tonight. " * 30


class Library:
    """One index as a worker of the app searches it."""

    def __init__(self, store: ChunkStore, bm25: Path) -> None:
        self.store = store
        self.uploads_dir = bm25.parent / "uploads"
        sparse = SparseCache(store, bm25, check_seconds=0)
        self.retriever = HybridRetriever(FakeEmbedder(), store, sparse, candidates=200, rrf_k=60)

    async def sync(self, raw: Path, embedder: FakeEmbedder | None = None) -> SyncReport:
        return await sync_corpus(
            corpus_entries(raw),
            embedder or FakeEmbedder(),
            self.store,
            chunk_tokens=100,
            chunk_overlap=10,
        )

    async def upload(self, name: str, text: str, collection: str = "uploads") -> str | None:
        """Upload a file through the service the API uses. Its document's id."""
        service = UploadService(
            FakeEmbedder(),
            self.store,
            uploads_dir=self.uploads_dir,
            max_bytes=1_000_000,
            chunk_tokens=100,
            chunk_overlap=10,
        )
        scheduled: list[Callable[[], Awaitable[None]]] = []
        job = await service.accept(
            name,
            text.encode(),
            collection_id=collection,
            doc_type="notes",
            doc_date=None,
            schedule=scheduled.append,
        )
        for run in scheduled:
            await run()
        finished = service.job(job.job_id)
        assert finished is not None and finished.status == "indexed"
        return finished.doc_id

    async def search(
        self, word: str, filters: SearchFilters | None = None
    ) -> dict[str, list[ScoredPassage]]:
        """The passages with the word in their text, by retriever. Dense search ranks every chunk,
        so without this it would find everything."""
        return {
            mode: [
                passage
                for passage in await self.retriever.search(word, 200, filters, mode)  # type: ignore[arg-type]
                if word in passage.chunk.text
            ]
            for mode in ("dense", "sparse")
        }

    async def find(self, word: str, filters: SearchFilters | None = None) -> list[str]:
        """The titles of the documents with the word, which both retrievers must agree on."""
        found = await self.search(word, filters)
        titles = {
            mode: sorted({(p.document.doc_id, p.document.title) for p in passages})
            for mode, passages in found.items()
        }
        assert titles["dense"] == titles["sparse"]
        return [title for _, title in titles["dense"]]


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
async def library(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Library]:
    if request.param == "memory":
        yield Library(MemoryStore(memory_index(DIMENSION)), tmp_path / "bm25")
        return
    async with empty_database(request.getfixturevalue("database_url")) as pool:
        yield await open_library(pool, tmp_path)


async def open_library(pool: Pool, tmp_path: Path, name: str = "test") -> Library:
    spec = IndexSpec(name=name, chunker="window", embedder_model="fake", dimension=DIMENSION)
    return Library(PgVectorStore(pool, await ensure_index(pool, spec)), tmp_path / "bm25")


async def test_new_documents_are_found_by_both_retrievers(library: Library, tmp_path: Path) -> None:
    write_collection(
        tmp_path / "v1",
        "runbooks",
        [doc("pager.md", title="Pager"), doc("disk.md", title="Disk")],
        texts={"pager.md": about("pagers"), "disk.md": about("disks")},
    )

    report = await library.sync(tmp_path / "v1")

    assert report.names("added") == ["runbooks/pager-md", "runbooks/disk-md"]
    assert await library.find("pagers") == ["Pager"]
    assert await library.find("disks") == ["Disk"]


class CountingEmbedder(FakeEmbedder):
    calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_documents(texts)


class RecordingEmbedder(FakeEmbedder):
    texts: Sequence[str] = ()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts = texts
        return super().embed_documents(texts)


def write(raw: Path, texts: dict[str, str], collection: str = "runbooks", **tags: Any) -> Path:
    """A collection of one document per text, all with the same tags."""
    documents = [doc(name, title=name.split(".")[0], **tags) for name in texts]
    write_collection(raw, collection, documents, texts=texts)
    return raw


async def test_unchanged_documents_are_not_embedded_again(library: Library, tmp_path: Path) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})
    await library.sync(raw)
    embedder = CountingEmbedder()

    report = await library.sync(raw, embedder)

    assert embedder.calls == 0
    assert report.names("skipped") == ["runbooks/pager-md", "runbooks/disk-md"]
    assert await library.find("pagers") == ["pager"]


async def test_one_sync_moves_the_generation_once_and_a_rerun_does_not(
    library: Library, tmp_path: Path
) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})

    assert list((await library.sync(raw)).published.values()) == [1]

    assert (await library.sync(raw)).published == {}
    assert (await library.store.publication()).generation == 1


async def test_a_changed_file_is_replaced(library: Library, tmp_path: Path) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    await library.find("pagers")

    report = await library.sync(write(tmp_path / "v2", {"pager.md": about("phones")}))

    assert report.names("replaced") == ["runbooks/pager-md"]
    assert await library.find("pagers") == []
    assert await library.find("phones") == ["pager"]
    assert (await library.store.publication()).generation == 2


@pytest.mark.parametrize(
    "tags",
    [
        {"title": "Renamed"},
        {"doc_type": "policy"},
        {"doc_date": "2025-06-30"},
        {"metadata": {"team": "core"}},
        {"access_groups": ["hr"]},
    ],
)
async def test_a_document_whose_tags_changed_is_replaced(
    library: Library, tmp_path: Path, tags: dict[str, Any]
) -> None:
    texts = {"pager.md": about("pagers")}
    before = doc("pager.md", title="Pager", doc_date="2024-01-31", metadata={"team": "web"})
    write_collection(tmp_path / "v1", "runbooks", [before], texts=texts)
    write_collection(tmp_path / "v2", "runbooks", [{**before, **tags}], texts=texts)
    await library.sync(tmp_path / "v1")
    embedder = RecordingEmbedder()

    report = await library.sync(tmp_path / "v2", embedder)

    assert report.names("replaced") == ["runbooks/pager-md"]
    [(document, _)] = await library.store.documents()
    stored = {
        "title": document.title,
        "doc_type": document.doc_type,
        "doc_date": document.doc_date and document.doc_date.isoformat(),
        "metadata": dict(document.metadata),
        "access_groups": list(document.access_groups),
    }
    assert {key: stored[key] for key in tags} == tags
    assert await library.find("pagers") == [document.title]
    # The header that is searched, not only the stored one.
    assert embedder.texts[0].startswith(document.title)


async def test_filters_match_the_new_tags_only(library: Library, tmp_path: Path) -> None:
    texts = {"pager.md": about("pagers")}
    await library.sync(write(tmp_path / "v1", texts, doc_type="runbook"))

    await library.sync(write(tmp_path / "v2", texts, doc_type="policy"))

    assert await library.find("pagers", SearchFilters(doc_types=("runbook",))) == []
    assert await library.find("pagers", SearchFilters(doc_types=("policy",))) == ["pager"]


async def test_groups_in_another_order_are_no_change(library: Library, tmp_path: Path) -> None:
    texts = {"pager.md": about("pagers")}
    await library.sync(write(tmp_path / "v1", texts, access_groups=["hr", "everyone"]))

    report = await library.sync(
        write(tmp_path / "v2", texts, access_groups=["everyone", "hr", "hr"])
    )

    assert report.names("skipped") == ["runbooks/pager-md"]


async def test_a_document_that_left_the_corpus_is_removed(library: Library, tmp_path: Path) -> None:
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})
    await library.sync(v1)
    await library.find("pagers")

    report = await library.sync(write(tmp_path / "v2", {"disk.md": about("disks")}))

    assert report.names("removed") == ["runbooks/pager-md"]
    assert await library.find("pagers") == []
    assert await library.find("disks") == ["disk"]


async def test_the_documents_of_a_dropped_collection_are_removed(
    library: Library, tmp_path: Path
) -> None:
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers")})
    write(v1, {"disk.md": about("disks")}, collection="storage")
    await library.sync(v1)

    report = await library.sync(write(tmp_path / "v2", {"pager.md": about("pagers")}))

    assert report.names("removed") == ["storage/disk-md"]
    assert await library.find("disks") == []
    assert await library.find("pagers") == ["pager"]


async def test_a_sync_with_no_entries_is_refused(library: Library, tmp_path: Path) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))

    with pytest.raises(ValueError, match="no corpus documents"):
        await library.sync(tmp_path / "missing")

    assert await library.find("pagers") == ["pager"]


async def test_a_sync_with_two_documents_of_one_key_is_refused(
    library: Library, tmp_path: Path
) -> None:
    entries = corpus_entries(write(tmp_path / "v1", {"pager.md": about("pagers")}))

    with pytest.raises(ValueError, match="runbooks/pager-md"):
        await sync_corpus(
            entries * 2, FakeEmbedder(), library.store, chunk_tokens=100, chunk_overlap=10
        )


async def test_text_moved_to_another_document_as_its_own_changes_is_found_there(
    library: Library, tmp_path: Path
) -> None:
    """As when a policy is archived under a new key and its key gets new text: the archived
    document now has the identity the current one had until this sync."""
    v1 = {"archive.md": about("telegrams"), "current.md": about("pagers")}
    await library.sync(write(tmp_path / "v1", v1))

    v2 = {"archive.md": about("pagers"), "current.md": about("phones")}
    await library.sync(write(tmp_path / "v2", v2))

    assert await library.find("telegrams") == []
    assert await library.find("pagers") == ["archive"]
    assert await library.find("phones") == ["current"]


async def test_a_renamed_key_leaves_one_document(library: Library, tmp_path: Path) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))

    report = await library.sync(write(tmp_path / "v2", {"beeper.md": about("pagers")}))

    assert report.names("removed") == ["runbooks/pager-md"]
    assert report.names("added") == ["runbooks/beeper-md"]
    assert await library.find("pagers") == ["beeper"]


async def test_two_documents_with_the_same_bytes_are_both_found(
    library: Library, tmp_path: Path
) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers"), "copy.md": about("pagers")})
    await library.sync(raw)

    report = await library.sync(raw)

    assert report.names("skipped") == ["runbooks/pager-md", "runbooks/copy-md"]
    assert await library.find("pagers") == ["pager", "copy"]


async def test_a_document_left_with_too_little_text_is_removed(
    library: Library, tmp_path: Path
) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))

    report = await library.sync(write(tmp_path / "v2", {"pager.md": "Too short."}))

    assert report.names("removed") == ["runbooks/pager-md"]
    assert await library.find("pagers") == []
    again = await library.sync(tmp_path / "v2")
    assert again.names("too little text") == ["runbooks/pager-md"]
    assert await library.store.count_documents() == 0


# Uploads


async def test_uploads_are_kept_also_in_a_corpus_collection(
    library: Library, tmp_path: Path
) -> None:
    await library.upload("notes.txt", about("notes"), "runbooks")
    await library.upload("other.txt", about("errands"))
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    v2 = write(tmp_path / "v2", {"disk.md": about("disks")}, collection="storage")

    report = await library.sync(v2)

    assert report.names("removed") == ["runbooks/pager-md"]
    assert await library.find("notes") == ["notes.txt"]
    assert await library.find("errands") == ["other.txt"]


async def test_an_upload_with_a_corpus_documents_bytes_stands_in_for_it(
    library: Library, tmp_path: Path
) -> None:
    """Same bytes, collection, groups and owner: left as it is, and never removed."""
    doc_id = await library.upload("mine.md", about("pagers"), "runbooks")
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})

    report = await library.sync(v1)

    assert report.names("skipped") == ["runbooks/pager-md"]
    assert await library.find("pagers") == ["mine.md"]

    report = await library.sync(write(tmp_path / "v2", {"disk.md": about("disks")}))

    assert report.names("removed") == []
    [passage, *_] = (await library.search("pagers"))["sparse"]
    assert (passage.document.doc_id, passage.document.title) == (doc_id, "mine.md")


async def test_a_document_changed_into_an_upload_leaves_only_the_upload(
    library: Library, tmp_path: Path
) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    await library.upload("mine.md", about("phones"), "runbooks")

    report = await library.sync(write(tmp_path / "v2", {"pager.md": about("phones")}))

    assert report.names("removed") == ["runbooks/pager-md"]
    assert await library.find("pagers") == []
    assert await library.find("phones") == ["mine.md"]


async def test_an_upload_for_other_readers_does_not_stand_in_for_a_document(
    library: Library, tmp_path: Path
) -> None:
    texts = {"pager.md": about("pagers")}
    await library.upload("mine.md", about("pagers"), "runbooks")

    await library.sync(write(tmp_path / "v1", texts, access_groups=["hr"]))
    assert await library.find("pagers") == ["mine.md", "pager"]

    await library.sync(write(tmp_path / "v2", texts, access_groups=["finance"]))
    assert await library.find("pagers") == ["mine.md", "pager"]


async def test_an_upload_of_a_corpus_document_removed_since_is_stored_again(
    library: Library, tmp_path: Path
) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    await library.sync(write(tmp_path / "v2", {"disk.md": about("disks")}))

    await library.upload("mine.md", about("pagers"), "runbooks")

    assert await library.find("pagers") == ["mine.md"]


async def test_files_ingested_without_a_key_are_not_removed(
    library: Library, tmp_path: Path
) -> None:
    """As the eval scripts ingest them, and as every document was before keys."""
    [entry] = corpus_entries(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    await ingest_file(
        entry.path, entry.meta, FakeEmbedder(), library.store, chunk_tokens=100, chunk_overlap=10
    )

    await library.sync(write(tmp_path / "v2", {"disk.md": about("disks")}))

    assert await library.find("pagers") == ["pager"]


async def store_duplicate(library: Library, raw: Path) -> str:
    """Store the only corpus document of `raw` as an upload too, as when both are stored at once
    and neither can see the other. The upload's document id."""
    [entry] = corpus_entries(raw)
    content = entry.path.read_bytes()
    prepared = prepare_document(
        loader_for(entry.path.name)(content),
        entry.meta,
        "x",
        FakeEmbedder(),
        chunk_tokens=100,
        chunk_overlap=10,
    )
    meta = replace(entry.meta, title="mine.md")
    upload = await library.store.add(
        meta, content_hash(content), prepared.chunks, prepared.embeddings, publish=True
    )
    return upload.doc_id


async def test_a_duplicate_of_an_upload_is_settled_by_the_next_sync(
    library: Library, tmp_path: Path
) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers")})
    await library.sync(raw)
    doc_id = await store_duplicate(library, raw)
    assert await library.find("pagers") == ["pager", "mine.md"]

    report = await library.sync(raw)

    assert report.names("removed") == ["runbooks/pager-md"]
    assert await library.find("pagers") == ["mine.md"]
    [(document, _)] = await library.store.documents()
    assert document.doc_id == doc_id

    again = await library.sync(raw)

    assert again.names("skipped") == ["runbooks/pager-md"]
    assert again.published == {}


# Access groups


async def test_a_document_moved_to_another_group_is_not_found_through_its_old_ones(
    library: Library, tmp_path: Path
) -> None:
    texts = {"pager.md": about("pagers")}
    await library.sync(write(tmp_path / "v1", texts, access_groups=["everyone"]))
    old = [chunk.chunk_id for chunk, _ in await library.store.all_passages()]
    [old_document] = [document for document, _ in await library.store.documents()]
    await library.find("pagers")

    await library.sync(write(tmp_path / "v2", texts, access_groups=["hr"]))

    store = library.store
    another_worker = Library(store, library.uploads_dir.parent / "bm25")
    for worker in (library, another_worker):
        for mode in ("reranked", "hybrid", "dense", "sparse"):
            found = await worker.retriever.search("pagers", 200, None, mode)  # type: ignore[arg-type]
            assert found
            assert {p.document.access_groups for p in found} == {("hr",)}
    assert {d.access_groups for d, _ in await store.documents()} == {("hr",)}
    assert {d.access_groups for _, d in await store.all_passages()} == {("hr",)}
    assert await store.count_documents() == 1
    assert await store.get(old) == {}
    assert await store.count_document_chunks(old_document.doc_id) == 0
    assert not await store.matching_ids(SearchFilters()) & set(old)
    # What an upload asks before it says "already indexed".
    digest = content_hash(texts["pager.md"].encode())
    assert await store.find_document(digest, old_document) is None
    uploaded = await library.upload("mine.md", texts["pager.md"], "runbooks")
    assert uploaded not in {d.doc_id for d, _ in await store.documents() if "hr" in d.access_groups}


async def test_a_replaced_document_is_gone_before_bm25_is_rebuilt(
    library: Library, tmp_path: Path
) -> None:
    """A worker's BM25 is from before the replacement until it is published, so it still finds the
    old chunks' ids. They must lead nowhere, also not to the new document's text."""
    texts = {"pager.md": about("pagers")}
    await library.sync(write(tmp_path / "v1", texts, access_groups=["everyone"]))
    assert await library.find("pagers") == ["pager"]
    [(chunk, _), *_] = await library.store.all_passages()
    moved = DocumentMeta(
        title="pager",
        collection_id="runbooks",
        doc_type="runbook",
        doc_date=None,
        filename="pager.md",
        access_groups=("hr",),
    )

    await library.store.add_corpus_document(
        moved, "h2", [chunk], [[1.0] * DIMENSION], manifest_key="pager-md"
    )

    assert await library.retriever.search("pagers", 10, None, "sparse") == []
    dense = await library.retriever.search("pagers", 10, None, "dense")
    assert {p.document.access_groups for p in dense} == {("hr",)}


# Failures


class FailsOn(FakeEmbedder):
    def __init__(self, word: str) -> None:
        super().__init__()
        self._word = word

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any(self._word in text for text in texts):
            raise RuntimeError("out of memory")
        return super().embed_documents(texts)


async def test_after_a_sync_fails_partway_and_is_rerun_every_document_is_found(
    library: Library, tmp_path: Path
) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})

    with pytest.raises(RuntimeError, match="out of memory"):
        await library.sync(raw, FailsOn("disks"))

    assert await library.find("pagers") == ["pager"]
    assert (await library.store.publication()).generation == 1

    report = await library.sync(raw)

    assert report.names("skipped") == ["runbooks/pager-md"]
    assert report.names("added") == ["runbooks/disk-md"]
    assert await library.find("pagers") == ["pager"]
    assert await library.find("disks") == ["disk"]
    assert (await library.store.publication()).generation == 2


async def test_a_failure_to_embed_keeps_the_old_document(library: Library, tmp_path: Path) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))

    with pytest.raises(RuntimeError, match="out of memory"):
        await library.sync(write(tmp_path / "v2", {"pager.md": about("phones")}), FailsOn("phones"))

    assert await library.find("pagers") == ["pager"]


async def test_documents_that_left_are_gone_even_if_the_sync_fails_after(
    library: Library, tmp_path: Path
) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    await library.find("pagers")

    with pytest.raises(RuntimeError, match="out of memory"):
        await library.sync(write(tmp_path / "v2", {"disk.md": about("disks")}), FailsOn("disks"))

    assert await library.find("pagers") == []
    assert (await library.store.publication()).generation == 2


async def test_what_a_killed_sync_stored_is_published_by_the_next_one(
    library: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers")})
    # Saves an empty BM25 index, as an API running before the ingest does.
    assert await library.find("pagers") == []

    async def killed() -> dict[str, int]:
        raise RuntimeError("the process died")

    with monkeypatch.context() as patch, pytest.raises(RuntimeError, match="died"):
        patch.setattr(library.store, "publish_all", killed)
        await library.sync(raw)

    report = await library.sync(raw)

    assert list(report.published.values()) == [1]
    assert await library.find("pagers") == ["pager"]


# Several indexes on one database


@pytest.fixture
async def two_libraries(pool: Pool, tmp_path: Path) -> tuple[Library, Library]:
    return await open_library(pool, tmp_path, "a"), await open_library(pool, tmp_path, "b")


@pytest.mark.db
async def test_a_changed_or_removed_document_is_gone_from_every_index(
    two_libraries: tuple[Library, Library], tmp_path: Path
) -> None:
    a, b = two_libraries
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})
    await a.sync(v1)
    await b.sync(v1)
    assert await b.find("pagers") == ["pager"]
    v2 = write(tmp_path / "v2", {"pager.md": about("phones"), "cpu.md": about("processors")})

    report = await a.sync(v2)

    assert report.published == {"a": 2, "b": 2}
    for library in (a, b):
        assert await library.find("pagers") == []
        assert await library.find("disks") == []
    assert await a.find("phones") == ["pager"]
    # The other index gets the new text at its own next sync.
    assert await b.find("phones") == []

    report = await b.sync(v2)

    assert report.names("added") == ["runbooks/pager-md", "runbooks/cpu-md"]
    assert report.published == {"b": 3}
    for library in (a, b):
        assert await library.find("phones") == ["pager"]
        assert await library.find("processors") == ["cpu"]
    assert (await a.sync(v2)).published == {}


@pytest.mark.db
async def test_indexes_that_lost_chunks_are_published_after_a_failure_too(
    two_libraries: tuple[Library, Library], tmp_path: Path
) -> None:
    a, b = two_libraries
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})
    await a.sync(v1)
    await b.sync(v1)
    await b.find("pagers")
    v2 = write(tmp_path / "v2", {"pager.md": about("phones"), "disk.md": about("platters")})

    with pytest.raises(RuntimeError, match="out of memory"):
        await a.sync(v2, FailsOn("platters"))

    assert ((await a.store.publication()).generation, (await b.store.publication()).generation) == (
        2,
        2,
    )
    assert await b.find("pagers") == []
    assert await b.find("disks") == ["disk"]


@pytest.mark.db
async def test_a_document_is_stored_in_an_index_whose_uploads_do_not_include_it(
    two_libraries: tuple[Library, Library], tmp_path: Path
) -> None:
    a, b = two_libraries
    await a.upload("mine.md", about("pagers"), "runbooks")
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers")})

    await a.sync(v1)
    await b.sync(v1)
    await a.sync(v1)

    assert await a.find("pagers") == ["mine.md"]
    assert await b.find("pagers") == ["pager"]

    await b.sync(write(tmp_path / "v2", {"disk.md": about("disks")}))

    assert await a.find("pagers") == ["mine.md"]
    assert await b.find("pagers") == []


@pytest.mark.db
async def test_an_index_without_the_upload_gets_a_settled_duplicate_back(
    two_libraries: tuple[Library, Library], tmp_path: Path
) -> None:
    """Settling takes the corpus document out of every index. One the upload is not in stores it
    again at its own next sync, and the two indexes stay that way."""
    a, b = two_libraries
    raw = write(tmp_path / "v1", {"pager.md": about("pagers")})
    await a.sync(raw)
    await b.sync(raw)
    await store_duplicate(a, raw)

    report = await a.sync(raw)

    assert report.names("removed") == ["runbooks/pager-md"]
    assert list(report.published) == ["a", "b"]
    assert await a.find("pagers") == ["mine.md"]
    assert await b.find("pagers") == []

    report = await b.sync(raw)

    assert report.names("added") == ["runbooks/pager-md"]
    for library in (a, b):
        assert (await library.sync(raw)).names("skipped") == ["runbooks/pager-md"]
    assert await a.find("pagers") == ["mine.md"]
    assert await b.find("pagers") == ["pager"]


# Syncs and uploads at the same time


class Gate(FakeEmbedder):
    """Stops where it embeds, until opened."""

    def __init__(self) -> None:
        super().__init__()
        self.reached = threading.Event()
        self.opened = threading.Event()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.reached.set()
        assert self.opened.wait(10)
        return super().embed_documents(texts)


async def test_two_syncs_at_once_store_each_document_once(library: Library, tmp_path: Path) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})
    embedder = CountingEmbedder()

    first, second = await asyncio.gather(library.sync(raw, embedder), library.sync(raw, embedder))

    assert len(first.names("added")) + len(second.names("added")) == 2
    assert embedder.calls == 2
    assert await library.find("pagers") == ["pager"]
    assert (await library.store.publication()).generation == 1


async def test_a_sync_waits_for_the_one_before_it(library: Library, tmp_path: Path) -> None:
    """Two corpora versions at once never mix: the index ends as the later sync describes it."""
    v1 = write(tmp_path / "v1", {"pager.md": about("pagers"), "disk.md": about("disks")})
    v2 = write(tmp_path / "v2", {"pager.md": about("phones")})
    gate = Gate()
    first = asyncio.create_task(library.sync(v1, gate))
    await asyncio.to_thread(gate.reached.wait, 10)

    second = asyncio.create_task(library.sync(v2))
    done, _ = await asyncio.wait([second], timeout=0.2)
    assert not done
    gate.opened.set()
    await asyncio.gather(first, second)

    assert await library.find("pagers") == []
    assert await library.find("disks") == []
    assert await library.find("phones") == ["pager"]


async def test_an_upload_is_stored_while_a_sync_runs(library: Library, tmp_path: Path) -> None:
    gate = Gate()
    sync = asyncio.create_task(
        library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}), gate)
    )
    await asyncio.to_thread(gate.reached.wait, 10)

    doc_id = await library.upload("notes.txt", about("notes"), "runbooks")
    assert await library.find("notes") == ["notes.txt"]
    gate.opened.set()
    await sync

    assert await library.find("pagers") == ["pager"]
    [passage, *_] = (await library.search("notes"))["sparse"]
    assert passage.document.doc_id == doc_id


async def test_an_upload_of_a_document_being_replaced_is_never_lost(
    library: Library, tmp_path: Path
) -> None:
    """The upload is of the text the document is about to get."""
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    gate = Gate()
    v2 = write(tmp_path / "v2", {"pager.md": about("phones")})
    sync = asyncio.create_task(library.sync(v2, gate))
    await asyncio.to_thread(gate.reached.wait, 10)

    await library.upload("mine.md", about("phones"), "runbooks")
    gate.opened.set()
    await sync
    await library.sync(write(tmp_path / "v3", {"disk.md": about("disks")}))

    assert await library.find("phones") == ["mine.md"]


async def test_an_upload_stored_while_its_corpus_document_embeds_stands_in_for_it(
    library: Library, tmp_path: Path
) -> None:
    raw = write(tmp_path / "v1", {"pager.md": about("pagers")})
    gate = Gate()
    sync = asyncio.create_task(library.sync(raw, gate))
    await asyncio.to_thread(gate.reached.wait, 10)

    doc_id = await library.upload("mine.md", about("pagers"), "runbooks")
    gate.opened.set()
    report = await sync

    assert report.names("skipped") == ["runbooks/pager-md"]
    assert await library.find("pagers") == ["mine.md"]
    [passage, *_] = (await library.search("pagers"))["sparse"]
    assert passage.document.doc_id == doc_id


async def test_an_upload_stored_while_a_replacement_embeds_stands_in_for_it(
    library: Library, tmp_path: Path
) -> None:
    await library.sync(write(tmp_path / "v1", {"pager.md": about("pagers")}))
    gate = Gate()
    v2 = write(tmp_path / "v2", {"pager.md": about("phones")})
    sync = asyncio.create_task(library.sync(v2, gate))
    await asyncio.to_thread(gate.reached.wait, 10)

    await library.upload("mine.md", about("phones"), "runbooks")
    gate.opened.set()
    report = await sync

    assert report.names("removed") == ["runbooks/pager-md"]
    assert await library.find("pagers") == []
    assert await library.find("phones") == ["mine.md"]


@pytest.mark.db
async def test_replacing_documents_beside_uploads_does_not_deadlock(
    two_libraries: tuple[Library, Library], tmp_path: Path
) -> None:
    a, b = two_libraries
    names = [f"doc{i}.md" for i in range(8)]
    v1 = write(tmp_path / "v1", {name: about(f"{name} pagers") for name in names})
    await a.sync(v1)
    await b.sync(v1)
    v2 = write(tmp_path / "v2", {name: about(f"{name} phones") for name in names})

    await asyncio.gather(
        a.sync(v2),
        *(a.upload(f"a{i}.txt", about(f"a{i} notes"), "runbooks") for i in range(8)),
        *(b.upload(f"b{i}.txt", about(f"b{i} notes"), "fresh") for i in range(8)),
    )

    assert len(await a.find("phones")) == 8
    assert await b.find("pagers") == []
    assert len(await a.find("notes")) == len(await b.find("notes")) == 8
