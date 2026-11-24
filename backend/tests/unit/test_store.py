"""The ChunkStore contract, run against the in-memory store and against Postgres."""

import math
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import date

import pytest
from conftest import empty_database
from psycopg.errors import DataException

from groundwork.db.pool import Pool
from groundwork.domain.models import Chunk, DocumentMeta, SearchFilters
from groundwork.retrieval.indexes import IndexSpec, ensure_index
from groundwork.retrieval.memory_store import MemoryData, MemoryStore, memory_index
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.store import ChunkStore

DIMENSION = 4


def meta(
    title: str,
    *,
    collection: str = "filings",
    doc_type: str = "10-K",
    doc_date: date | None = date(2024, 1, 31),
    metadata: dict[str, str] | None = None,
    groups: tuple[str, ...] = ("everyone",),
    owner: str | None = None,
) -> DocumentMeta:
    return DocumentMeta(
        title=title,
        collection_id=collection,
        doc_type=doc_type,
        doc_date=doc_date,
        filename=f"{title}.txt",
        metadata=metadata or {},
        access_groups=groups,
        owner_id=owner,
    )


def chunks(*texts: str) -> list[Chunk]:
    # Ids and doc ids are placeholders: the store assigns them.
    return [
        Chunk("x", "x", text, ("Item 1", text[:5]), None, i, i + 1) for i, text in enumerate(texts)
    ]


def vector(*values: float) -> list[float]:
    return [*values, *[0.0] * (DIMENSION - len(values))]


# Opens another store on the same data, as another process would.
OpenStore = Callable[[], ChunkStore]


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
async def open_store(request: pytest.FixtureRequest) -> AsyncIterator[OpenStore]:
    if request.param == "memory":
        data = MemoryData()
        yield lambda: MemoryStore(memory_index(DIMENSION), data)
        return
    async with empty_database(request.getfixturevalue("database_url")) as pool:
        spec = IndexSpec(name="test", chunker="window", embedder_model="fake", dimension=DIMENSION)
        index = await ensure_index(pool, spec)
        yield lambda: PgVectorStore(pool, index)


@pytest.fixture
def store(open_store: OpenStore) -> ChunkStore:
    return open_store()


async def test_added_chunks_get_ids_from_their_document(store: ChunkStore) -> None:
    document = await store.add(
        meta("acme"), "h1", chunks("ACME a", "ACME b"), [vector(1), vector(0, 1)]
    )

    passages = await store.all_passages()

    assert [chunk.chunk_id for chunk, _ in passages] == [
        f"{document.doc_id}:0",
        f"{document.doc_id}:1",
    ]
    assert all(chunk.doc_id == document.doc_id for chunk, _ in passages)
    chunk, stored = passages[0]
    assert chunk.text == "ACME a"
    assert chunk.section_path == ("Item 1", "ACME ")
    assert (chunk.char_start, chunk.char_end) == (0, 1)
    assert stored.title == "acme"
    assert stored.doc_date == date(2024, 1, 31)


async def test_search_ranks_by_cosine_similarity(store: ChunkStore) -> None:
    await store.add(meta("acme"), "h1", chunks("east", "north"), [vector(1), vector(0, 1)])

    hits = await store.search(vector(0.2, 1), 2)

    assert [hit.chunk.text for hit in hits] == ["north", "east"]
    assert hits[0].similarity == pytest.approx(1 / (1.04**0.5), abs=1e-4)


async def test_search_returns_at_most_n(store: ChunkStore) -> None:
    await store.add(
        meta("acme"), "h1", chunks("a", "b", "c"), [vector(1), vector(1, 1), vector(0, 1)]
    )

    assert len(await store.search(vector(1), 2)) == 2


async def test_find_document_needs_the_same_bytes_collection_groups_and_owner(
    store: ChunkStore,
) -> None:
    document = await store.add(
        meta("acme", groups=("hr", "everyone")), "h1", chunks("a"), [vector(1)]
    )

    assert (
        await store.find_document("h1", meta("acme", groups=("everyone", "hr"))) == document.doc_id
    )
    assert await store.find_document("h2", meta("acme", groups=("everyone", "hr"))) is None
    assert await store.find_document("h1", meta("acme", collection="news")) is None
    assert await store.find_document("h1", meta("acme")) is None


async def test_groups_are_stored_sorted(store: ChunkStore) -> None:
    document = await store.add(
        meta("a", groups=("hr", "everyone", "hr")), "h", chunks("a"), [vector(1)]
    )

    assert document.access_groups == ("everyone", "hr")


async def test_adding_again_keeps_the_tags_and_replaces_the_chunks(store: ChunkStore) -> None:
    first = await store.add(meta("old title"), "h1", chunks("a", "b", "c"), [vector(1)] * 3)

    again = await store.add(meta("new title"), "h1", chunks("x", "y"), [vector(1)] * 2)

    assert again.doc_id == first.doc_id
    assert again.title == "old title"
    assert [chunk.text for chunk, _ in await store.all_passages()] == ["x", "y"]
    assert await store.count_chunks() == 2


async def test_one_file_in_two_collections_is_two_documents(store: ChunkStore) -> None:
    a = await store.add(meta("a"), "same", chunks("a"), [vector(1)])
    b = await store.add(meta("a", collection="news"), "same", chunks("a"), [vector(1)])

    assert a.doc_id != b.doc_id
    assert await store.count_documents() == 2


async def test_get_returns_requested_chunks_and_skips_unknown_ids(store: ChunkStore) -> None:
    document = await store.add(meta("acme"), "h1", chunks("a", "b"), [vector(1)] * 2)

    found = await store.get([f"{document.doc_id}:1", f"{document.doc_id}:9", "999:0"])

    assert list(found) == [f"{document.doc_id}:1"]
    assert found[f"{document.doc_id}:1"][0].text == "b"


async def test_filters_narrow_search_and_matching_ids(store: ChunkStore) -> None:
    acme = await store.add(
        meta("acme", metadata={"ticker": "ACME"}), "h1", chunks("a"), [vector(1)]
    )
    await store.add(
        meta(
            "beta",
            collection="news",
            doc_type="8-K",
            doc_date=date(2025, 6, 30),
            metadata={"ticker": "BETA"},
        ),
        "h2",
        chunks("b"),
        [vector(1)],
    )
    await store.add(meta("undated", doc_date=None), "h3", chunks("c"), [vector(1)])
    only_acme = f"{acme.doc_id}:0"
    cases = [
        SearchFilters(collections=("filings",), date_to=date(2024, 12, 31)),
        SearchFilters(doc_types=("10-K",), date_from=date(2024, 1, 1), date_to=date(2024, 1, 31)),
        SearchFilters(metadata={"ticker": "ACME"}),
    ]
    for filters in cases:
        hits = await store.search(vector(1), 10, filters)
        assert [hit.chunk.chunk_id for hit in hits] == [only_acme], filters
        assert await store.matching_ids(filters) == {only_acme}, filters


async def test_documents_lists_chunk_counts_by_title(store: ChunkStore) -> None:
    await store.add(meta("zeta"), "h1", chunks("a"), [vector(1)])
    await store.add(meta("alpha"), "h2", chunks("a", "b"), [vector(1)] * 2)

    listed = [(document.title, count) for document, count in await store.documents()]

    assert listed == [("alpha", 2), ("zeta", 1)]
    assert await store.count_documents() == 2


async def test_count_document_chunks(store: ChunkStore) -> None:
    document = await store.add(meta("acme"), "h1", chunks("a", "b"), [vector(1)] * 2)

    assert await store.count_document_chunks(document.doc_id) == 2
    assert await store.count_document_chunks("12345") == 0


async def test_an_empty_store_finds_nothing(store: ChunkStore) -> None:
    assert await store.search(vector(1), 5) == []
    assert await store.all_passages() == []
    assert await store.count_chunks() == 0


async def test_a_chunk_count_must_match_its_embeddings(store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="one embedding"):
        await store.add(meta("acme"), "h1", chunks("a", "b"), [vector(1)])


async def test_publishing_moves_the_generation_once_per_batch_of_writes(store: ChunkStore) -> None:
    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)])
    await store.add(meta("beta"), "h2", chunks("b"), [vector(1)])
    assert (await store.publication()).generation == 0

    assert await store.publish() == 1
    assert (await store.publication()).generation == 1


async def test_publishing_without_writes_does_nothing(store: ChunkStore) -> None:
    assert await store.publish() is None

    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)])
    await store.publish()

    assert await store.publish() is None
    assert (await store.publication()).generation == 1


async def test_a_write_is_published_by_another_store_instance(open_store: OpenStore) -> None:
    await open_store().add(meta("acme"), "h1", chunks("a"), [vector(1)])
    later = open_store()

    assert await later.publish() == 1
    assert await open_store().publish() is None
    assert (await open_store().publication()).generation == 1


async def test_a_write_can_be_published_as_it_is_stored(store: ChunkStore) -> None:
    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)], publish=True)

    assert (await store.publication()).generation == 1
    assert await store.publish() is None


async def test_publishing_a_write_publishes_the_ones_before_it(store: ChunkStore) -> None:
    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)])
    await store.add(meta("beta"), "h2", chunks("b"), [vector(1)], publish=True)

    assert (await store.publication()).generation == 1
    assert await store.publish() is None


async def test_each_publication_gets_its_own_token(store: ChunkStore) -> None:
    new = await store.publication()
    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)])
    await store.publish()
    first = await store.publication()
    await store.add(meta("beta"), "h2", chunks("b"), [vector(1)], publish=True)
    second = await store.publication()

    assert [p.generation for p in (new, first, second)] == [0, 1, 2]
    assert len({new.token, first.token, second.token}) == 3


async def test_the_token_changes_only_by_publishing(open_store: OpenStore) -> None:
    store = open_store()
    before = await store.publication()
    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)])
    assert (await store.publication()).token == before.token

    await store.publish()
    published = await store.publication()
    await store.publish()

    assert await open_store().publication() == published


async def test_a_publication_tells_whether_the_index_was_written_since(store: ChunkStore) -> None:
    assert not (await store.publication()).written_since

    await store.add(meta("acme"), "h1", chunks("a"), [vector(1)])
    assert (await store.publication()).written_since

    await store.publish()
    assert not (await store.publication()).written_since


# Corpus documents


async def test_a_corpus_document_and_an_upload_of_the_same_bytes_are_two_documents(
    store: ChunkStore,
) -> None:
    corpus = await store.add_corpus_document(
        meta("a"), "same", chunks("a"), [vector(1)], manifest_key="a"
    )
    upload = await store.add(meta("a"), "same", chunks("a"), [vector(1)])

    assert corpus.doc_id != upload.doc_id
    assert [item.document.doc_id for item in await store.corpus_documents()] == [corpus.doc_id]
    # An upload of those bytes again is the upload, not the corpus document.
    assert await store.find_document("same", meta("a")) == upload.doc_id
    assert (await store.add(meta("a"), "same", chunks("a"), [vector(1)])).doc_id == upload.doc_id


async def test_find_upload_never_finds_a_corpus_document(store: ChunkStore) -> None:
    corpus = await store.add_corpus_document(
        meta("a"), "same", chunks("a"), [vector(1)], manifest_key="a"
    )

    assert await store.find_document("same", meta("a")) == corpus.doc_id
    assert await store.find_upload("same", meta("a")) is None

    upload = await store.add(meta("a"), "same", chunks("a"), [vector(1)])

    assert await store.find_upload("same", meta("a")) == upload.doc_id


async def test_a_corpus_document_is_not_stored_beside_an_upload_of_the_same_bytes(
    store: ChunkStore,
) -> None:
    upload = await store.add(meta("a"), "same", chunks("a"), [vector(1)], publish=True)

    stored = await store.add_corpus_document(
        meta("a"), "same", chunks("a"), [vector(1)], manifest_key="a"
    )

    assert stored is None
    assert await store.documents() == [(upload, 1)]
    assert await store.publish() is None


async def test_a_corpus_document_stored_again_unchanged_keeps_its_id(store: ChunkStore) -> None:
    first = await store.add_corpus_document(
        meta("a", groups=("hr", "everyone")),
        "h",
        chunks("a", "b"),
        [vector(1)] * 2,
        manifest_key="a",
    )

    again = await store.add_corpus_document(
        meta("a", groups=("everyone", "hr")), "h", chunks("c"), [vector(1)], manifest_key="a"
    )

    assert again.doc_id == first.doc_id
    assert [chunk.text for chunk, _ in await store.all_passages()] == ["c"]


async def test_one_key_in_two_collections_is_two_documents(store: ChunkStore) -> None:
    await store.add_corpus_document(meta("a"), "h1", chunks("a"), [vector(1)], manifest_key="a")
    await store.add_corpus_document(
        meta("a", collection="news"), "h2", chunks("b"), [vector(1)], manifest_key="a"
    )

    assert await store.count_documents() == 2


async def test_a_changed_corpus_document_replaces_the_old_one_to_publish(store: ChunkStore) -> None:
    old = await store.add_corpus_document(
        meta("a"), "h1", chunks("old"), [vector(1)], manifest_key="a"
    )
    await store.publish()

    new = await store.add_corpus_document(
        meta("a"), "h2", chunks("new"), [vector(1)], manifest_key="a"
    )

    # A new id, so that a BM25 index from before cannot lead to the new text.
    assert new.doc_id != old.doc_id
    [item] = await store.corpus_documents()
    assert (item.document, item.key, item.content_hash, item.in_index) == (new, "a", "h2", True)
    assert [chunk.text for chunk, _ in await store.all_passages()] == ["new"]
    assert await store.publish() == 2


async def test_removed_corpus_documents_are_gone_and_to_publish(store: ChunkStore) -> None:
    gone = await store.add_corpus_document(
        meta("a"), "h1", chunks("a"), [vector(1)], manifest_key="a"
    )
    kept = await store.add_corpus_document(
        meta("b"), "h2", chunks("b"), [vector(1)], manifest_key="b"
    )
    await store.publish()

    await store.remove_corpus_documents([gone.doc_id, "12345"])

    assert [document for document, _ in await store.documents()] == [kept]
    assert [item.document for item in await store.corpus_documents()] == [kept]
    assert await store.get([f"{gone.doc_id}:0"]) == {}
    assert await store.publish_all() == {store.index.name: 2}


async def test_removing_corpus_documents_never_removes_an_upload(store: ChunkStore) -> None:
    upload = await store.add(meta("a"), "h1", chunks("a"), [vector(1)])
    await store.publish()

    await store.remove_corpus_documents([upload.doc_id])

    assert [document for document, _ in await store.documents()] == [upload]
    assert await store.publish() is None


async def test_removing_nothing_leaves_nothing_to_publish(store: ChunkStore) -> None:
    await store.remove_corpus_documents([])

    assert await store.publish_all() == {}


async def test_removed_documents_are_published_by_another_store_instance(
    open_store: OpenStore,
) -> None:
    store = open_store()
    gone = await store.add_corpus_document(
        meta("a"), "h1", chunks("a"), [vector(1)], manifest_key="a"
    )
    await store.publish()
    await store.remove_corpus_documents([gone.doc_id])

    assert await open_store().publish() == 2


@pytest.mark.db
async def test_a_failed_replacement_changes_no_index(pool: Pool) -> None:
    spec = IndexSpec(name="a", chunker="window", embedder_model="fake", dimension=DIMENSION)
    a = PgVectorStore(pool, await ensure_index(pool, spec))
    b = PgVectorStore(pool, await ensure_index(pool, replace(spec, name="b")))
    for store in (a, b):
        await store.add_corpus_document(
            meta("old"), "h1", chunks("old"), [vector(1)], manifest_key="a"
        )
        await store.publish()

    # The last chunk's vector has the wrong length, so its insert fails after the first's.
    with pytest.raises(DataException):
        await a.add_corpus_document(
            meta("new"), "h2", chunks("new", "er"), [vector(1), [1.0] * 5], manifest_key="a"
        )

    for store in (a, b):
        [(document, count)] = await store.documents()
        assert (document.title, count) == ("old", 1)
        assert await store.publish() is None
    [item] = await a.corpus_documents()
    assert item.content_hash == "h1"


# Dense search through the HNSW index: Postgres only.


def arc(count: int) -> list[list[float]]:
    """Distinct directions fanning out from vector(1), nearest first."""
    return [vector(math.cos(i / count), math.sin(i / count)) for i in range(count)]


async def hnsw_store(pool: Pool) -> PgVectorStore:
    spec = IndexSpec(name="test", chunker="window", embedder_model="fake", dimension=DIMENSION)
    return PgVectorStore(pool, await ensure_index(pool, spec))


async def unaided_count(pool: Pool, store: PgVectorStore, n: int) -> int:
    """How many chunks the bare query finds on the pool's connection, without the settings
    `search` adds. 40, pgvector's default, shows the query goes through the HNSW index and the
    connection still has that default: a sort would find all n."""
    sql, params = store.search_sql(vector(1), n)
    async with pool.connection() as connection:
        return len(await (await connection.execute(sql, params)).fetchall())


@pytest.mark.db
async def test_dense_search_returns_more_than_40_candidates(hnsw_pool: Pool) -> None:
    store = await hnsw_store(hnsw_pool)
    await store.add(meta("acme"), "h1", chunks(*["text"] * 200), arc(200))
    assert await unaided_count(hnsw_pool, store, 60) == 40

    hits = await store.search(vector(1), 60)

    assert len(hits) == 60


@pytest.mark.db
async def test_dense_search_asks_for_no_more_than_pgvector_allows(hnsw_pool: Pool) -> None:
    store = await hnsw_store(hnsw_pool)
    await store.add(meta("acme"), "h1", chunks(*["text"] * 200), arc(200))

    assert len(await store.search(vector(1), 1001)) == 200


@pytest.mark.db
async def test_a_large_search_does_not_change_the_next_one_on_its_connection(
    hnsw_pool: Pool,
) -> None:
    store = await hnsw_store(hnsw_pool)
    await store.add(meta("acme"), "h1", chunks(*["text"] * 200), arc(200))

    assert len(await store.search(vector(1), 60)) == 60

    assert await unaided_count(hnsw_pool, store, 60) == 40


@pytest.mark.db
async def test_a_filtered_search_scans_on_until_it_has_more_than_40_candidates(
    hnsw_pool: Pool,
) -> None:
    store = await hnsw_store(hnsw_pool)
    directions = arc(300)
    await store.add(meta("near"), "h1", chunks(*["near"] * 200), directions[:200])
    await store.add(meta("far", collection="news"), "h2", chunks(*["far"] * 100), directions[200:])

    hits = await store.search(vector(1), 60, SearchFilters(collections=("news",)))

    assert [hit.chunk.text for hit in hits] == ["far"] * 60
