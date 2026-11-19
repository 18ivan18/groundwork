"""The ChunkStore contract, run against the in-memory store and against Postgres."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import date
from urllib.parse import quote

import pytest
from conftest import empty_database

from groundwork.db.pool import open_pool
from groundwork.domain.models import (
    UNRESTRICTED,
    Access,
    Chunk,
    DocumentChanges,
    DocumentMeta,
    SearchFilters,
)
from groundwork.retrieval.indexes import IndexSpec, ensure_index, hnsw_index_name
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.store import ChunkStore, DuplicateDocumentError

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


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[ChunkStore]:
    if request.param == "memory":
        yield MemoryStore(memory_index(DIMENSION))
        return
    async with empty_database(request.getfixturevalue("database_url")) as pool:
        spec = IndexSpec(name="test", chunker="window", embedder_model="fake", dimension=DIMENSION)
        yield PgVectorStore(pool, await ensure_index(pool, spec))


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

    hits = await store.search(vector(0.2, 1), 2, access=UNRESTRICTED)

    assert [hit.chunk.text for hit in hits] == ["north", "east"]
    assert hits[0].similarity == pytest.approx(1 / (1.04**0.5), abs=1e-4)


async def test_search_returns_at_most_n(store: ChunkStore) -> None:
    await store.add(
        meta("acme"), "h1", chunks("a", "b", "c"), [vector(1), vector(1, 1), vector(0, 1)]
    )

    assert len(await store.search(vector(1), 2, access=UNRESTRICTED)) == 2


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

    found = await store.get(
        [f"{document.doc_id}:1", f"{document.doc_id}:9", "999:0"], access=UNRESTRICTED
    )

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
        hits = await store.search(vector(1), 10, filters, access=UNRESTRICTED)
        assert [hit.chunk.chunk_id for hit in hits] == [only_acme], filters
        assert await store.matching_ids(filters) == {only_acme}, filters


async def test_documents_lists_chunk_counts_by_title(store: ChunkStore) -> None:
    await store.add(meta("zeta"), "h1", chunks("a"), [vector(1)])
    await store.add(meta("alpha"), "h2", chunks("a", "b"), [vector(1)] * 2)

    listed = [
        (document.title, count) for document, count in await store.documents(access=UNRESTRICTED)
    ]

    assert listed == [("alpha", 2), ("zeta", 1)]
    assert await store.count_documents() == 2


async def test_count_document_chunks(store: ChunkStore) -> None:
    document = await store.add(meta("acme"), "h1", chunks("a", "b"), [vector(1)] * 2)

    assert await store.count_document_chunks(document.doc_id) == 2
    assert await store.count_document_chunks("12345") == 0


async def test_one_document_is_found_only_by_who_may_read_it(store: ChunkStore) -> None:
    hr_meta = replace(meta("pay", groups=("hr",)), storage_path="uploads/pay.txt")
    document = await store.add(hr_meta, "h1", chunks("a", "b"), [vector(1)] * 2)

    found = await store.document(document.doc_id, access=Access.of({"hr"}))

    assert found is not None and found[1] == 2
    assert found[0].storage_path == "uploads/pay.txt"
    assert await store.document(document.doc_id, access=Access.of({"everyone"})) is None
    assert await store.document("999", access=UNRESTRICTED) is None
    assert await store.document("x", access=UNRESTRICTED) is None
    assert await store.document_indexes(document.doc_id) == [store.index.name]


async def test_an_edit_changes_what_searches_see(store: ChunkStore) -> None:
    document = await store.add(meta("acme", groups=("hr",)), "h1", chunks("a"), [vector(1)])
    everyone = Access.of({"everyone"})

    edited = await store.update_document(
        document.doc_id,
        DocumentChanges(title="ACME", doc_date=date(2025, 1, 1), access_groups=("everyone",)),
    )

    assert (edited.title, edited.doc_date, edited.access_groups) == (
        "ACME",
        date(2025, 1, 1),
        ("everyone",),
    )
    assert edited.doc_type == "10-K"
    [hit] = await store.search(
        vector(1), 5, SearchFilters(date_from=date(2025, 1, 1)), access=everyone
    )
    assert hit.document.title == "ACME"
    hr_only = await store.search(vector(1), 5, access=Access.of({"hr"}))
    assert hr_only == []


async def test_an_edit_cannot_make_two_documents_one(store: ChunkStore) -> None:
    await store.add(meta("acme", groups=("everyone",)), "h1", chunks("a"), [vector(1)])
    hr = await store.add(meta("acme", groups=("hr",)), "h1", chunks("a"), [vector(1)])

    with pytest.raises(DuplicateDocumentError):
        await store.update_document(hr.doc_id, DocumentChanges(access_groups=("everyone",)))


async def test_a_deleted_document_is_gone_everywhere(store: ChunkStore) -> None:
    kept = await store.add(meta("kept"), "h1", chunks("a"), [vector(1)])
    gone = await store.add(
        replace(meta("gone"), storage_path="uploads/gone.txt"), "h2", chunks("b"), [vector(1)]
    )

    await store.delete_document(gone.doc_id)

    assert [d.doc_id for d, _ in await store.documents(access=UNRESTRICTED)] == [kept.doc_id]
    assert [
        hit.document.doc_id for hit in await store.search(vector(1), 5, access=UNRESTRICTED)
    ] == [kept.doc_id]
    assert await store.get([f"{gone.doc_id}:0"], access=UNRESTRICTED) == {}
    assert not await store.storage_path_in_use("uploads/gone.txt")


async def test_collections_can_be_protected(store: ChunkStore) -> None:
    await store.protect_collection("sec-10k")
    await store.protect_collection("sec-10k")

    assert await store.protected_collections() == {"sec-10k"}


async def test_an_empty_store_finds_nothing(store: ChunkStore) -> None:
    assert await store.search(vector(1), 5, access=UNRESTRICTED) == []
    assert await store.all_passages() == []
    assert await store.count_chunks() == 0


async def test_a_chunk_count_must_match_its_embeddings(store: ChunkStore) -> None:
    with pytest.raises(ValueError, match="one embedding"):
        await store.add(meta("acme"), "h1", chunks("a", "b"), [vector(1)])


def graph_only(database_url: str) -> str:
    """The test database, with every session forced through the HNSW index and searching only
    10 neighbours, as a much larger table would be. Without a sort, the btree on index_id cannot
    order by distance, so it cannot compete whatever the statistics say."""
    options = quote(
        "-c enable_seqscan=off -c enable_bitmapscan=off -c enable_sort=off -c hnsw.ef_search=10"
    )
    return f"{database_url}{'&' if '?' in database_url else '?'}options={options}"


@pytest.mark.db
async def test_a_restrictive_filter_still_returns_n_rows(database_url: str) -> None:
    """#39. 200 chunks everyone may read sit next to the query; 10 hr chunks are further away.
    HNSW finds its ef_search nearest neighbours and filters after: an hr-only user gets fewer
    than 10 rows unless pgvector keeps scanning (hnsw.iterative_scan)."""
    hr_only = Access.of({"hr"})
    query = [1.0, 0.0, 0.0, 0.0]
    async with empty_database(database_url) as setup:
        spec = IndexSpec(name="acl", chunker="window", embedder_model="fake", dimension=DIMENSION)
        index = await ensure_index(setup, spec)
        writer = PgVectorStore(setup, index)
        near = [[1.0, 0.01 * i, 0.001 * (i % 7), 0.0] for i in range(200)]
        await writer.add(meta("everyone"), "e", chunks(*["e"] * 200), near)
        far = [[0.3, 1.0, 0.01 * i, 0.0] for i in range(10)]
        await writer.add(meta("salaries", groups=("hr",)), "h", chunks(*["h"] * 10), far)

    pool = await open_pool(graph_only(database_url))
    try:
        store = PgVectorStore(pool, index)
        sql, params = store.search_sql(query, 10, None, hr_only)
        async with pool.connection() as connection, connection.transaction():
            await connection.execute("SET LOCAL hnsw.iterative_scan = off")
            plan = await connection.execute("EXPLAIN " + sql, params)
            assert hnsw_index_name(index.id) in str(await plan.fetchall())
            without = await (await connection.execute(sql, params)).fetchall()

        hits = await store.search(query, 10, access=hr_only)
    finally:
        await pool.close()

    assert len(without) < 10
    assert len(hits) == 10
    assert {hit.document.title for hit in hits} == {"salaries"}
