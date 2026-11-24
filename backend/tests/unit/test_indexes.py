"""The index registry, and the query plans D19 depends on."""

import asyncio
import json
from dataclasses import replace

import pytest
from psycopg.errors import NumericValueOutOfRange

from groundwork.db.pool import Pool
from groundwork.domain.models import Chunk, DocumentMeta, SearchFilters
from groundwork.retrieval.indexes import (
    IndexMismatchError,
    IndexSpec,
    activate,
    active_index,
    drop_index,
    ensure_index,
    get_index,
    hnsw_index_name,
    mark_unpublished,
    model_slug,
    window_spec,
)
from groundwork.retrieval.pgvector_store import PgVectorStore

pytestmark = pytest.mark.db

SPEC = IndexSpec(name="test", chunker="window", embedder_model="fake", dimension=3)


def test_window_spec_names_the_index_the_app_has_always_built() -> None:
    spec = window_spec(400, 60, "BAAI/bge-small-en-v1.5", 384)

    assert spec.name == "window-400-header-bge-small"
    assert spec.chunker_params == {"chunk_tokens": 400, "chunk_overlap": 60}
    assert model_slug("sentence-transformers/all-MiniLM-L6-v2") == "all-minilm-l6-v2"


async def test_ensure_index_creates_once(pool: Pool) -> None:
    first = await ensure_index(pool, SPEC)
    again = await ensure_index(pool, SPEC)

    assert again == first
    assert first.status == "building"
    assert first.generation == 0
    assert await get_index(pool, "test") == first


async def test_ensure_index_refuses_a_different_build_under_the_same_name(pool: Pool) -> None:
    await ensure_index(pool, SPEC)

    with pytest.raises(IndexMismatchError, match="dimension"):
        await ensure_index(pool, replace(SPEC, dimension=4))


async def test_one_index_is_active(pool: Pool) -> None:
    a = await ensure_index(pool, SPEC)
    b = await ensure_index(pool, replace(SPEC, name="other"))

    await activate(pool, a.id)
    await activate(pool, b.id)

    active = await active_index(pool)
    assert active is not None
    assert (active.id, active.status) == (b.id, "ready")


def document(title: str) -> DocumentMeta:
    return DocumentMeta(title=title, collection_id="c", doc_type="x", doc_date=None, filename=title)


CHUNK = Chunk("x", "x", "a", (), None, 0, 1)
VECTOR = [1.0, 0.0, 0.0]


async def test_publishing_one_index_leaves_the_others(pool: Pool) -> None:
    a = PgVectorStore(pool, await ensure_index(pool, SPEC))
    b = PgVectorStore(pool, await ensure_index(pool, replace(SPEC, name="other")))
    await a.add(document("a"), "h", [CHUNK], [VECTOR])
    await b.add(document("a"), "h", [CHUNK], [VECTOR])

    assert await a.publish() == 1

    assert (await b.publication()).generation == 0
    assert await b.publish() == 1


async def test_a_write_under_way_is_not_lost_by_a_publish(pool: Pool) -> None:
    """The index is marked already, so nothing but the write's own mark can hold the publish back
    until the write can be seen."""
    index = await ensure_index(pool, SPEC)
    await PgVectorStore(pool, index).add(document("a"), "h", [CHUNK], [VECTOR])
    store = PgVectorStore(pool, index)

    async with pool.connection() as connection, connection.transaction():
        await mark_unpublished(connection, index.id)
        publishing = asyncio.create_task(store.publish())
        done, _ = await asyncio.wait([publishing], timeout=0.2)
        assert not done

    assert await publishing == 1
    assert await store.publish() is None


async def test_a_write_that_cannot_be_published_is_not_stored(pool: Pool) -> None:
    index = await ensure_index(pool, SPEC)
    store = PgVectorStore(pool, index)
    async with pool.connection() as connection:
        # The largest bigint, so the generation cannot move.
        await connection.execute("UPDATE indexes SET generation = 9223372036854775807")

    with pytest.raises(NumericValueOutOfRange):
        await store.add(document("a"), "h", [CHUNK], [VECTOR], publish=True)

    assert await store.find_document("h", document("a")) is None
    assert await store.publish() is None


async def test_publishing_a_dropped_index_does_nothing(pool: Pool) -> None:
    index = await ensure_index(pool, SPEC)
    store = PgVectorStore(pool, index)
    await store.add(document("a"), "h", [CHUNK], [VECTOR])

    await drop_index(pool, index.id)

    assert await store.publish() is None


async def test_drop_index_removes_its_chunks_and_hnsw_index(pool: Pool) -> None:
    index = await ensure_index(pool, SPEC)
    store = PgVectorStore(pool, index)
    await store.add(document("t"), "h", [CHUNK], [VECTOR])

    await drop_index(pool, index.id)

    async with pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT count(*) FROM pg_indexes WHERE indexname = %s", (hnsw_index_name(index.id),)
        )
        assert await cursor.fetchone() == (0,)
        cursor = await connection.execute("SELECT count(*) FROM chunks")
        assert await cursor.fetchone() == (0,)
        cursor = await connection.execute("SELECT count(*) FROM documents")
        assert await cursor.fetchone() == (1,)


@pytest.mark.parametrize("filters", [None, SearchFilters(collections=("c",))])
async def test_a_generic_plan_still_uses_the_partial_hnsw_index(
    pool: Pool, filters: SearchFilters | None
) -> None:
    """psycopg prepares a statement after five runs, and Postgres may then plan it generically.
    With the index id written into the SQL, even a forced generic plan can use the index."""
    index = await ensure_index(pool, SPEC)
    store = PgVectorStore(pool, index)
    sql, params = store.search_sql([1.0, 0.0, 0.0], 5, filters)

    # Written in, so a generic plan knows them (docs/RETRIEVAL.md).
    assert f"index_id = {index.id}" in sql and "LIMIT 5" in sql

    async with pool.connection() as connection, connection.transaction():
        # The table is tiny, so the planner would rightly prefer a sequential scan, or the btree
        # on index_id and a sort by distance. The question here is whether a generic plan can
        # use the HNSW index at all, so those are taken away.
        await connection.execute("SET LOCAL enable_seqscan = off")
        await connection.execute("SET LOCAL enable_sort = off")
        await connection.execute("SET LOCAL plan_cache_mode = force_generic_plan")
        for _ in range(6):
            cursor = await connection.execute("EXPLAIN (FORMAT JSON) " + sql, params, prepare=True)
            row = await cursor.fetchone()
    assert row is not None
    plan = json.dumps(row[0])
    assert hnsw_index_name(index.id) in plan
