"""The index registry: one row in `indexes` per way of chunking and embedding the corpus.

Every index keeps its chunks in the shared `chunks` table and has its own partial HNSW index over
them (D19 in docs/HANDOFF.md): `(embedding::vector(<dimension>))` where `index_id = <id>`, with
both numbers written into the SQL. They come from this table, never from a user. One index is
active: the one the app searches.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from groundwork.db.pool import Pool

IndexStatus = Literal["building", "ready", "retired"]

# HNSW build parameters, written out rather than left to pgvector's defaults.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


class IndexMismatchError(ValueError):
    """An index of that name exists, built some other way."""


@dataclass(frozen=True)
class IndexSpec:
    """How an index is built. Two indexes with one name must have the same spec."""

    name: str
    chunker: str
    embedder_model: str
    dimension: int
    chunker_params: dict[str, Any] = field(default_factory=dict)
    contextualizer: str = "header"
    contextualizer_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexInfo:
    id: int
    spec: IndexSpec
    status: IndexStatus
    generation: int
    is_active: bool

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dimension(self) -> int:
        return self.spec.dimension


@dataclass(frozen=True)
class Publication:
    """The state of an index's chunks that BM25 is built from. The generation alone does not name
    it: a recreated database, or one restored from a backup, counts through the same numbers."""

    generation: int
    token: str
    # The index holds writes that are not in this publication.
    written_since: bool


def model_slug(model: str) -> str:
    """BAAI/bge-small-en-v1.5 -> bge-small"""
    name = model.rsplit("/", 1)[-1].lower()
    return re.sub(r"-en(-v[\d.]+)?$", "", name)


def window_spec(chunk_tokens: int, chunk_overlap: int, model: str, dimension: int) -> IndexSpec:
    """The window chunker with a title-and-heading header: the index the app has always built."""
    return IndexSpec(
        name=f"window-{chunk_tokens}-header-{model_slug(model)}",
        chunker="window",
        chunker_params={"chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap},
        contextualizer="header",
        embedder_model=model,
        dimension=dimension,
    )


def hnsw_index_name(index_id: int) -> str:
    return f"chunks_hnsw_{index_id}"


def _int(value: int) -> int:
    """Guards the integers that are written into SQL: they must be real ints."""
    if type(value) is not int:
        raise TypeError(f"expected an int, got {value!r}")
    return value


def vector_cast(index: IndexInfo, table: str | None = None) -> str:
    """`(embedding::vector(384))`: the expression the partial HNSW index is built over. Queries
    must repeat it exactly for the planner to use that index."""
    column = f"{table}.embedding" if table else "embedding"
    return f"({column}::vector({_int(index.dimension)}))"


_COLUMNS = (
    "id, name, chunker, chunker_params, contextualizer, contextualizer_params, embedder_model, "
    "dimension, status, generation, is_active"
)


def _from_row(row: TupleRow) -> IndexInfo:
    (
        index_id,
        name,
        chunker,
        chunker_params,
        contextualizer,
        contextualizer_params,
        model,
        dimension,
        status,
        generation,
        is_active,
    ) = row
    spec = IndexSpec(
        name=name,
        chunker=chunker,
        chunker_params=chunker_params,
        contextualizer=contextualizer,
        contextualizer_params=contextualizer_params,
        embedder_model=model,
        dimension=dimension,
    )
    return IndexInfo(index_id, spec, status, generation, is_active)


async def _create_hnsw(connection: AsyncConnection[TupleRow], index: IndexInfo) -> None:
    await connection.execute(
        f"CREATE INDEX IF NOT EXISTS {hnsw_index_name(_int(index.id))} ON chunks "
        f"USING hnsw ({vector_cast(index)} vector_cosine_ops) "
        f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}) "
        f"WHERE index_id = {_int(index.id)}"
    )


async def get_index(pool: Pool, name: str) -> IndexInfo | None:
    async with pool.connection() as connection:
        cursor = await connection.execute(
            f"SELECT {_COLUMNS} FROM indexes WHERE name = %s", (name,)
        )
        row = await cursor.fetchone()
    return _from_row(row) if row else None


async def ensure_index(pool: Pool, spec: IndexSpec) -> IndexInfo:
    """The index with this spec's name, created with its HNSW index if it is new. Raises
    IndexMismatchError if one of that name was built another way."""
    async with pool.connection() as connection, connection.transaction():
        await connection.execute(
            "INSERT INTO indexes (name, chunker, chunker_params, contextualizer, "
            "contextualizer_params, embedder_model, dimension) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
            (
                spec.name,
                spec.chunker,
                Jsonb(spec.chunker_params),
                spec.contextualizer,
                Jsonb(spec.contextualizer_params),
                spec.embedder_model,
                spec.dimension,
            ),
        )
        cursor = await connection.execute(
            f"SELECT {_COLUMNS} FROM indexes WHERE name = %s", (spec.name,)
        )
        row = await cursor.fetchone()
        assert row is not None
        index = _from_row(row)
        if index.spec != spec:
            raise IndexMismatchError(
                f"Index {spec.name} exists with {json.dumps(_describe(index.spec))}, "
                f"not {json.dumps(_describe(spec))}."
            )
        await _create_hnsw(connection, index)
    return index


def _describe(spec: IndexSpec) -> dict[str, Any]:
    return {
        "chunker": spec.chunker,
        "chunker_params": spec.chunker_params,
        "contextualizer": spec.contextualizer,
        "embedder_model": spec.embedder_model,
        "dimension": spec.dimension,
    }


async def activate(pool: Pool, index_id: int) -> None:
    """Make this the index the app searches, and mark it ready."""
    async with pool.connection() as connection, connection.transaction():
        await connection.execute("UPDATE indexes SET is_active = false WHERE is_active")
        await connection.execute(
            "UPDATE indexes SET is_active = true, status = 'ready' WHERE id = %s", (index_id,)
        )


async def active_index(pool: Pool) -> IndexInfo | None:
    async with pool.connection() as connection:
        cursor = await connection.execute(f"SELECT {_COLUMNS} FROM indexes WHERE is_active")
        row = await cursor.fetchone()
    return _from_row(row) if row else None


async def publication(pool: Pool, index_id: int) -> Publication:
    """The index's last publication."""
    async with pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT generation, generation_token, unpublished FROM indexes WHERE id = %s",
            (index_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        raise LookupError(f"No index {index_id}")
    generation, token, unpublished = row
    return Publication(generation, str(token), unpublished)


async def mark_unpublished(connection: AsyncConnection[TupleRow], index_id: int) -> None:
    """Record that this index has writes BM25 has not seen. Called in the transaction of any write
    that changes the index's chunks, as its last step: it locks the index's row until the commit.

    Unconditional on purpose, also when the index is marked already: skipping the update would
    skip the lock, and a publish between it and the commit would clear the mark of a write it
    could not see."""
    await connection.execute("UPDATE indexes SET unpublished = true WHERE id = %s", (index_id,))


async def publish(connection: AsyncConnection[TupleRow], index_id: int) -> int | None:
    """Move the index's generation and give it a new token, so each API worker rebuilds BM25, and
    clear its record of unpublished writes. The new generation, or None if there was nothing to
    publish.

    Takes a connection so that a write can be published in its own transaction."""
    cursor = await connection.execute(
        "UPDATE indexes SET generation = generation + 1, generation_token = gen_random_uuid(), "
        "unpublished = false "
        "WHERE id = %s AND unpublished RETURNING generation",
        (index_id,),
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else None


async def drop_index(pool: Pool, index_id: int) -> None:
    """Delete an index with its chunks and its HNSW index. Documents stay."""
    async with pool.connection() as connection, connection.transaction():
        await connection.execute(f"DROP INDEX IF EXISTS {hnsw_index_name(_int(index_id))}")
        await connection.execute("DELETE FROM indexes WHERE id = %s", (index_id,))
