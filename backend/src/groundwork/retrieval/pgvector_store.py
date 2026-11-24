"""The chunk store on Postgres and pgvector.

The index's id and dimension are written into the SQL as literals (D19): psycopg prepares a
statement after five runs, and a generic plan for `index_id = $1` cannot use the partial HNSW
index, so a parameter there would quietly turn the vector search into a sequential scan.
"""

from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from groundwork.db.pool import Pool
from groundwork.domain.models import (
    Chunk,
    Document,
    DocumentMeta,
    SearchFilters,
    chunk_header,
    make_chunk_id,
    parse_chunk_id,
)
from groundwork.retrieval import indexes
from groundwork.retrieval.indexes import IndexInfo, Publication, vector_cast
from groundwork.retrieval.store import CorpusDocument, DenseHit, normalized_groups

# pgvector's default and limit for hnsw.ef_search, the most chunks one pass over the graph finds.
_DEFAULT_EF_SEARCH = 40
_MAX_EF_SEARCH = 1000

# The advisory lock a corpus sync holds. Any number that nothing else in this database locks.
_SYNC_LOCK = 139_000_001

_DOCUMENT_COLUMNS = (
    "d.id, d.title, d.collection_id, d.doc_type, d.doc_date, d.filename, d.metadata, "
    "d.access_groups, d.owner_id, d.created_at"
)
_CHUNK_COLUMNS = "c.ordinal, c.text, c.section_path, c.page, c.char_start, c.char_end"


def _document(row: Sequence[Any]) -> Document:
    doc_id, title, collection, doc_type, doc_date, filename, metadata, groups, owner, created = row
    return Document(
        doc_id=str(doc_id),
        title=title,
        collection_id=collection,
        doc_type=doc_type,
        doc_date=doc_date,
        filename=filename,
        metadata=metadata,
        access_groups=tuple(groups),
        owner_id=owner,
        ingested_at=created,
    )


def _passage(row: Sequence[Any]) -> tuple[Chunk, Document]:
    """A row of _DOCUMENT_COLUMNS followed by _CHUNK_COLUMNS."""
    document = _document(row[:10])
    ordinal, text, section_path, page, char_start, char_end = row[10:16]
    chunk = Chunk(
        chunk_id=make_chunk_id(document.doc_id, ordinal),
        doc_id=document.doc_id,
        text=text,
        section_path=tuple(section_path),
        page=page,
        char_start=char_start,
        char_end=char_end,
    )
    return chunk, document


def _filter_sql(filters: SearchFilters | None) -> tuple[str, list[Any]]:
    """Extra `AND` clauses on `chunks c`, with their parameters."""
    if filters is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if filters.collections:
        clauses.append("c.collection_id = ANY(%s)")
        params.append(list(filters.collections))
    if filters.date_from:
        clauses.append("c.doc_date >= %s")
        params.append(filters.date_from)
    if filters.date_to:
        clauses.append("c.doc_date <= %s")
        params.append(filters.date_to)
    if filters.doc_types:
        clauses.append("c.document_id IN (SELECT id FROM documents WHERE doc_type = ANY(%s))")
        params.append(list(filters.doc_types))
    if filters.metadata:
        clauses.append("c.document_id IN (SELECT id FROM documents WHERE metadata @> %s)")
        params.append(Jsonb(dict(filters.metadata)))
    return "".join(f" AND {clause}" for clause in clauses), params


class PgVectorStore:
    def __init__(self, pool: Pool, index: IndexInfo, *, exact: bool = False) -> None:
        """`exact` turns off index scans for the vector query, so results do not depend on the
        HNSW graph: the CI eval uses it to be deterministic."""
        self._pool = pool
        self._index = index
        self._exact = exact
        self._id = int(index.id)

    @property
    def index(self) -> IndexInfo:
        return self._index

    @property
    def pool(self) -> Pool:
        return self._pool

    async def find_document(self, content_hash: str, meta: DocumentMeta) -> str | None:
        async with self._pool.connection() as connection:
            return await self._find(connection, content_hash, meta)

    async def find_upload(self, content_hash: str, meta: DocumentMeta) -> str | None:
        async with self._pool.connection() as connection:
            return await self._find_upload(connection, content_hash, meta)

    async def _find_upload(
        self, connection: AsyncConnection[TupleRow], content_hash: str, meta: DocumentMeta
    ) -> str | None:
        return await self._find(connection, content_hash, meta, "AND d.manifest_key IS NULL ")

    async def _find(
        self,
        connection: AsyncConnection[TupleRow],
        content_hash: str,
        meta: DocumentMeta,
        only: str = "",
    ) -> str | None:
        cursor = await connection.execute(
            f"SELECT d.id FROM documents d WHERE d.content_hash = %s AND d.collection_id = %s "
            f"AND d.access_groups = %s AND d.owner_id IS NOT DISTINCT FROM %s {only}"
            f"AND EXISTS (SELECT 1 FROM chunks c WHERE c.index_id = {self._id} "
            f"AND c.document_id = d.id) ORDER BY d.manifest_key IS NOT NULL, d.id LIMIT 1",
            (
                content_hash,
                meta.collection_id,
                list(normalized_groups(meta.access_groups)),
                meta.owner_id,
            ),
        )
        row = await cursor.fetchone()
        return str(row[0]) if row else None

    async def add(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        publish: bool = False,
    ) -> Document:
        async with self._pool.connection() as connection, connection.transaction():
            # Every writer locks in the same order, so two can't deadlock: the collection, the
            # document, its chunks, and last the rows of the indexes, by id.
            await self._ensure_collection(connection, meta)
            document = await self._upsert_upload(connection, meta, content_hash)
            await self._store_chunks(connection, document, chunks, embeddings)
            await self._mark_unpublished(connection, {self._id})
            if publish:
                await indexes.publish(connection, self._id)
        return document

    async def add_corpus_document(
        self,
        meta: DocumentMeta,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        *,
        manifest_key: str,
    ) -> Document | None:
        # Locks in the same order as `add`.
        async with self._pool.connection() as connection, connection.transaction():
            await self._ensure_collection(connection, meta)
            stored = await self._corpus_documents(
                connection,
                "WHERE d.collection_id = %s AND d.manifest_key = %s FOR UPDATE OF d",
                (meta.collection_id, manifest_key),
            )
            uploaded = await self._find_upload(connection, content_hash, meta) is not None
            outdated = [
                item
                for item in stored
                if not item.is_current(content_hash, meta) or (uploaded and item.in_index)
            ]
            changed = await self._remove(connection, [item.document.doc_id for item in outdated])
            document = None
            if not uploaded:
                if stored and not outdated:
                    document = stored[0].document
                else:
                    document = await self._insert_document(
                        connection, meta, content_hash, manifest_key
                    )
                await self._store_chunks(connection, document, chunks, embeddings)
                changed.add(self._id)
            await self._mark_unpublished(connection, changed)
        return document

    @staticmethod
    async def _ensure_collection(connection: AsyncConnection[TupleRow], meta: DocumentMeta) -> None:
        await connection.execute(
            "INSERT INTO collections (id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (meta.collection_id, meta.collection_id),
        )

    async def _store_chunks(
        self,
        connection: AsyncConnection[TupleRow],
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Make this index hold exactly these chunks of the document."""
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk needs one embedding")
        rows = [
            (
                self._id,
                int(document.doc_id),
                ordinal,
                chunk.text,
                chunk_header(chunk, document),
                list(chunk.section_path),
                chunk.page,
                chunk.char_start,
                chunk.char_end,
                list(document.access_groups),
                document.collection_id,
                document.doc_date,
                np.asarray(embedding, dtype=np.float32),
            )
            for ordinal, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        async with connection.cursor() as cursor:
            await cursor.executemany(
                "INSERT INTO chunks (index_id, document_id, ordinal, text, context, "
                "section_path, page, char_start, char_end, access_groups, collection_id, "
                "doc_date, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (index_id, document_id, ordinal) DO UPDATE SET "
                "text = EXCLUDED.text, context = EXCLUDED.context, "
                "section_path = EXCLUDED.section_path, page = EXCLUDED.page, "
                "char_start = EXCLUDED.char_start, char_end = EXCLUDED.char_end, "
                "access_groups = EXCLUDED.access_groups, "
                "collection_id = EXCLUDED.collection_id, doc_date = EXCLUDED.doc_date, "
                "embedding = EXCLUDED.embedding",
                rows,
            )
        # A document chunked again into fewer chunks leaves none of the old ones behind.
        await connection.execute(
            f"DELETE FROM chunks WHERE index_id = {self._id} AND document_id = %s "
            f"AND ordinal >= %s",
            (int(document.doc_id), len(chunks)),
        )

    async def _upsert_upload(
        self, connection: AsyncConnection[TupleRow], meta: DocumentMeta, content_hash: str
    ) -> Document:
        # The no-op update makes RETURNING give the existing row, whose tags are kept.
        return await self._insert_document(
            connection,
            meta,
            content_hash,
            None,
            "ON CONFLICT (content_hash, collection_id, access_groups, owner_id) "
            "WHERE manifest_key IS NULL DO UPDATE SET content_hash = EXCLUDED.content_hash ",
        )

    async def _insert_document(
        self,
        connection: AsyncConnection[TupleRow],
        meta: DocumentMeta,
        content_hash: str,
        manifest_key: str | None,
        on_conflict: str = "",
    ) -> Document:
        cursor = await connection.execute(
            "INSERT INTO documents AS d (manifest_key, content_hash, collection_id, title, "
            "filename, doc_type, doc_date, metadata, access_groups, owner_id) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) {on_conflict}"
            f"RETURNING {_DOCUMENT_COLUMNS}",
            (
                manifest_key,
                content_hash,
                meta.collection_id,
                meta.title,
                meta.filename,
                meta.doc_type,
                meta.doc_date,
                Jsonb(dict(meta.metadata)),
                list(normalized_groups(meta.access_groups)),
                meta.owner_id,
            ),
        )
        row = await cursor.fetchone()
        assert row is not None
        return _document(row)

    async def corpus_documents(self) -> list[CorpusDocument]:
        async with self._pool.connection() as connection:
            return await self._corpus_documents(
                connection, "WHERE d.manifest_key IS NOT NULL ORDER BY d.id"
            )

    async def _corpus_documents(
        self, connection: AsyncConnection[TupleRow], where: str, params: Sequence[Any] = ()
    ) -> list[CorpusDocument]:
        cursor = await connection.execute(
            f"SELECT {_DOCUMENT_COLUMNS}, d.manifest_key, d.content_hash, "
            f"EXISTS (SELECT 1 FROM chunks c WHERE c.index_id = {self._id} "
            f"AND c.document_id = d.id) FROM documents d {where}",
            params,
        )
        return [
            CorpusDocument(_document(document), key, content_hash, in_index)
            for *document, key, content_hash, in_index in await cursor.fetchall()
        ]

    async def remove_corpus_documents(self, doc_ids: Sequence[str]) -> None:
        async with self._pool.connection() as connection, connection.transaction():
            await self._mark_unpublished(connection, await self._remove(connection, doc_ids))

    @staticmethod
    async def _remove(connection: AsyncConnection[TupleRow], doc_ids: Sequence[str]) -> set[int]:
        """Delete corpus documents and their chunks. The ids of the indexes that held any."""
        # Locked first: nothing can add chunks of these documents between here and the commit.
        cursor = await connection.execute(
            "SELECT id FROM documents WHERE id = ANY(%s) AND manifest_key IS NOT NULL "
            "ORDER BY id FOR UPDATE",
            ([int(doc_id) for doc_id in doc_ids],),
        )
        ids = [doc_id for (doc_id,) in await cursor.fetchall()]
        cursor = await connection.execute(
            "WITH gone AS (DELETE FROM chunks WHERE document_id = ANY(%s) RETURNING index_id) "
            "SELECT DISTINCT index_id FROM gone",
            (ids,),
        )
        held = {int(index_id) for (index_id,) in await cursor.fetchall()}
        await connection.execute("DELETE FROM documents WHERE id = ANY(%s)", (ids,))
        return held

    @staticmethod
    async def _mark_unpublished(
        connection: AsyncConnection[TupleRow], index_ids: Iterable[int]
    ) -> None:
        for index_id in sorted(index_ids):
            await indexes.mark_unpublished(connection, index_id)

    @asynccontextmanager
    async def syncing(self) -> AsyncIterator[None]:
        # Keeps one connection for the whole sync. The lock goes with its transaction, so also
        # when the process dies.
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (_SYNC_LOCK,))
            yield

    def search_sql(
        self, query_embedding: Sequence[float], n: int, filters: SearchFilters | None = None
    ) -> tuple[str, list[Any]]:
        """The vector query and its parameters. Public so tests and docs/RETRIEVAL.md can EXPLAIN
        exactly what `search` runs.

        The index id and the limit are written into the SQL, not passed as parameters. psycopg
        prepares a statement on its fifth run, and Postgres may then plan it generically: with
        the index id unknown, the partial HNSW index cannot be used (D19). With the limit
        unknown, Postgres keeps re-planning every search rather than use a generic plan; with it
        written in, the generic plan uses the index and planning is skipped (docs/RETRIEVAL.md).
        Both are ints from this code, never user input."""
        where, params = _filter_sql(filters)
        cast = vector_cast(self._index, "c")
        query = np.asarray(query_embedding, dtype=np.float32)
        sql = (
            f"SELECT {_DOCUMENT_COLUMNS}, {_CHUNK_COLUMNS}, c.distance FROM ("
            f"SELECT c.*, {cast} <=> %s::vector({self._index.dimension}) AS distance "
            f"FROM chunks c WHERE c.index_id = {self._id}{where} "
            f"ORDER BY distance LIMIT {int(n)}"
            f") c JOIN documents d ON d.id = c.document_id ORDER BY c.distance"
        )
        return sql, [query, *params]

    async def search(
        self, query_embedding: Sequence[float], n: int, filters: SearchFilters | None = None
    ) -> list[DenseHit]:
        sql, params = self.search_sql(query_embedding, n, filters)
        async with self._pool.connection() as connection, connection.transaction():
            if self._exact:
                await connection.execute("SET LOCAL enable_indexscan = off")
            else:
                if n > _DEFAULT_EF_SEARCH:
                    ef_search = min(int(n), _MAX_EF_SEARCH)
                    await connection.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                if filters is not None:
                    # pgvector 0.8: keep scanning the graph until n rows pass the filter, rather
                    # than filtering the first ef_search rows and returning fewer than n.
                    await connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            cursor = await connection.execute(sql, params)
            rows = await cursor.fetchall()
        hits = []
        for row in rows:
            chunk, document = _passage(row)
            hits.append(DenseHit(chunk, document, similarity=1.0 - float(row[16])))
        return hits

    async def get(self, chunk_ids: Sequence[str]) -> dict[str, tuple[Chunk, Document]]:
        if not chunk_ids:
            return {}
        keys = [parse_chunk_id(chunk_id) for chunk_id in chunk_ids]
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS}, {_CHUNK_COLUMNS} FROM chunks c "
                f"JOIN documents d ON d.id = c.document_id "
                f"JOIN unnest(%s::bigint[], %s::int[]) AS k(document_id, ordinal) "
                f"ON k.document_id = c.document_id AND k.ordinal = c.ordinal "
                f"WHERE c.index_id = {self._id}",
                ([int(doc_id) for doc_id, _ in keys], [ordinal for _, ordinal in keys]),
            )
            rows = await cursor.fetchall()
        passages = [_passage(row) for row in rows]
        return {chunk.chunk_id: (chunk, document) for chunk, document in passages}

    async def all_passages(self) -> list[tuple[Chunk, Document]]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS}, {_CHUNK_COLUMNS} FROM chunks c "
                f"JOIN documents d ON d.id = c.document_id WHERE c.index_id = {self._id} "
                f"ORDER BY c.document_id, c.ordinal"
            )
            rows = await cursor.fetchall()
        return [_passage(row) for row in rows]

    async def matching_ids(self, filters: SearchFilters) -> set[str]:
        where, params = _filter_sql(filters)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT c.document_id, c.ordinal FROM chunks c "
                f"WHERE c.index_id = {self._id}{where}",
                params,
            )
            rows = await cursor.fetchall()
        return {make_chunk_id(str(doc_id), ordinal) for doc_id, ordinal in rows}

    async def documents(self) -> list[tuple[Document, int]]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS}, count(*) FROM documents d "
                f"JOIN chunks c ON c.document_id = d.id AND c.index_id = {self._id} "
                f"GROUP BY d.id ORDER BY d.title, d.id"
            )
            rows = await cursor.fetchall()
        return [(_document(row[:10]), int(row[10])) for row in rows]

    async def count_chunks(self) -> int:
        return await self._count(f"SELECT count(*) FROM chunks WHERE index_id = {self._id}")

    async def count_documents(self) -> int:
        return await self._count(
            f"SELECT count(DISTINCT document_id) FROM chunks WHERE index_id = {self._id}"
        )

    async def count_document_chunks(self, doc_id: str) -> int:
        return await self._count(
            f"SELECT count(*) FROM chunks WHERE index_id = {self._id} AND document_id = %s",
            (int(doc_id),),
        )

    async def publication(self) -> Publication:
        return await indexes.publication(self._pool, self._id)

    async def publish(self) -> int | None:
        async with self._pool.connection() as connection:
            return await indexes.publish(connection, self._id)

    async def publish_all(self) -> dict[str, int]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT id, name FROM indexes WHERE unpublished ORDER BY id"
            )
            rows = await cursor.fetchall()
        published = {}
        for index_id, name in rows:
            # One transaction each: holding several rows could deadlock with another writer.
            async with self._pool.connection() as connection:
                generation = await indexes.publish(connection, index_id)
            if generation is not None:
                published[name] = generation
        return published

    async def vacuum(self) -> None:
        async with self._pool.connection() as connection:
            # VACUUM can't run in a transaction.
            await connection.set_autocommit(True)
            try:
                # Left to itself, VACUUM skips the indexes when few of the table's pages hold
                # removed chunks, and the HNSW graphs would go on leading to them.
                await connection.execute("VACUUM (INDEX_CLEANUP ON) chunks")
            finally:
                await connection.set_autocommit(False)

    async def _count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(sql, params)
            row = await cursor.fetchone()
        assert row is not None
        return int(row[0])
