"""The chunk store on Postgres and pgvector.

The index's id and dimension are written into the SQL as literals (D19): psycopg prepares a
statement after five runs, and a generic plan for `index_id = $1` cannot use the partial HNSW
index, so a parameter there would quietly turn the vector search into a sequential scan.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from groundwork.db.pool import Pool
from groundwork.domain.models import (
    UNRESTRICTED,
    Access,
    Chunk,
    Document,
    DocumentChanges,
    DocumentMeta,
    SearchFilters,
    chunk_header,
    make_chunk_id,
    parse_chunk_id,
)
from groundwork.retrieval import indexes
from groundwork.retrieval.indexes import IndexInfo, vector_cast
from groundwork.retrieval.store import DenseHit, DuplicateDocumentError, normalized_groups

_DOCUMENT_COLUMNS = (
    "d.id, d.title, d.collection_id, d.doc_type, d.doc_date, d.filename, d.metadata, "
    "d.access_groups, d.owner_id, d.created_at, d.storage_path"
)
# Rows start with the document's columns; what a query adds follows.
_D = 11
_CHUNK_COLUMNS = "c.ordinal, c.text, c.section_path, c.page, c.char_start, c.char_end"


def _document(row: Sequence[Any]) -> Document:
    (
        doc_id,
        title,
        collection,
        doc_type,
        doc_date,
        filename,
        metadata,
        groups,
        owner,
        created,
        storage_path,
    ) = row
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
        storage_path=storage_path,
    )


def _passage(row: Sequence[Any]) -> tuple[Chunk, Document]:
    """A row of _DOCUMENT_COLUMNS followed by _CHUNK_COLUMNS."""
    document = _document(row[:_D])
    ordinal, text, section_path, page, char_start, char_end = row[_D : _D + 6]
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


def _filter_sql(
    filters: SearchFilters | None, access: Access = UNRESTRICTED
) -> tuple[str, list[Any]]:
    """Extra `AND` clauses on `chunks c`, with their parameters. The groups are checked on the
    copy in `chunks`, so the ACL runs in the vector query itself."""
    clauses: list[str] = []
    params: list[Any] = []
    if access.groups is not None:
        clauses.append("c.access_groups && %s::text[]")
        params.append(sorted(access.groups))
    if filters is None:
        return "".join(f" AND {clause}" for clause in clauses), params
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
            cursor = await connection.execute(
                f"SELECT d.id FROM documents d WHERE d.content_hash = %s AND d.collection_id = %s "
                f"AND d.access_groups = %s AND d.owner_id IS NOT DISTINCT FROM %s "
                f"AND EXISTS (SELECT 1 FROM chunks c WHERE c.index_id = {self._id} "
                f"AND c.document_id = d.id)",
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
    ) -> Document:
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk needs one embedding")
        async with self._pool.connection() as connection, connection.transaction():
            document = await self._upsert_document(connection, meta, content_hash)
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
        return document

    async def _upsert_document(
        self, connection: AsyncConnection[TupleRow], meta: DocumentMeta, content_hash: str
    ) -> Document:
        await connection.execute(
            "INSERT INTO collections (id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (meta.collection_id, meta.collection_id),
        )
        # The no-op update makes RETURNING give the existing row, whose tags are kept.
        cursor = await connection.execute(
            "INSERT INTO documents AS d (content_hash, collection_id, title, filename, doc_type, "
            "doc_date, metadata, access_groups, owner_id, storage_path) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (content_hash, collection_id, access_groups, owner_id) "
            "DO UPDATE SET content_hash = EXCLUDED.content_hash "
            f"RETURNING {_DOCUMENT_COLUMNS}",
            (
                content_hash,
                meta.collection_id,
                meta.title,
                meta.filename,
                meta.doc_type,
                meta.doc_date,
                Jsonb(dict(meta.metadata)),
                list(normalized_groups(meta.access_groups)),
                meta.owner_id,
                meta.storage_path,
            ),
        )
        row = await cursor.fetchone()
        assert row is not None
        return _document(row)

    def search_sql(
        self,
        query_embedding: Sequence[float],
        n: int,
        filters: SearchFilters | None = None,
        access: Access = UNRESTRICTED,
    ) -> tuple[str, list[Any]]:
        """The vector query and its parameters. Public so tests and docs/RETRIEVAL.md can EXPLAIN
        exactly what `search` runs.

        The index id and the limit are written into the SQL, not passed as parameters. psycopg
        prepares a statement on its fifth run, and Postgres may then plan it generically: with
        the index id unknown, the partial HNSW index cannot be used (D19). With the limit
        unknown, Postgres keeps re-planning every search rather than use a generic plan; with it
        written in, the generic plan uses the index and planning is skipped (docs/RETRIEVAL.md).
        Both are ints from this code, never user input."""
        where, params = _filter_sql(filters, access)
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
        self,
        query_embedding: Sequence[float],
        n: int,
        filters: SearchFilters | None = None,
        *,
        access: Access,
    ) -> list[DenseHit]:
        sql, params = self.search_sql(query_embedding, n, filters, access)
        async with self._pool.connection() as connection, connection.transaction():
            if self._exact:
                await connection.execute("SET LOCAL enable_indexscan = off")
            elif filters is not None or access.groups is not None:
                # pgvector 0.8: keep scanning the graph until n rows pass the filter, rather than
                # filtering the first ef_search rows and returning fewer than n
                # (test_a_restrictive_filter_still_returns_n_rows).
                await connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            cursor = await connection.execute(sql, params)
            rows = await cursor.fetchall()
        hits = []
        for row in rows:
            chunk, document = _passage(row)
            hits.append(DenseHit(chunk, document, similarity=1.0 - float(row[_D + 6])))
        return hits

    async def get(
        self, chunk_ids: Sequence[str], *, access: Access
    ) -> dict[str, tuple[Chunk, Document]]:
        if not chunk_ids:
            return {}
        keys = [parse_chunk_id(chunk_id) for chunk_id in chunk_ids]
        where, params = _filter_sql(None, access)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS}, {_CHUNK_COLUMNS} FROM chunks c "
                f"JOIN documents d ON d.id = c.document_id "
                f"JOIN unnest(%s::bigint[], %s::int[]) AS k(document_id, ordinal) "
                f"ON k.document_id = c.document_id AND k.ordinal = c.ordinal "
                f"WHERE c.index_id = {self._id}{where}",
                [[int(doc_id) for doc_id, _ in keys], [ordinal for _, ordinal in keys], *params],
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

    async def documents(self, *, access: Access) -> list[tuple[Document, int]]:
        where, params = _filter_sql(None, access)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS}, count(*) FROM documents d "
                f"JOIN chunks c ON c.document_id = d.id AND c.index_id = {self._id}{where} "
                f"GROUP BY d.id ORDER BY d.title, d.id",
                params,
            )
            rows = await cursor.fetchall()
        return [(_document(row[:_D]), int(row[_D])) for row in rows]

    async def document(self, doc_id: str, *, access: Access) -> tuple[Document, int] | None:
        if not doc_id.isdigit():
            return None
        where, params = _filter_sql(None, access)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_DOCUMENT_COLUMNS}, count(*) FROM documents d "
                f"JOIN chunks c ON c.document_id = d.id AND c.index_id = {self._id}{where} "
                f"WHERE d.id = %s GROUP BY d.id",
                [*params, int(doc_id)],
            )
            row = await cursor.fetchone()
        return (_document(row[:_D]), int(row[_D])) if row else None

    async def document_indexes(self, doc_id: str) -> list[str]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT i.name FROM indexes i WHERE EXISTS (SELECT 1 FROM chunks c "
                "WHERE c.index_id = i.id AND c.document_id = %s) ORDER BY i.name",
                (int(doc_id),),
            )
            return [name for (name,) in await cursor.fetchall()]

    async def update_document(self, doc_id: str, changes: DocumentChanges) -> Document:
        groups = (
            None
            if changes.access_groups is None
            else list(normalized_groups(changes.access_groups))
        )
        try:
            async with self._pool.connection() as connection, connection.transaction():
                cursor = await connection.execute(
                    "UPDATE documents d SET title = coalesce(%s, title), "
                    "doc_type = coalesce(%s, doc_type), doc_date = coalesce(%s, doc_date), "
                    "access_groups = coalesce(%s, access_groups) "
                    f"WHERE d.id = %s RETURNING {_DOCUMENT_COLUMNS}",
                    (changes.title, changes.doc_type, changes.doc_date, groups, int(doc_id)),
                )
                row = await cursor.fetchone()
                assert row is not None, doc_id
                document = _document(row)
                # Chunks carry copies of the fields searches filter on, in every index.
                await connection.execute(
                    "UPDATE chunks SET access_groups = %s, doc_date = %s WHERE document_id = %s",
                    (list(document.access_groups), document.doc_date, int(doc_id)),
                )
        except UniqueViolation as error:
            raise DuplicateDocumentError(
                "The same file is already stored for these groups."
            ) from error
        return document

    async def delete_document(self, doc_id: str) -> None:
        async with self._pool.connection() as connection:
            # Chunks go with it (ON DELETE CASCADE); its jobs keep their history.
            await connection.execute("DELETE FROM documents WHERE id = %s", (int(doc_id),))

    async def storage_path_in_use(self, storage_path: str) -> bool:
        return bool(
            await self._count(
                "SELECT count(*) FROM documents WHERE storage_path = %s", (storage_path,)
            )
        )

    async def protected_collections(self) -> frozenset[str]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute("SELECT id FROM collections WHERE protected")
            return frozenset(collection for (collection,) in await cursor.fetchall())

    async def protect_collection(self, collection_id: str) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                "INSERT INTO collections (id, name, protected) VALUES (%s, %s, true) "
                "ON CONFLICT (id) DO UPDATE SET protected = true",
                (collection_id, collection_id),
            )

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

    async def generation(self) -> int:
        return await indexes.generation(self._pool, self._id)

    async def bump_generation(self) -> int:
        return await indexes.bump_generation(self._pool, self._id)

    async def _count(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(sql, params)
            row = await cursor.fetchone()
        assert row is not None
        return int(row[0])
