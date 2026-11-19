"""EXPLAIN ANALYZE the dense query the store runs, for docs/RETRIEVAL.md.

Run from backend/: uv run python scripts/explain_search.py [--index NAME] [--query TEXT]

Prints, for one index (the active one unless --index names another):
1. the plain top-20 query,
2. the same with a selective collection filter and pgvector's iterative scan,
3. the plain query at three hnsw.ef_search values, with the overlap of their results,
4. with --questions FILE, HNSW's top 20 against exact search for every question in the file,
5. a search on a pooled connection after at least twelve others, as the API runs it: psycopg
   prepares a query on its fifth run, and after five custom plans Postgres may plan it
   generically.
"""

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg import AsyncClientCursor, AsyncConnection

from groundwork.config import Settings
from groundwork.container import build_embedder
from groundwork.db.pool import open_pool
from groundwork.domain.models import SearchFilters
from groundwork.retrieval.indexes import active_index, get_index
from groundwork.retrieval.pgvector_store import PgVectorStore

N = 20


def plan_text(rows: list[Any]) -> str:
    """The plan, with the query vector's 384 numbers shown as <query>."""
    return re.sub(r"'\[[-0-9.e, ]+\]'", "<query>", "\n".join(row[0] for row in rows))


async def explain(connection: AsyncConnection[Any], sql: str, params: list[Any]) -> str:
    cursor = await connection.execute(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {sql}", params)
    return plan_text(await cursor.fetchall())


async def ids(connection: AsyncConnection[Any], sql: str, params: list[Any]) -> list[str]:
    cursor = await connection.execute(sql, params)
    # Column 0 is the document id and the chunk ordinal follows the document columns.
    return [f"{row[0]}:{row[10]}" for row in await cursor.fetchall()]


async def exact_ids(connection: AsyncConnection[Any], sql: str, params: list[Any]) -> list[str]:
    async with connection.transaction():
        await connection.execute("SET LOCAL enable_indexscan = off")
        return await ids(connection, sql, params)


def cursor_row(row: tuple[Any, ...] | None) -> tuple[str, int, int]:
    assert row is not None, "psycopg did not prepare the search"
    return str(row[0]), int(row[1]), int(row[2])


SAMPLE_QUESTIONS = [
    "How many weeks of parental leave do employees get?",
    "What was Apple's revenue in fiscal 2026?",
    "How does a Kubernetes Service find its Pods?",
    "Who is Elizabeth Bennet's eldest sister?",
    "What is the expense limit for hotels?",
    "How many people did Microsoft employ?",
]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index")
    parser.add_argument("--query", default="How many weeks of parental leave do employees get?")
    parser.add_argument("--collection", default="northwind")
    parser.add_argument("--questions", type=Path, help="Compare HNSW with exact search on these.")
    args = parser.parse_args()

    settings = Settings()
    embedder = build_embedder(settings)
    vector = embedder.embed_query(args.query)
    pool = await open_pool(settings.database_url)
    try:
        index = await (get_index(pool, args.index) if args.index else active_index(pool))
        assert index is not None, "no such index"
        store = PgVectorStore(pool, index)
        print(f"Index {index.name} (id {index.id}), {await store.count_chunks()} chunks\n")
        plain_sql, plain = store.search_sql(vector, N)
        filtered_sql, filtered = store.search_sql(
            vector, N, SearchFilters(collections=(args.collection,))
        )
        # Sections 1 to 3 change planner settings, so they use their own connection and leave
        # the pool as the app finds it. Autocommit makes each `transaction()` a real one, so its
        # SET LOCAL ends with it, and nothing is prepared: a prepared statement keeps its plan
        # when settings change.
        async with await AsyncConnection.connect(
            settings.database_url, autocommit=True, prepare_threshold=None
        ) as connection:
            await register_vector_async(connection)
            await connection.execute(plain_sql, plain)  # warm the cache
            print("## 1. Top 20, no filter\n")
            print(await explain(connection, plain_sql, plain))

            print(f"\n## 2. Top 20 in {args.collection}, iterative scan\n")
            async with connection.transaction():
                await connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                print(await explain(connection, filtered_sql, filtered))
            async with connection.transaction():
                await connection.execute("SET LOCAL hnsw.iterative_scan = off")
                found = await ids(connection, filtered_sql, filtered)
                print(f"\nWithout the iterative scan: {len(found)} rows of {N}.")

            print("\n## 3. hnsw.ef_search\n")
            exact = await exact_ids(connection, plain_sql, plain)
            for ef in (10, 40, 100):
                async with connection.transaction():
                    await connection.execute(f"SET LOCAL hnsw.ef_search = {ef}")
                    plan = await explain(connection, plain_sql, plain)
                    found = await ids(connection, plain_sql, plain)
                time = re.search(r"Execution Time: .*", plan)
                overlap = len(set(found) & set(exact))
                print(f"ef_search {ef}: {overlap}/{N} of the exact top 20; {time and time[0]}")

        if args.questions:
            # Through the store, as the app searches: each search is its own transaction.
            exact_store = PgVectorStore(pool, index, exact=True)
            lines = args.questions.read_text().splitlines()
            questions = [json.loads(line)["question"] for line in lines if line.strip()]
            overlaps = []
            for question in questions:
                query = embedder.embed_query(question)
                approximate = {hit.chunk.chunk_id for hit in await store.search(query, N)}
                exact_hits = {hit.chunk.chunk_id for hit in await exact_store.search(query, N)}
                overlaps.append(len(approximate & exact_hits))
            differ = sum(overlap < N for overlap in overlaps)
            print(
                f"\n## 4. HNSW against exact search, {len(questions)} questions\n\n"
                f"Top 20 identical for {len(questions) - differ}; differs for {differ} "
                f"(fewest shared: {min(overlaps)}/{N}, mean {statistics.mean(overlaps):.2f})."
            )

        print("\n## 5. A search on a pooled connection after many others\n")
        async with pool.connection() as connection:
            # As the API runs it: psycopg prepares the statement on its fifth run, and Postgres
            # may switch to a generic plan from the sixth.
            # psycopg prepares on the fifth run; Postgres weighs a generic plan after five custom
            # plans of the prepared statement. Twelve runs get past both.
            for question in SAMPLE_QUESTIONS * 2:
                sql, params = store.search_sql(embedder.embed_query(question), N)
                await connection.execute(sql, params)
            cursor = await connection.execute(
                "SELECT name, generic_plans, custom_plans FROM pg_prepared_statements "
                "WHERE statement LIKE %s",
                (f"%index_id = {index.id}%",),
            )
            name, generic, custom = cursor_row(await cursor.fetchone())
            print(f"Prepared as {name}: {custom} custom plans, then {generic} generic.\n")
            # EXECUTE takes no bind parameters, so the values are written into the statement.
            literal = [str(v.tolist()) if isinstance(v, np.ndarray) else v for v in params]
            client = AsyncClientCursor(connection)
            execute = f"EXECUTE {name}(" + ", ".join(["%s"] * len(literal)) + ")"
            await client.execute(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {execute}", literal)
            print(plan_text(await client.fetchall()))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
