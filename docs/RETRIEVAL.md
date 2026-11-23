# Retrieval on Postgres

How the dense query runs on pgvector, measured on the evaluation's 400-token index: 4,800 chunks
from 205 documents in four collections, `bge-small-en-v1.5` (384 dimensions), pgvector 0.8.7,
HNSW with `m = 16`, `ef_construction = 64`. Every number here comes from

```sh
uv run python eval/run_eval.py    # leaves its indexes in the eval database
uv run python scripts/explain_search.py --eval --index window-400-header-bge-small \
    --questions eval/questions.jsonl
```

run on a laptop with the whole table in shared buffers. Timings are single runs; the plans are
what matter.

## The query

One `chunks` table holds every index (D18). The vector column is untyped, so each index has its
own partial HNSW index over a cast to its dimension, and the query repeats that expression:

```sql
CREATE INDEX chunks_hnsw_4 ON chunks USING hnsw ((embedding::vector(384)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64) WHERE index_id = 4;

SELECT <document and chunk columns>, c.distance FROM (
    SELECT c.*, (c.embedding::vector(384)) <=> $1::vector(384) AS distance
    FROM chunks c WHERE c.index_id = 4 <AND filters>
    ORDER BY distance LIMIT 20
) c JOIN documents d ON d.id = c.document_id ORDER BY c.distance
```

The index id and the limit are written into the SQL; only the query vector and filter values
are parameters (see [Prepared statements](#prepared-statements)).

## 1. No filter: the HNSW index

```
Sort (actual time=0.404..0.406 rows=20 loops=1)
  ->  Hash Join (actual time=0.330..0.388 rows=20 loops=1)
        ->  Limit (actual time=0.244..0.298 rows=20 loops=1)
              ->  Index Scan using chunks_hnsw_4 on chunks c (actual time=0.244..0.296 rows=20 loops=1)
                    Order By: ((embedding)::vector(384) <=> <query>::vector(384))
                    Buffers: shared hit=604
        ->  Hash (actual time=0.084..0.084 rows=205 loops=1)
              ->  Seq Scan on documents d (actual time=0.003..0.028 rows=205 loops=1)
Execution Time: 0.420 ms
```

A sequential scan of the same chunks takes about 10 ms, and grows with the table.

## 2. A selective filter: exact search, chosen by the planner

Filtered to `northwind` (132 of the 4,800 chunks), the planner leaves the graph alone. It reads
those rows through the btree on `(index_id, collection_id, doc_date)` and sorts them by distance,
which is exact search and still under a millisecond:

```
->  Limit (actual time=0.587..0.589 rows=20 loops=1)
      ->  Sort (actual time=0.587..0.588 rows=20 loops=1)
            Sort Method: top-N heapsort  Memory: 42kB
            ->  Index Scan using chunks_filters on chunks c_1 (actual time=0.043..0.553 rows=132 loops=1)
                  Index Cond: ((index_id = 4) AND (collection_id = ANY ('{northwind}'::text[])))
Execution Time: 0.663 ms
```

When the planner does use the graph with a filter, HNSW finds its `ef_search` nearest neighbours
first and filters after, so a restrictive filter can return fewer than n rows. Filtered searches
set `hnsw.iterative_scan = strict_order` (pgvector 0.8), which keeps scanning until n rows pass.
M3's access groups make every user's search filtered; its tests show the failure without the
setting.

## 3. `hnsw.ef_search`

The share of exact search's top 20 that HNSW returns, for one question:

| `ef_search` | Shared with exact | Time |
|---|---|---|
| 10 | 10 / 20 | 0.33 ms |
| 40 (default) | 19 / 20 | 0.43 ms |
| 100 | 19 / 20 | 1.63 ms |

`ef_search` must be at least the limit (20) to fill it from the graph alone. The default is
kept.

## 4. HNSW against exact search, all 126 questions

Searched through the store, as the app does, with the default `ef_search`: **the top 20 is
identical for 123 questions**; the other 3 share 18 or 19 of 20. On this corpus HNSW changes
nothing measurable: the retrieval eval's Success@6 matches Chroma's in every row (see
`eval/decisions.md`), and the CI gate uses exact search anyway so the graph cannot make it flaky.

## Prepared statements

psycopg prepares a statement on its fifth run on a connection, and after five custom plans
Postgres may switch to one generic plan, planned without the parameter values:

- **With the index id as a parameter, the generic plan cannot use the partial index**: Postgres
  does not know the rows match `WHERE index_id = 4`. That is why the id is written into the SQL
  (D19; `tests/unit/test_indexes.py` checks a forced generic plan).
- **With the limit as a parameter**, Postgres judged the generic plan worse and kept re-planning
  every search (7 custom plans, 0 generic). Written in, it settles on a generic plan that still
  uses the index, and planning is skipped. A search after 259 others on one pooled connection:

```
Prepared as _pg3_0: 5 custom plans, then 254 generic.

->  Limit (actual time=0.300..0.581 rows=20 loops=1)
      ->  Index Scan using chunks_hnsw_4 on chunks c (actual time=0.299..0.561 rows=20 loops=1)
            Order By: ((embedding)::vector(384) <=> ($1)::vector(384))
Execution Time: 0.731 ms
```

Both values are ints from the code, never user input.

## Two traps, found while measuring

- **A prepared statement keeps its plan when planner settings change.** A search run with
  `enable_indexscan = off` (the exact store) can leave a sequential-scan plan cached on the
  connection for the searches after it. The app never turns index scans off; the CI gate's exact
  store is the only user, and it runs alone.
- **`SET LOCAL` on a pooled connection is only as local as its transaction.** psycopg's pool hands
  out connections outside autocommit: if anything runs before `connection.transaction()`, that
  block becomes a savepoint inside an implicit transaction, and the setting lasts until the
  connection goes back to the pool. The store always opens its transaction first, so its
  `SET LOCAL` ends with the search. The first versions of the measuring script did not, and
  reported sequential scans that the app never runs.
