"""Postgres full-text search as a sparse retriever, for comparison with bm25s only (D17).

`ts_rank` scores term frequency in a chunk but has no inverse document frequency, so a rare
token such as "7A" or a figure counts no more than "company". The query ORs the question's words,
as BM25 does; `to_tsquery` stems them and drops stop words with the same `english` configuration
that builds the `tsv` column (the chunk's header and text).
"""

import re

from groundwork.domain.models import Access, ScoredPassage, make_chunk_id
from groundwork.retrieval.pgvector_store import PgVectorStore

_WORD = re.compile(r"[a-z0-9]+")


def or_query(question: str) -> str:
    """ "What was ACME's revenue?" -> "what | was | acme | s | revenue" """
    return " | ".join(_WORD.findall(question.lower()))


async def ts_rank_search(
    store: PgVectorStore, question: str, n: int, access: Access
) -> list[ScoredPassage]:
    query = or_query(question)
    if not query:
        return []
    async with store.pool.connection() as connection:
        cursor = await connection.execute(
            f"SELECT c.document_id, c.ordinal FROM chunks c, to_tsquery('english', %s) q "
            f"WHERE c.index_id = {int(store.index.id)} AND c.tsv @@ q "
            f"ORDER BY ts_rank(c.tsv, q) DESC, c.id LIMIT %s",
            (query, n),
        )
        rows = await cursor.fetchall()
    ids = [make_chunk_id(str(doc_id), ordinal) for doc_id, ordinal in rows]
    # Unranked when the user may not read them, as the other retrievers drop them before the cut.
    stored = await store.get(ids, access=access)
    visible = [chunk_id for chunk_id in ids if chunk_id in stored]
    return [
        ScoredPassage(*stored[chunk_id], None, rank, 0.0, None)
        for rank, chunk_id in enumerate(visible, start=1)
    ]
