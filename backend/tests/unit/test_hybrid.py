from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pytest

from groundwork.domain.models import UNRESTRICTED, Chunk, DocumentMeta, SearchFilters
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.hybrid import HybridRetriever, SearchMode
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.reranker import Reranker
from groundwork.retrieval.sparse import FixedSparse, build_sparse
from groundwork.retrieval.store import DenseHit


def make_meta(
    ticker: str,
    doc_type: str = "10-K",
    collection_id: str = "filings",
    doc_date: date | None = None,
) -> DocumentMeta:
    return DocumentMeta(
        title=f"{ticker} report",
        collection_id=collection_id,
        doc_type=doc_type,
        doc_date=doc_date,
        metadata={"ticker": ticker},
        filename=f"{ticker.lower()}.txt",
    )


BETA = make_meta("BETA", "8-K", collection_id="news", doc_date=date(2025, 6, 30))


async def add(store: MemoryStore, meta: DocumentMeta, *texts: str) -> str:
    chunks = [Chunk("x", "x", text, (), None, 0, 1) for text in texts]
    document = await store.add(meta, meta.title, chunks, FakeEmbedder().embed_documents(texts))
    return document.doc_id


@dataclass(frozen=True)
class Library:
    store: MemoryStore
    acme: str
    beta: str


@pytest.fixture
async def library() -> Library:
    store = MemoryStore(memory_index(FakeEmbedder().dimension))
    acme_meta = make_meta("ACME", doc_date=date(2024, 1, 31))
    acme = await add(store, acme_meta, "ACME sells anvils", "ACME roof leaks")
    beta = await add(store, BETA, "BETA sells anvils", "BETA hires engineers")
    return Library(store, acme, beta)


async def make_retriever(
    store: MemoryStore, reranker: Reranker | None = None, rerank_candidates: int = 20
) -> HybridRetriever:
    sparse = FixedSparse(await build_sparse(store))
    return HybridRetriever(
        FakeEmbedder(),
        store,
        sparse,
        candidates=10,
        rrf_k=60,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )


async def texts(
    store: MemoryStore,
    query: str,
    k: int,
    filters: SearchFilters | None = None,
    mode: SearchMode = "reranked",
) -> list[str]:
    retriever = await make_retriever(store)
    return [
        p.chunk.text for p in await retriever.search(query, k, filters, mode, access=UNRESTRICTED)
    ]


async def test_hybrid_reports_both_ranks(library: Library) -> None:
    passage = (
        await (await make_retriever(library.store)).search("anvils", 1, access=UNRESTRICTED)
    )[0]

    assert passage.dense_rank is not None
    assert passage.sparse_rank is not None
    assert passage.dense_similarity is not None
    assert passage.rrf_score > 0


async def test_dense_mode_has_no_sparse_ranks(library: Library) -> None:
    passages = await (await make_retriever(library.store)).search(
        "anvils", 4, mode="dense", access=UNRESTRICTED
    )

    assert all(p.sparse_rank is None for p in passages)
    assert [p.dense_rank for p in passages] == [1, 2, 3, 4]


async def test_sparse_mode_finds_chunks_without_any_dense_data(library: Library) -> None:
    retriever = await make_retriever(library.store)

    passages = await retriever.search("engineers", 4, mode="sparse", access=UNRESTRICTED)

    assert [p.chunk.chunk_id for p in passages] == [f"{library.beta}:1"]
    assert passages[0].dense_rank is None
    assert passages[0].dense_similarity is None
    assert passages[0].document.metadata["ticker"] == "BETA"


async def test_chunk_found_only_by_bm25_is_fetched_in_hybrid_mode(
    library: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = await make_retriever(library.store)

    async def nothing(*_: object, **__: object) -> list[DenseHit]:
        return []

    monkeypatch.setattr(library.store, "search", nothing)

    passage = (await retriever.search("engineers", 4, access=UNRESTRICTED))[0]

    assert passage.chunk.text == "BETA hires engineers"
    assert passage.document.metadata["ticker"] == "BETA"
    assert passage.dense_rank is None
    assert passage.sparse_rank == 1


async def test_k_limits_results(library: Library) -> None:
    assert len(await texts(library.store, "anvils", 2)) == 2


# Each filter field, alone, keeps only BETA's passages.
FILTERS_FOR_BETA = {
    "collection": SearchFilters(collections=("news",)),
    "doc_type": SearchFilters(doc_types=("8-K",)),
    "date_from": SearchFilters(date_from=date(2025, 1, 1)),
    "date_to_inclusive": SearchFilters(date_from=date(2025, 6, 30), date_to=date(2025, 6, 30)),
    "metadata": SearchFilters(metadata={"ticker": "BETA"}),
}


@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
@pytest.mark.parametrize("name", FILTERS_FOR_BETA)
async def test_each_filter_applies_in_every_mode(
    library: Library, mode: SearchMode, name: str
) -> None:
    retriever = await make_retriever(library.store)

    passages = await retriever.search(
        "anvils", 4, FILTERS_FOR_BETA[name], mode, access=UNRESTRICTED
    )

    assert passages
    assert {p.document.metadata["ticker"] for p in passages} == {"BETA"}


async def test_filters_combine(library: Library) -> None:
    filters = SearchFilters(collections=("filings", "news"), doc_types=("8-K",))

    passages = await (await make_retriever(library.store)).search(
        "anvils", 4, filters, access=UNRESTRICTED
    )

    assert {p.document.doc_type for p in passages} == {"8-K"}


async def test_an_undated_document_never_matches_a_date_range() -> None:
    store = MemoryStore(memory_index(FakeEmbedder().dimension))
    await add(store, make_meta("UNDATED"), "UNDATED sells anvils")

    assert await texts(store, "anvils", 4, SearchFilters(date_to=date(2100, 1, 1))) == []


async def test_empty_filters_restrict_nothing(library: Library) -> None:
    assert len(await texts(library.store, "anvils", 4, SearchFilters())) == 4


async def test_filter_matching_nothing_returns_nothing(library: Library) -> None:
    filters = SearchFilters(metadata={"ticker": "NOPE"})

    assert await texts(library.store, "anvils", 4, filters) == []


async def test_a_chunk_gone_from_the_store_since_bm25_was_built_is_skipped(
    library: Library,
) -> None:
    retriever = await make_retriever(library.store)
    # BETA is stored again with no chunks: BM25 still has the old ones.
    await library.store.add(BETA, BETA.title, [], [])

    assert await retriever.search("engineers", 4, mode="sparse", access=UNRESTRICTED) == []


class ScriptedReranker:
    """Scores passages by a fixed list of preferred words, and records what it was given."""

    def __init__(self, *preferred: str) -> None:
        self.preferred = preferred
        self.seen: list[str] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.seen.extend(passages)
        return [float(sum(word in p for word in self.preferred)) for p in passages]


async def test_the_reranker_sets_the_order(library: Library) -> None:
    retriever = await make_retriever(library.store, ScriptedReranker("roof"), 4)

    passages = await retriever.search("ACME anvils", 4, access=UNRESTRICTED)

    assert passages[0].chunk.text == "ACME roof leaks"
    assert [p.rerank_rank for p in passages] == [1, 2, 3, 4]
    assert passages[0].rerank_score == 1.0


async def test_the_reranker_reads_the_title_with_the_passage(library: Library) -> None:
    reranker = ScriptedReranker()

    await (await make_retriever(library.store, reranker, 4)).search(
        "ACME anvils", 1, access=UNRESTRICTED
    )

    assert "ACME report\nACME sells anvils" in reranker.seen


async def test_only_the_top_candidates_are_reranked(library: Library) -> None:
    reranker = ScriptedReranker("engineers")
    unranked = await (await make_retriever(library.store)).search(
        "ACME anvils", 4, access=UNRESTRICTED
    )

    passages = await (await make_retriever(library.store, reranker, 2)).search(
        "ACME anvils", 4, access=UNRESTRICTED
    )

    assert len(reranker.seen) == 2
    assert {p.chunk.chunk_id for p in passages[:2]} == {p.chunk.chunk_id for p in unranked[:2]}
    assert passages[2:] == unranked[2:]
    assert [p.rerank_rank for p in passages[2:]] == [None, None]


async def test_hybrid_mode_skips_the_reranker(library: Library) -> None:
    reranker = ScriptedReranker("roof")
    retriever = await make_retriever(library.store, reranker, 4)

    passages = await retriever.search("ACME anvils", 4, mode="hybrid", access=UNRESTRICTED)

    assert reranker.seen == []
    assert passages == await (await make_retriever(library.store)).search(
        "ACME anvils", 4, access=UNRESTRICTED
    )


async def test_best_similarity_counts_candidates_the_reranker_pushed_out(
    library: Library,
) -> None:
    retriever = await make_retriever(library.store, ScriptedReranker("engineers"), 4)
    plain = await make_retriever(library.store)
    best = (await plain.search("ACME sells anvils", 1, mode="dense", access=UNRESTRICTED))[0]

    result = await retriever.retrieve("ACME sells anvils", 1, access=UNRESTRICTED)

    assert result.passages[0].chunk.text == "BETA hires engineers"
    assert result.best_similarity == best.dense_similarity
