from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from groundwork.domain.models import Chunk, Document, SearchFilters
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.sparse import Positions, SparseCache, SparseIndex

DOCUMENT = Document(
    title="Report",
    collection_id="uploads",
    doc_type="10-K",
    doc_date=None,
    metadata={"company": "ACME Corp", "ticker": "ACME"},
    filename="acme.txt",
    doc_id="doc",
    ingested_at=datetime(2025, 1, 1, tzinfo=UTC),
)


def make_passages(*texts: str) -> list[tuple[Chunk, Document]]:
    return [
        (Chunk(f"doc:{i}", "doc", text, (), None, 0, 1), DOCUMENT) for i, text in enumerate(texts)
    ]


CHUNKS = make_passages(
    "ACME sells anvils to coyotes",
    "The warehouse roof leaks in winter",
    "Anvil sales and anvil returns rose",
)


def ids(hits: list[tuple[str, float]]) -> list[str]:
    return [chunk_id for chunk_id, _ in hits]


def test_stemming_matches_word_forms() -> None:
    assert "doc:0" in ids(SparseIndex.build(CHUNKS).search("selling", 5))


def test_best_match_comes_first() -> None:
    assert ids(SparseIndex.build(CHUNKS).search("anvil", 5))[0] == "doc:2"


def test_chunks_without_a_query_term_are_not_returned() -> None:
    assert ids(SparseIndex.build(CHUNKS).search("warehouse", 5)) == ["doc:1"]


def test_query_of_stopwords_returns_nothing() -> None:
    assert SparseIndex.build(CHUNKS).search("the and of", 5) == []


def test_unknown_word_returns_nothing() -> None:
    assert SparseIndex.build(CHUNKS).search("zebra", 5) == []


def test_results_respect_n() -> None:
    assert len(SparseIndex.build(CHUNKS).search("anvil", 1)) == 1


OTHER = replace(DOCUMENT, doc_id="other", collection_id="news", metadata={"ticker": "BETA"})
MIXED = [
    *CHUNKS,
    (Chunk("other:0", "other", "BETA anvil anvil anvil", (), None, 0, 1), OTHER),
]


def test_filters_apply_before_n() -> None:
    """The best match overall is BETA's; filtered to ACME, n=1 still finds ACME's best."""
    index = SparseIndex.build(MIXED)

    assert ids(index.search("anvil", 1)) == ["other:0"]
    assert ids(index.search("anvil", 1, SearchFilters(collections=("uploads",)))) == ["doc:2"]
    assert ids(index.search("anvil", 5, SearchFilters(metadata={"ticker": "BETA"}))) == ["other:0"]


def test_a_filter_matching_nothing_finds_nothing() -> None:
    index = SparseIndex.build(MIXED)

    assert index.search("anvil", 5, SearchFilters(collections=("nowhere",))) == []


def test_filters_survive_save_and_load(tmp_path: Path) -> None:
    SparseIndex.build(MIXED).save(tmp_path)

    loaded = SparseIndex.load(tmp_path)

    assert ids(loaded.search("anvil", 5, SearchFilters(collections=("news",)))) == ["other:0"]


def test_groups_get_one_bit_each_per_build() -> None:
    hr = replace(OTHER, access_groups=("everyone", "hr"))
    positions = Positions.from_passages([CHUNKS[0], (MIXED[3][0], hr)])

    assert positions.group_bit == {"everyone": 0, "hr": 1}
    assert positions.group_bits.tolist() == [0b01, 0b11]


class GenerationStore(MemoryStore):
    """Counts how often BM25 is rebuilt from it."""

    builds = 0

    async def all_passages(self) -> list[tuple[Chunk, Document]]:
        self.builds += 1
        return await super().all_passages()


async def store_with(*texts: str) -> GenerationStore:
    store = GenerationStore(memory_index(FakeEmbedder().dimension))
    chunks = [Chunk("x", "x", text, (), None, 0, 1) for text in texts]
    await store.add(DOCUMENT, "hash", chunks, FakeEmbedder().embed_documents(list(texts)))
    return store


async def add_beta(store: GenerationStore) -> None:
    await store.add(OTHER, "beta", [Chunk("x", "x", "BETA anvils", (), None, 0, 1)], [[1.0] * 64])


async def test_the_cache_rebuilds_only_when_the_generation_moves(tmp_path: Path) -> None:
    store = await store_with("ACME sells anvils")
    now = [0.0]
    cache = SparseCache(store, tmp_path, check_seconds=1.0, clock=lambda: now[0])

    first = await cache.current()
    await add_beta(store)
    same = await cache.current()
    await store.publish()
    still_cached = await cache.current()
    now[0] = 2.0
    rebuilt = await cache.current()

    assert same is first and still_cached is first
    assert len(rebuilt) == 2
    assert store.builds == 2


async def test_a_second_worker_loads_the_saved_generation(tmp_path: Path) -> None:
    store = await store_with("ACME sells anvils")
    await SparseCache(store, tmp_path).current()

    loaded = await SparseCache(store, tmp_path).current()

    assert store.builds == 1
    assert ids(loaded.search("anvils", 5)) == ["1:0"]
    assert sorted(p.name for p in (tmp_path / "1").iterdir()) == ["0"]


async def test_old_generations_are_removed(tmp_path: Path) -> None:
    store = await store_with("ACME sells anvils")
    for _ in range(3):
        await SparseCache(store, tmp_path).current()
        await add_beta(store)
        await store.publish()

    assert sorted(p.name for p in (tmp_path / "1").iterdir()) == ["1", "2"]


def test_document_title_and_section_are_searchable() -> None:
    chunk = Chunk("doc:0", "doc", "Headcount grew.", ("Item 1 — Business",), None, 0, 1)
    index = SparseIndex.build([(chunk, DOCUMENT)])

    assert ids(index.search("report", 5)) == ["doc:0"]
    assert ids(index.search("business", 5)) == ["doc:0"]


def test_empty_index_returns_nothing(tmp_path: Path) -> None:
    index = SparseIndex.build([])
    index.save(tmp_path)

    assert index.search("anvil", 5) == []
    assert SparseIndex.load(tmp_path).search("anvil", 5) == []


def test_save_and_load_keep_results(tmp_path: Path) -> None:
    index = SparseIndex.build(CHUNKS)
    index.save(tmp_path)

    assert SparseIndex.load(tmp_path).search("anvil", 5) == index.search("anvil", 5)
