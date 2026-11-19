"""The ACL test matrix (#47): no user ever receives a chunk from a document outside their groups,
through any retrieval mode, on either store, or through the API.

Three documents share the query's words. The restricted ones score higher, and the scripted
re-ranker prefers them too, so a leak anywhere would put them at the top.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import pytest
from conftest import empty_database
from fastapi.testclient import TestClient

from groundwork.api.deps import get_qa_service
from groundwork.config import Settings
from groundwork.domain.models import Access, Chunk, DocumentMeta
from groundwork.domain.users import load_users
from groundwork.llm.fake import FakeLLM
from groundwork.main import create_app
from groundwork.qa.service import QAService
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.hybrid import HybridRetriever, SearchMode
from groundwork.retrieval.indexes import IndexSpec, ensure_index
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.sparse import FixedSparse, build_sparse
from groundwork.retrieval.store import ChunkStore

USERS = load_users(Path(__file__).parents[2] / "conf" / "users.yaml")
QUERY = "bonus salary budget for 2026"

DOCUMENTS = {
    "handbook": (("everyone",), ["The 2026 handbook mentions a bonus and the salary review."]),
    "salaries": (("hr",), ["Salary bands 2026: bonus salary salary budget.", "Salary review."]),
    "budget": (("finance",), ["Budget 2026: bonus budget budget salary costs."]),
    "pay-slip": (("private:filip",), ["Filip's pay slip: Salary and bonus, Budget 2026."]),
}
# What each demo user may read, from conf/users.yaml.
VISIBLE = {
    "alice": {"handbook"},
    "hana": {"handbook", "salaries"},
    "filip": {"handbook", "budget", "pay-slip"},
    # Admin reads everything, private uploads included.
    "admin": {"handbook", "salaries", "budget", "pay-slip"},
}


class PreferRestricted:
    """A re-ranker that scores restricted text highest, so it would surface any leak."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [p.count("Salary") + p.count("Budget") * 1.0 for p in passages]


@dataclass(frozen=True)
class Library:
    store: ChunkStore
    retriever: HybridRetriever


async def fill(store: ChunkStore) -> Library:
    embedder = FakeEmbedder()
    for title, (groups, texts) in DOCUMENTS.items():
        meta = DocumentMeta(
            title=title,
            collection_id="northwind",
            doc_type="policy",
            doc_date=None,
            filename=f"{title}.md",
            access_groups=groups,
        )
        chunks = [Chunk("x", "x", text, (), None, 0, len(text)) for text in texts]
        await store.add(meta, title, chunks, embedder.embed_documents(texts))
    sparse = FixedSparse(await build_sparse(store))
    retriever = HybridRetriever(
        embedder, store, sparse, candidates=20, rrf_k=60, reranker=PreferRestricted()
    )
    return Library(store, retriever)


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.db)])
async def library(request: pytest.FixtureRequest) -> AsyncIterator[Library]:
    dimension = FakeEmbedder().dimension
    if request.param == "memory":
        yield await fill(MemoryStore(memory_index(dimension)))
        return
    async with empty_database(request.getfixturevalue("database_url")) as pool:
        spec = IndexSpec(name="acl", chunker="window", embedder_model="fake", dimension=dimension)
        yield await fill(PgVectorStore(pool, await ensure_index(pool, spec)))


def access(user: str) -> Access:
    return USERS[user].access


@pytest.mark.parametrize("mode", get_args(SearchMode))
@pytest.mark.parametrize("user", sorted(VISIBLE))
async def test_search_returns_exactly_what_the_user_may_read(
    library: Library, user: str, mode: SearchMode
) -> None:
    passages = await library.retriever.search(QUERY, 20, mode=mode, access=access(user))

    # Every chunk the user may read, and nothing else: a restrictive ACL still fills n.
    assert {p.document.title for p in passages} == VISIBLE[user]


@pytest.mark.parametrize("mode", get_args(SearchMode))
@pytest.mark.parametrize("user", sorted(VISIBLE))
async def test_each_retriever_applies_the_acl_before_its_top_n(
    library: Library, user: str, mode: SearchMode
) -> None:
    """With n=1 the best match overall is restricted. A retriever that left the ACL to the fetch
    by id would return nothing here, though nothing would leak."""
    [passage] = await library.retriever.search(QUERY, 1, mode=mode, access=access(user))

    assert passage.document.title in VISIBLE[user]


async def test_fetching_by_id_checks_the_acl_again(library: Library) -> None:
    every = {chunk.chunk_id: document for chunk, document in await library.store.all_passages()}

    fetched = await library.store.get(list(every), access=access("alice"))

    assert {document.title for _, document in fetched.values()} == {"handbook"}


@pytest.mark.parametrize("user", sorted(VISIBLE))
async def test_the_document_list_shows_what_the_user_may_read(library: Library, user: str) -> None:
    listed = await library.store.documents(access=access(user))

    assert {document.title for document, _ in listed} == VISIBLE[user]


@pytest.mark.parametrize("user", sorted(VISIBLE))
def test_the_api_answers_from_what_the_user_may_read(
    library: Library, tmp_path: Path, user: str
) -> None:
    app = create_app(Settings(_env_file=None, data_dir=tmp_path), store=library.store)
    service = QAService(
        library.retriever, FakeLLM(["See [1] and [2]."]), top_k=6, min_relevance=0.0, max_tokens=50
    )
    app.dependency_overrides[get_qa_service] = lambda: service
    client = TestClient(app, headers={"X-User": user})

    body = client.post("/api/ask", json={"question": QUERY}).json()
    listed = client.get("/api/documents").json()

    assert {p["title"] for p in body["retrieved"]} == VISIBLE[user]
    assert {s["title"] for s in body["sources"]} <= VISIBLE[user]
    assert {d["title"] for d in listed} == VISIBLE[user]
