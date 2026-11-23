"""The retrieval eval and the CI gate run in a database of their own, emptied by each run."""

import json
from pathlib import Path
from typing import Any

import pytest
import run_eval
from collection_builder import doc, write_collection

from groundwork.db.databases import (
    WrongDatabaseError,
    create_and_migrate,
    database_name,
    empty_eval_database,
    eval_database_url,
)
from groundwork.db.pool import Pool, open_pool
from groundwork.domain.models import Chunk, DocumentMeta
from groundwork.ingestion.corpora_bundle import LOCAL_CORPORA_ENV
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.indexes import IndexSpec, activate, ensure_index, get_index
from groundwork.retrieval.pgvector_store import PgVectorStore

QUESTIONS = [
    {
        "id": "n1",
        "collection": "northwind",
        "question": "parental leave",
        "type": "single",
        "documents": ["b-md"],
        "evidence": ["beta twenty weeks"],
    },
    {
        "id": "s1",
        "collection": "sec-10k",
        "question": "anvil revenue",
        "type": "single",
        "documents": ["acme-md"],
        "evidence": ["sold ten anvils"],
    },
]


def write_northwind(raw: Path, titles: dict[str, str]) -> None:
    folder = write_collection(
        raw, "northwind", [doc(name, title=title) for name, title in titles.items()]
    )
    (folder / "a.md").write_text("The alpha ten days apply to everyone who works at the company.")
    (folder / "b.md").write_text("The beta twenty weeks apply to everyone who works there.")


def retitle_northwind(raw: Path, titles: dict[str, str]) -> None:
    """Change the manifest only: the files keep their bytes."""
    manifest = raw / "northwind" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["documents"] = [doc(name, title=title) for name, title in titles.items()]
    manifest.write_text(json.dumps(data))


@pytest.fixture
def raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database_url: str) -> Path:
    """Local corpora for the gate, with the test database as the app's."""
    raw = tmp_path / "data" / "raw"
    folder = write_collection(raw, "sec-10k", [doc("acme.md", title="Anvil revenue")])
    (folder / "acme.md").write_text("ACME sold ten anvils to a coyote in the desert last year.")
    (tmp_path / "questions.jsonl").write_text("\n".join(map(json.dumps, QUESTIONS)))
    monkeypatch.setattr(run_eval, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(run_eval, "build_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(LOCAL_CORPORA_ENV, "local")
    monkeypatch.setenv("RERANKER_MODEL", "")
    # One passage decides each question, so the titles alone decide the score.
    monkeypatch.setenv("TOP_K", "1")
    return raw


async def app_data(pool: Pool) -> dict[str, list[Any]]:
    async with pool.connection() as connection:
        return {
            table: await (
                await connection.execute(f"SELECT t::text FROM {table} t ORDER BY 1")
            ).fetchall()
            for table in ("indexes", "documents", "chunks", "collections")
        }


async def add_app_document(pool: Pool) -> None:
    index = await ensure_index(
        pool, IndexSpec(name="app", chunker="window", embedder_model="fake", dimension=3)
    )
    await activate(pool, index.id)
    meta = DocumentMeta(
        title="Upload", collection_id="northwind", doc_type="note", doc_date=None, filename="u.md"
    )
    store = PgVectorStore(pool, index)
    await store.add(meta, "hash", [Chunk("x", "x", "text", (), None, 0, 4)], [[1.0, 0.0, 0.0]])


@pytest.mark.db
async def test_each_run_scores_the_corpora_as_they_are_and_leaves_the_app_alone(
    pool: Pool, raw: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await add_app_document(pool)
    before = await app_data(pool)

    write_northwind(raw, {"a.md": "Parental leave", "b.md": "Travel"})
    await run_eval.ci(update_baseline=True)
    assert "northwind: Success@1 0.000" in capsys.readouterr().out

    retitle_northwind(raw, {"a.md": "Travel", "b.md": "Parental leave"})
    await run_eval.ci(update_baseline=True)
    assert "northwind: Success@1 1.000" in capsys.readouterr().out

    assert await app_data(pool) == before


@pytest.mark.db
async def test_emptying_refuses_every_database_but_the_eval_one(
    pool: Pool, database_url: str
) -> None:
    await add_app_document(pool)
    before = await app_data(pool)

    with pytest.raises(WrongDatabaseError, match=f"it is not {database_name(database_url)}_eval"):
        await empty_eval_database(pool, database_url)

    assert await app_data(pool) == before


@pytest.mark.db
async def test_emptying_refuses_an_app_database_whose_name_looks_like_an_eval_one(
    pool: Pool, database_url: str
) -> None:
    """Pointing DATABASE_URL at an eval database makes it the app's, so it isn't emptied."""
    eval_url = eval_database_url(database_url)
    create_and_migrate(eval_url)
    opened = await open_pool(eval_url)
    try:
        with pytest.raises(WrongDatabaseError):
            await empty_eval_database(opened, eval_url)
    finally:
        await opened.close()


@pytest.mark.db
async def test_the_last_runs_index_stays_until_the_next_run(
    raw: Path, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    write_northwind(raw, {"a.md": "Parental leave", "b.md": "Travel"})
    await run_eval.ci(update_baseline=True)

    opened = await open_pool(eval_database_url(database_url))
    try:
        index = await get_index(opened, "window-400-header-bge-small")
        assert index is not None
        assert await PgVectorStore(opened, index).count_documents() == 3
    finally:
        await opened.close()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@host:5434/groundwork", "postgresql://u:p@host:5434/groundwork_eval"),
        (
            "postgresql://u:p%40x@host/app?sslmode=require",
            "postgresql://u:p%40x@host/app_eval?sslmode=require",
        ),
    ],
)
def test_the_eval_database_is_named_after_the_apps_on_the_same_server(
    url: str, expected: str
) -> None:
    assert eval_database_url(url) == expected


@pytest.mark.parametrize(
    "url", ["postgresql://u:p@host:5434", "postgresql://u:p@host/app?dbname=other"]
)
def test_a_url_that_cannot_name_an_eval_database_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="database URL"):
        eval_database_url(url)
