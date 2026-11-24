from collections.abc import Sequence
from pathlib import Path

import ingest
import pytest
from collection_builder import doc, write_collection
from ingest import main

from groundwork.db.pool import Pool
from groundwork.ingestion.corpora_bundle import LOCAL_CORPORA_ENV
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.pgvector_store import PgVectorStore


async def test_without_corpus_documents_ingest_fails_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LOCAL_CORPORA_ENV, "local")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Unreachable, so a regression can't index anything into a developer's database.
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:1/none")

    with pytest.raises(SystemExit) as stopped:
        await main()

    assert f"No corpus documents in {tmp_path / 'raw'}" in str(stopped.value)


class FailsOnDisks(FakeEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any("disks" in text for text in texts):
            raise RuntimeError(f"cannot embed {texts}")
        return super().embed_documents(texts)


def write_version(
    data_dir: Path, texts: dict[str, str], groups: Sequence[str] = ("everyone",)
) -> None:
    documents = [doc(name, title=name.split(".")[0], access_groups=list(groups)) for name in texts]
    write_collection(data_dir / "raw", "runbooks", documents, texts=texts)


@pytest.fixture
def use_test_database(database_url: str, pool: Pool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingest into the emptied test database, from local corpora, failing on one word."""
    monkeypatch.setenv(LOCAL_CORPORA_ENV, "local")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(ingest, "build_embedder", lambda settings: FailsOnDisks())


ABOUT_PAGERS = "The runbook explains pagers to whoever is on call tonight. " * 30
ABOUT_DISKS = ABOUT_PAGERS.replace("pagers", "disks")


@pytest.mark.db
@pytest.mark.usefixtures("use_test_database")
async def test_ingest_indexes_the_rest_and_fails_naming_each_failed_document(
    tmp_path: Path, pool: Pool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    texts = {"disk.md": ABOUT_DISKS, "pager.md": ABOUT_PAGERS, "platter.md": ABOUT_DISKS}
    write_version(tmp_path, texts)

    with pytest.raises(SystemExit) as stopped:
        await main()

    message = str(stopped.value)
    assert "2 documents failed" in message
    for name in ("runbooks/disk-md", "runbooks/platter-md"):
        assert f"{name}: RuntimeError" in message
    assert "pager" not in message
    printed = capsys.readouterr()
    assert "1 added" in printed.out and "2 failed" in printed.out
    assert "on call" not in printed.out + printed.err + message


@pytest.mark.db
@pytest.mark.usefixtures("use_test_database")
async def test_ingest_says_loudly_that_a_document_is_still_found_under_its_old_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "v1"))
    write_version(tmp_path / "v1", {"pager.md": ABOUT_PAGERS})
    await main()
    write_version(tmp_path / "v2", {"pager.md": ABOUT_DISKS}, groups=["hr"])
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "v2"))

    async def fails(self: PgVectorStore, doc_ids: Sequence[str]) -> None:
        raise ConnectionError("the database went away")

    monkeypatch.setattr(PgVectorStore, "remove_corpus_documents", fails)
    with pytest.raises(SystemExit) as stopped:
        await main()

    assert "STILL READABLE BY ITS OLD ACCESS GROUPS" in str(stopped.value)
    assert "runbooks/pager-md" in str(stopped.value)
