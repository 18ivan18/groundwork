from pathlib import Path

import pytest
from ingest import main

from groundwork.ingestion.corpora_bundle import LOCAL_CORPORA_ENV


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
