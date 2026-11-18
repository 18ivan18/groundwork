from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from groundwork.config import Settings
from groundwork.container import Container
from groundwork.domain.models import Chunk, Document
from groundwork.main import create_app
from groundwork.retrieval.embedder import FakeEmbedder


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, data_dir=tmp_path)))


def test_health_reports_configuration(tmp_path: Path) -> None:
    response = make_client(tmp_path).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_provider": "openai_compatible",
        "embedder": "BAAI/bge-small-en-v1.5",
        "documents": 0,
        "chunks": 0,
    }


def test_health_counts_what_is_indexed(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    container: Container = client.app.state.container  # type: ignore[attr-defined]
    document = Document(
        title="ACME Corp · Form 10-K",
        collection_id="uploads",
        doc_type="10-K",
        doc_date=None,
        metadata={"company": "ACME Corp", "ticker": "ACME"},
        filename="acme.txt",
        doc_id="doc",
        ingested_at=datetime.now(UTC),
    )
    chunks = [Chunk(f"doc:{i}", "doc", "anvils", (), None, 0, 6) for i in range(2)]
    container.dense_index.add(document, chunks, FakeEmbedder().embed_documents(["a", "b"]))

    body = client.get("/api/health").json()

    assert body["documents"] == 1
    assert body["chunks"] == 2
