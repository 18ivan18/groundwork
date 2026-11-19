import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from api_helpers import as_user
from fastapi import FastAPI
from pdf_builder import make_pdf
from upload_harness import ALICE, FORM, MAX_BYTES, TEXT, USERS, Harness

from groundwork.domain.models import SearchFilters
from groundwork.ingestion import pipeline
from groundwork.ingestion.jobs import Job
from groundwork.ingestion.worker import FAILED_ERROR
from groundwork.retrieval.embedder import FakeEmbedder


def test_upload_is_indexed_in_the_background(harness: Harness) -> None:
    response = harness.upload()

    assert response.status_code == 202
    body = response.json()
    # A new file gets its document ID once it is indexed.
    assert body["doc_id"] is None
    job = harness.job(body["job_id"])
    assert (job["status"], job["progress"], job["error"]) == ("indexed", 100, None)
    assert job["chunks"] == harness.count_chunks() > 0
    assert harness.generation() == 1


def test_uploaded_document_joins_the_library_with_its_tags(harness: Harness) -> None:
    harness.upload(collection="team-notes", doc_type="runbook", doc_date="2025-06-30")

    [document] = harness.client.get("/api/documents").json()

    assert document["title"] == "notes.txt"
    assert document["collection_id"] == "team-notes"
    assert document["doc_type"] == "runbook"
    assert document["doc_date"] == "2025-06-30"
    assert document["metadata"] == {}


def test_uploads_go_to_the_uploads_collection_by_default(harness: Harness) -> None:
    harness.upload()

    [document] = harness.client.get("/api/documents").json()

    assert document["collection_id"] == "uploads"
    assert document["doc_date"] is None


def test_uploaded_document_is_searchable_by_both_retrievers(harness: Harness) -> None:
    harness.upload()

    assert harness.search("blacksmiths", 3, mode="sparse")
    assert harness.search("blacksmiths", 3, mode="dense")


def test_uploads_can_be_searched_on_their_own(harness: Harness) -> None:
    harness.upload()
    uploads_only = SearchFilters(collections=("uploads",))
    filings_only = SearchFilters(collections=("sec-10k",))

    assert harness.search("blacksmiths", 3, uploads_only)
    assert harness.search("blacksmiths", 3, filings_only) == []


def test_html_upload_is_indexed_without_its_markup(harness: Harness) -> None:
    page = b"<html><body><div>" + TEXT + b"</div><script>var hidden = 1;</script></body></html>"

    job = harness.job(harness.upload(page, name="report.html").json()["job_id"])

    assert job["status"] == "indexed"
    [passage] = harness.search("blacksmiths", 1, mode="sparse")
    assert "<div>" not in passage.chunk.text
    assert "hidden" not in passage.chunk.text


def test_html_with_only_hidden_text_is_rejected_at_once(harness: Harness) -> None:
    page = b'<html><body><div style="display:none">' + TEXT + b"</div></body></html>"

    response = harness.upload(page, name="report.htm")

    assert response.status_code == 422


def test_markdown_and_uppercase_suffixes_are_accepted(harness: Harness) -> None:
    assert harness.upload(name="NOTES.MD").status_code == 202


def test_duplicate_returns_a_finished_job_and_adds_nothing(harness: Harness) -> None:
    first = harness.upload().json()
    chunks = harness.count_chunks()
    [document] = harness.client.get("/api/documents").json()

    second = harness.upload(name="renamed.txt").json()

    assert second["doc_id"] == document["doc_id"]
    assert second["job_id"] != first["job_id"]
    job = harness.job(second["job_id"])
    assert (job["status"], job["chunks"]) == ("indexed", chunks)
    assert harness.count_chunks() == chunks
    assert harness.generation() == 1


def test_identical_uploads_in_flight_index_once(harness: Harness) -> None:
    async def accept_both_then_index() -> list[Job]:
        jobs = [
            await harness.service.accept(
                "notes.txt",
                TEXT,
                owner=ALICE,
                collection_id="uploads",
                doc_type="notes",
                doc_date=None,
            )
            for _ in range(2)
        ]
        await harness.worker.drain()
        return jobs

    jobs = asyncio.run(accept_both_then_index())

    assert harness.generation() == 1
    assert harness.count_chunks() == len(asyncio.run(harness.store.all_passages()))
    assert [harness.job(job.job_id)["status"] for job in jobs] == ["indexed", "indexed"]


def test_too_large_file_is_rejected(harness: Harness) -> None:
    response = harness.upload(b"x" * (MAX_BYTES + 1))

    assert response.status_code == 413
    assert "limit" in response.json()["detail"]
    assert not harness.uploads_dir.exists()


def test_file_at_the_limit_is_accepted(harness: Harness) -> None:
    assert harness.upload((TEXT * 10)[:MAX_BYTES]).status_code == 202


def test_pdf_upload_is_searchable_with_page_numbers(harness: Harness) -> None:
    sentence = "Blacksmiths buy ACME anvils every spring for the season."
    pdf = make_pdf([["Cover page."], [sentence] * 8])

    response = harness.upload(pdf, "catalogue.pdf")

    assert harness.job(response.json()["job_id"])["status"] == "indexed", response.json()
    [passage] = harness.search("blacksmiths spring", 1, mode="sparse")
    # The cover is too short to be a chunk of its own, so the passage starts on page 2.
    assert passage.chunk.page == 2
    assert passage.document.filename == "catalogue.pdf"


def test_pdf_without_a_text_layer_is_rejected_at_once(harness: Harness) -> None:
    response = harness.upload(make_pdf([None]), "scan.pdf")

    assert response.status_code == 422
    assert "no text layer" in response.json()["detail"]
    assert harness.client.get("/api/documents").json() == []


@pytest.mark.parametrize("name", ["report.docx", "archive.txt.zip", "noextension"])
def test_unsupported_type_is_rejected(harness: Harness, name: str) -> None:
    response = harness.upload(name=name)

    assert response.status_code == 415
    assert ".htm, .html, .md, .pdf, .txt" in response.json()["detail"]


def test_file_with_no_text_is_rejected_at_once(harness: Harness) -> None:
    response = harness.upload(b"  \n\t \n")

    assert response.status_code == 422
    assert response.json() == {"detail": "No text was found in this file."}
    # glob, not iterdir: the folder itself is not created for a rejected file.
    assert list(harness.uploads_dir.glob("*")) == []


def test_file_too_short_to_chunk_fails_its_job(harness: Harness) -> None:
    job = harness.job(harness.upload(b"Too short.").json()["job_id"])

    assert job["status"] == "failed"
    assert job["error"] == "This file has too little text to index."
    assert harness.count_chunks() == 0


def test_an_indexing_crash_is_retried_then_dead_and_hides_the_cause(app: FastAPI) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("secret-path /etc/key")

    harness = Harness(app, BrokenEmbedder())

    job_id = harness.upload().json()["job_id"]

    job = harness.job(job_id)
    assert job["status"] == "dead"
    assert job["error"] == FAILED_ERROR
    stored = asyncio.run(harness.jobs.get(job_id))
    assert stored is not None and stored.attempts == stored.max_attempts == 3
    assert harness.client.get("/api/documents").json() == []


def test_a_failed_attempt_is_retried(app: FastAPI) -> None:
    class FlakyEmbedder(FakeEmbedder):
        failed = False

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            if not self.failed:
                self.failed = True
                raise RuntimeError("try again")
            return super().embed_documents(texts)

    harness = Harness(app, FlakyEmbedder())

    job_id = harness.upload().json()["job_id"]

    assert harness.job(job_id)["status"] == "indexed"
    stored = asyncio.run(harness.jobs.get(job_id))
    assert stored is not None and stored.attempts == 2


def test_progress_follows_the_embedding_batches(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "EMBED_BATCH", 1)
    seen: list[tuple[str, int]] = []

    class WatchingEmbedder(FakeEmbedder):
        """Slow enough per batch for a heartbeat to record each one."""

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            time.sleep(0.03)
            for job in asyncio.run(harness.jobs.list()):
                seen.append((job.status, job.progress))
            return super().embed_documents(texts)

    harness = Harness(app, WatchingEmbedder(), heartbeat_seconds=0.01)
    harness.upload()

    assert len(seen) > 1
    assert all(status == "embedding" for status, _ in seen)
    progress = [value for _, value in seen]
    assert progress == sorted(progress)
    assert 20 <= progress[0] < progress[-1] < 90


def test_path_in_the_filename_cannot_escape_the_uploads_folder(
    harness: Harness, tmp_path: Path
) -> None:
    harness.upload(name="../../evil.txt")
    harness.upload(TEXT + b"more", name="..\\..\\evil2.txt")

    names = {document["filename"] for document in harness.client.get("/api/documents").json()}
    assert names == {"evil.txt", "evil2.txt"}
    assert not (tmp_path.parent / "evil.txt").exists()
    assert len(list(harness.uploads_dir.iterdir())) == 2


@pytest.mark.parametrize(
    "form",
    [
        {"doc_type": ""},
        {"doc_type": "   "},
        {"doc_type": "x" * 51},
        {"collection": "Team Notes"},
        {"collection": "../etc"},
        {"doc_date": "last year"},
    ],
)
def test_invalid_tags_are_rejected(harness: Harness, form: dict[str, str]) -> None:
    assert harness.upload(**form).status_code == 422


def test_missing_file_is_rejected(harness: Harness) -> None:
    assert harness.client.post("/api/documents", data=FORM).status_code == 422


def test_unknown_job_is_a_404(harness: Harness) -> None:
    response = harness.client.get("/api/jobs/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "No such job."}


def test_an_upload_is_private_to_its_owner_by_default(harness: Harness) -> None:
    """#41."""
    job = harness.job(harness.upload(user="hana").json()["job_id"], user="hana")

    [document] = harness.documents("hana")
    assert document["access_groups"] == ["private:hana"]
    assert document["owner_id"] == "hana"
    assert harness.documents("alice") == []
    assert harness.search("blacksmiths", 5, access=USERS["alice"].access) == []
    # An admin reads everything.
    assert len(harness.documents("admin")) == 1
    assert harness.search("blacksmiths", 5, access=USERS["admin"].access)
    assert job["status"] == "indexed"


def test_an_upload_can_be_shared_with_the_uploaders_groups(harness: Harness) -> None:
    harness.upload(user="hana", groups=["hr"])

    assert [d["access_groups"] for d in harness.documents("admin")] == [["hr"]]
    assert harness.documents("filip") == []


def test_sharing_with_a_group_the_uploader_is_not_in_is_refused(harness: Harness) -> None:
    response = harness.upload(user="alice", groups=["everyone", "hr"])

    assert response.status_code == 403
    assert "not hr" in response.json()["detail"]
    assert harness.count_chunks() == 0


def test_the_same_file_from_another_user_is_a_separate_document(harness: Harness) -> None:
    """Deduplication is scoped: alice's upload must not reveal that hana has the file."""
    hana_job = harness.upload(user="hana", groups=["hr"]).json()
    alice_job = harness.upload(user="alice").json()

    # Not answered from hana's copy: a new job, indexed separately.
    assert alice_job["doc_id"] is None
    assert harness.job(alice_job["job_id"])["status"] == "indexed"
    [alices] = harness.documents("alice")
    assert alices["owner_id"] == "alice"
    assert [d["owner_id"] for d in harness.documents("hana")] == ["hana"]
    assert asyncio.run(harness.store.count_documents()) == 2
    assert hana_job["job_id"] != alice_job["job_id"]


def test_the_same_file_from_the_same_user_is_stored_once(harness: Harness) -> None:
    harness.upload(user="hana", groups=["hr"])

    again = harness.upload(user="hana", groups=["hr"]).json()

    assert again["doc_id"] is not None
    assert len(harness.documents("hana")) == 1


def test_another_users_job_is_not_found(harness: Harness) -> None:
    job_id = harness.upload(user="hana").json()["job_id"]

    response = as_user(harness.app, "alice").get(f"/api/jobs/{job_id}")

    assert response.status_code == 404
