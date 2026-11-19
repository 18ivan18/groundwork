"""The Library API (#44, #45): list, detail, download, edit and delete, by who may do what."""

import asyncio

from api_helpers import as_user
from upload_harness import TEXT, USERS, Harness

from groundwork.domain.models import Access


def upload(harness: Harness, user: str = "alice", name: str = "notes.txt", **form: str) -> str:
    content = TEXT + name.encode()
    response = harness.upload(content, name, user=user, **form)
    job = harness.job(response.json()["job_id"], user)
    assert job["status"] == "indexed", job
    doc_id: str = job["doc_id"]
    return doc_id


def titles(harness: Harness, user: str, **params: object) -> list[str]:
    response = as_user(harness.app, user).get("/api/documents", params=params)
    assert response.status_code == 200
    return [document["title"] for document in response.json()]


# --- #44: list, detail, download ---


def test_the_list_hides_others_documents(harness: Harness) -> None:
    upload(harness, "alice", "alice.txt")
    upload(harness, "hana", "hana.txt", access_groups="hr")

    assert titles(harness, "alice") == ["alice.txt"]
    assert titles(harness, "hana") == ["hana.txt"]
    assert titles(harness, "admin") == ["hana.txt"]


def test_the_list_filters_searches_sorts_and_pages(harness: Harness) -> None:
    upload(harness, name="b-runbook.md", doc_type="runbook", collection="ops")
    upload(harness, name="a-notes.txt", doc_type="notes")
    upload(harness, name="c-runbook.md", doc_type="runbook")

    assert titles(harness, "alice", doc_type="runbook") == ["b-runbook.md", "c-runbook.md"]
    assert titles(harness, "alice", collection="ops") == ["b-runbook.md"]
    assert titles(harness, "alice", q="RUNBOOK", sort="-title") == ["c-runbook.md", "b-runbook.md"]
    response = as_user(harness.app).get("/api/documents", params={"offset": 1, "limit": 1})
    assert [d["title"] for d in response.json()] == ["b-runbook.md"]
    assert response.headers["X-Total-Count"] == "3"


def test_mine_lists_only_the_users_uploads(harness: Harness) -> None:
    upload(harness, "alice", "alice.txt", access_groups="everyone")
    upload(harness, "hana", "hana.txt", access_groups="everyone")

    assert titles(harness, "alice", mine=True) == ["alice.txt"]
    assert titles(harness, "alice") == ["alice.txt", "hana.txt"]


def test_the_detail_shows_indexes_jobs_and_rights(harness: Harness) -> None:
    doc_id = upload(harness)

    body = as_user(harness.app).get(f"/api/documents/{doc_id}").json()

    assert body["indexes"] == [harness.store.index.name]
    assert [job["status"] for job in body["jobs"]] == ["indexed"]
    assert body["jobs"][0]["filename"] == "notes.txt"
    assert (body["can_edit"], body["can_delete"], body["downloadable"]) == (True, True, True)


def test_a_reader_who_is_not_the_owner_cannot_change_it(harness: Harness) -> None:
    doc_id = upload(harness, "hana", access_groups="everyone")

    body = as_user(harness.app, "alice").get(f"/api/documents/{doc_id}").json()

    assert (body["can_edit"], body["can_delete"]) == (False, False)
    # The job is hana's: alice sees that there was one, not its file.
    assert body["jobs"][0]["filename"] is None


def test_another_users_private_document_is_not_found(harness: Harness) -> None:
    doc_id = upload(harness, "hana")

    for path in (f"/api/documents/{doc_id}", f"/api/documents/{doc_id}/file"):
        assert as_user(harness.app, "alice").get(path).status_code == 404
        assert as_user(harness.app, "admin").get(path).status_code == 404


def test_the_original_file_downloads(harness: Harness) -> None:
    doc_id = upload(harness, name="report.txt")

    response = as_user(harness.app).get(f"/api/documents/{doc_id}/file")

    assert response.status_code == 200
    assert response.content == TEXT + b"report.txt"
    assert 'filename="report.txt"' in response.headers["content-disposition"]


# --- #45: edit ---


def test_an_edit_changes_the_tags(harness: Harness) -> None:
    doc_id = upload(harness)

    response = as_user(harness.app).patch(
        f"/api/documents/{doc_id}", json={"title": "Field notes", "doc_date": "2025-03-01"}
    )

    assert response.status_code == 200
    assert (response.json()["title"], response.json()["doc_date"]) == (
        "Field notes",
        "2025-03-01",
    )
    assert response.json()["doc_type"] == "notes"


def test_a_new_group_takes_effect_on_the_next_search(harness: Harness) -> None:
    """Every API worker rebuilds BM25 once the generation moves, so sharing reaches all modes."""
    doc_id = upload(harness, "hana")
    hr_reader = Access.of(USERS["admin"].groups)
    assert harness.search("blacksmiths", 3, mode="sparse", access=hr_reader) == []
    generation = harness.generation()

    response = as_user(harness.app, "hana").patch(
        f"/api/documents/{doc_id}", json={"access_groups": ["hr"]}
    )

    assert response.json()["access_groups"] == ["hr"]
    assert harness.generation() == generation + 1
    for mode in ("dense", "sparse", "hybrid"):
        assert harness.search("blacksmiths", 3, mode=mode, access=hr_reader), mode
    assert titles(harness, "admin") == ["notes.txt"]


def test_groups_are_limited_to_the_editors_own(harness: Harness) -> None:
    doc_id = upload(harness)

    response = as_user(harness.app).patch(
        f"/api/documents/{doc_id}", json={"access_groups": ["finance"]}
    )

    assert response.status_code == 403


def test_only_the_owner_or_an_admin_may_edit(harness: Harness) -> None:
    doc_id = upload(harness, "hana", access_groups="everyone")

    by_alice = as_user(harness.app, "alice").patch(f"/api/documents/{doc_id}", json={"title": "x"})
    by_admin = as_user(harness.app, "admin").patch(f"/api/documents/{doc_id}", json={"title": "y"})

    assert by_alice.status_code == 404
    assert (by_admin.status_code, by_admin.json()["title"]) == (200, "y")


def test_an_edit_that_duplicates_a_document_is_refused(harness: Harness) -> None:
    upload(harness, "hana", access_groups="hr")
    private = upload(harness, "hana")

    response = as_user(harness.app, "hana").patch(
        f"/api/documents/{private}", json={"access_groups": ["hr"]}
    )

    assert response.status_code == 409


# --- #45: delete ---


def test_a_delete_by_a_non_owner_is_not_found(harness: Harness) -> None:
    doc_id = upload(harness, "hana", access_groups="everyone")

    response = as_user(harness.app, "alice").delete(f"/api/documents/{doc_id}")

    assert response.status_code == 404
    assert titles(harness, "alice") == ["notes.txt"]


def test_after_a_delete_nothing_returns_the_document(harness: Harness) -> None:
    doc_id = upload(harness)
    assert harness.search("blacksmiths", 3, mode="sparse")
    stored = list(harness.uploads_dir.iterdir())

    response = as_user(harness.app).delete(f"/api/documents/{doc_id}")

    assert response.status_code == 204
    for mode in ("dense", "sparse", "hybrid", "reranked"):
        assert harness.search("blacksmiths", 3, mode=mode) == [], mode
    assert titles(harness, "alice") == []
    assert as_user(harness.app).get(f"/api/documents/{doc_id}").status_code == 404
    assert all(not path.exists() for path in stored)


def test_a_file_another_document_still_names_is_kept(harness: Harness) -> None:
    private = upload(harness, "hana")
    upload(harness, "hana", access_groups="hr")

    as_user(harness.app, "hana").delete(f"/api/documents/{private}")

    assert len(list(harness.uploads_dir.iterdir())) == 1


def test_a_protected_collection_needs_an_admin(harness: Harness) -> None:
    doc_id = upload(harness, "alice", collection="sec-10k")
    asyncio.run(harness.store.protect_collection("sec-10k"))

    by_owner = as_user(harness.app, "alice").delete(f"/api/documents/{doc_id}")
    by_admin_unreadable = as_user(harness.app, "admin").delete(f"/api/documents/{doc_id}")

    assert by_owner.status_code == 403
    # Admin cannot read alice's private upload, so for admin it does not exist.
    assert by_admin_unreadable.status_code == 404


def test_bulk_delete_reports_what_it_could_not_delete(harness: Harness) -> None:
    mine = upload(harness, "alice", "alice.txt", access_groups="everyone")
    hanas = upload(harness, "hana", "hana.txt", access_groups="everyone")

    response = as_user(harness.app, "alice").post(
        "/api/documents/delete", json={"doc_ids": [mine, hanas, "999"]}
    )

    assert response.json() == {"deleted": [mine], "not_found": [hanas, "999"]}
    assert titles(harness, "alice") == ["hana.txt"]


def test_deleting_cancels_the_files_unfinished_ingestion(harness: Harness) -> None:
    """The same file is queued again (a retry, say) when the document is deleted: the job is
    cancelled, so the document does not come back."""
    doc_id = upload(harness)
    [done] = asyncio.run(harness.jobs.list())
    again = asyncio.run(harness.jobs.enqueue(done.kind, done.payload, owner_id="alice"))

    as_user(harness.app).delete(f"/api/documents/{doc_id}")
    harness.work()

    job = asyncio.run(harness.jobs.get(again.job_id))
    assert job is not None and job.status == "cancelled"
    assert harness.count_chunks() == 0
    assert titles(harness, "alice") == []
