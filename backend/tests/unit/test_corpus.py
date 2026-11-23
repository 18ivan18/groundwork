import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from collection_builder import doc, write_collection

from groundwork.domain.models import EVERYONE
from groundwork.ingestion.corpus import (
    SEC_COLLECTION,
    ManifestError,
    load_corpus,
    read_manifest,
)

# Small excerpts of each corpus format, with manifests shaped like the real ones.
FIXTURES = Path(__file__).parents[1] / "fixtures" / "corpora"


def test_documents_get_their_collection_and_manifest_fields(tmp_path: Path) -> None:
    write_collection(
        tmp_path,
        "northwind",
        [
            doc(
                doc_date="2026-03-01",
                access_groups=["hr"],
                metadata={"team": "people"},
                planted=["contradiction"],
            )
        ],
    )

    [(path, meta)] = load_corpus(tmp_path)

    assert path == tmp_path / "northwind" / "a.md"
    assert meta.collection_id == "northwind"
    assert (meta.title, meta.doc_type, meta.doc_date) == ("A", "runbook", date(2026, 3, 1))
    assert meta.access_groups == ("hr",)
    assert meta.metadata == {"team": "people"}


def test_access_defaults_to_everyone(tmp_path: Path) -> None:
    write_collection(tmp_path, "notes", [doc()])

    [(_, meta)] = load_corpus(tmp_path)

    assert meta.access_groups == (EVERYONE,)


def test_formats_without_a_loader_are_left_out(tmp_path: Path) -> None:
    write_collection(tmp_path, "notes", [doc("a.txt"), doc("a.HTM"), doc("a.docx")])

    assert [path.name for path, _ in load_corpus(tmp_path)] == ["a.txt", "a.HTM"]


def test_collections_can_be_chosen(tmp_path: Path) -> None:
    write_collection(tmp_path, "one", [doc("a.md")])
    write_collection(tmp_path, "two", [doc("b.md")])

    assert [path.name for path, _ in load_corpus(tmp_path, frozenset({"two"}))] == ["b.md"]
    with pytest.raises(ManifestError, match="No manifest for three"):
        load_corpus(tmp_path, frozenset({"three"}))


@pytest.mark.parametrize(
    ("documents", "fields", "problem"),
    [
        ([doc(access_groups=[])], {}, "at least one group"),
        ([doc(access_groups=["Human Resources"])], {}, "access_groups"),
        ([doc(doc_type=" ")], {}, "doc_type"),
        ([doc(doc_date="March 2026")], {}, "doc_date"),
        ([doc(colour="red")], {}, "colour"),
        ([doc(sha256="abc")], {}, "sha256"),
        ([doc("a.md"), doc("a.md", key="other")], {}, "filenames listed twice"),
        ([doc("a.md"), doc("b.md", key="a-md")], {}, "keys listed twice"),
        ([doc(key="A md")], {}, "key"),
        ([doc("../a.md", key="up")], {}, "plain file name"),
        ([doc()], {"licence": ""}, "licence"),
    ],
)
def test_malformed_manifests_are_refused(
    tmp_path: Path, documents: list[dict[str, Any]], fields: dict[str, Any], problem: str
) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    manifest = {
        "collection": "notes",
        "description": "d",
        "source": "s",
        "licence": "l",
        "documents": documents,
        **fields,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "a.md").write_text("text")

    with pytest.raises(ManifestError, match=problem):
        read_manifest(folder)


def test_collection_must_match_its_folder(tmp_path: Path) -> None:
    folder = write_collection(tmp_path, "notes", [doc()], collection="other")

    with pytest.raises(ManifestError, match="not the folder name"):
        read_manifest(folder)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    folder = write_collection(tmp_path, "notes", [doc()])
    (folder / "a.md").unlink()

    with pytest.raises(ManifestError, match=r"a\.md is missing"):
        read_manifest(folder)


def test_a_changed_file_is_refused(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"text").hexdigest()
    folder = write_collection(tmp_path, "notes", [doc(sha256=digest)])
    read_manifest(folder)

    (folder / "a.md").write_text("edited")

    with pytest.raises(ManifestError, match="does not match its sha256"):
        read_manifest(folder)


# The test fixtures


@pytest.mark.parametrize("name", ["books", "engineering", SEC_COLLECTION])
def test_fixture_manifests_are_valid_and_hashed(name: str) -> None:
    manifest = read_manifest(FIXTURES / name)

    assert manifest.documents
    assert all(document.sha256 for document in manifest.documents)


def test_filings_keep_their_metadata() -> None:
    filings = [meta for _, meta in load_corpus(FIXTURES, frozenset({SEC_COLLECTION}))]

    assert len(filings) == 3
    assert {meta.doc_type for meta in filings} == {"10-K"}
    assert {meta.metadata["form"] for meta in filings} == {"10-K", "10-K405"}
    assert {meta.metadata["ticker"] for meta in filings} == {"AAPL", "MSFT", "NVDA"}
    nvda_2001 = next(m for m in filings if m.metadata["form"] == "10-K405")
    assert nvda_2001.title == "NVIDIA Corporation · Form 10-K"
    assert nvda_2001.doc_date == date(2001, 1, 28)


def test_books_are_public_and_typed_as_books() -> None:
    books = [meta for _, meta in load_corpus(FIXTURES, frozenset({"books"}))]

    assert {meta.title for meta in books} == {
        "Pride and Prejudice",
        "Frankenstein; or, The Modern Prometheus",
        "The Adventures of Sherlock Holmes",
    }
    assert all(meta.doc_type == "book" and meta.access_groups == (EVERYONE,) for meta in books)
