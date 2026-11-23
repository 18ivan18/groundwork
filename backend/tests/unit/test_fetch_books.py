from datetime import date
from pathlib import Path

import pytest
from fetch_books import BOOKS, COLLECTION, build_manifest, check_ebook

from groundwork.domain.models import EVERYONE
from groundwork.ingestion.corpus import MANIFEST, read_manifest, write_manifest

# sha256 of b"fixture"
FIXTURE_SHA256 = "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"


def test_manifest_lists_each_book_with_its_gutenberg_source_and_hash() -> None:
    manifest = build_manifest({book.ebook_number: b"fixture" for book in BOOKS})

    assert manifest.collection == "books"
    assert [document.key for document in manifest.documents] == [
        "pride-and-prejudice",
        "frankenstein",
        "sherlock-holmes",
    ]
    frankenstein = manifest.documents[1]
    assert frankenstein.filename == "pg84.txt"
    assert frankenstein.title == "Frankenstein; or, The Modern Prometheus"
    assert frankenstein.doc_type == "book"
    assert frankenstein.doc_date == date(1818, 1, 1)
    assert frankenstein.metadata == {"author": "Mary Wollstonecraft Shelley"}
    assert frankenstein.source_url == "https://www.gutenberg.org/ebooks/84"
    assert frankenstein.provenance == {"ebook_number": "84"}
    assert frankenstein.access_groups == (EVERYONE,)
    assert frankenstein.sha256 == FIXTURE_SHA256


def test_written_manifest_is_read_back_by_the_corpus_reader(tmp_path: Path) -> None:
    folder = tmp_path / COLLECTION
    folder.mkdir()
    contents = {book.ebook_number: f"Book {book.ebook_number}".encode() for book in BOOKS}
    for book in BOOKS:
        (folder / book.filename).write_bytes(contents[book.ebook_number])
    manifest = build_manifest(contents)
    (folder / MANIFEST).write_text(write_manifest(manifest))

    assert read_manifest(folder) == manifest


EBOOK = (
    b"The Project Gutenberg eBook of Anvils\r\n"
    b"\r\n"
    b"Release date: June 1, 1998 [eBook #84]\r\n"
    b"\r\n"
    b"*** START OF THE PROJECT GUTENBERG EBOOK ANVILS ***\r\n"
    b"Chapter 1\r\n"
    b"*** END OF THE PROJECT GUTENBERG EBOOK ANVILS ***\r\n"
    b"Licence.\r\n"
)
FRANKENSTEIN = next(book for book in BOOKS if book.ebook_number == "84")


def test_the_ebook_itself_is_accepted() -> None:
    check_ebook(FRANKENSTEIN, EBOOK)


def test_a_maintenance_page_is_refused() -> None:
    page = b"<!DOCTYPE html><title>Temporarily Down for Maintenance</title>"

    with pytest.raises(ValueError, match=r"pg84\.txt"):
        check_ebook(FRANKENSTEIN, page)


def test_another_ebook_is_refused() -> None:
    with pytest.raises(ValueError, match="#84"):
        check_ebook(FRANKENSTEIN, EBOOK.replace(b"[eBook #84]", b"[eBook #841]"))


def test_an_ebook_without_its_end_line_is_refused() -> None:
    truncated = EBOOK.split(b"*** END")[0]

    with pytest.raises(ValueError, match="END"):
        check_ebook(FRANKENSTEIN, truncated)
