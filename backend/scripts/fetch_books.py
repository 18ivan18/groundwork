"""Download three Project Gutenberg novels into data/raw/books/.

Each book is Gutenberg's plain-text UTF-8 file, kept byte for byte: the catalogue header and the
Project Gutenberg licence stay in the file, and only the text between its START and END lines is
indexed (see groundwork.ingestion.structure.books). The books are dated by first publication.

Gutenberg revises its files now and then ("Most recently updated" in each header), so a later run
can give different bytes; the manifest's sha256 values say which revision a corpus holds.

Run from backend/: uv run python scripts/fetch_books.py
"""

import hashlib
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from groundwork.config import Settings
from groundwork.ingestion.corpora_bundle import http_client
from groundwork.ingestion.corpus import MANIFEST, Manifest, ManifestDocument, write_manifest

COLLECTION = "books"
# The same files, tried in order: www.gutenberg.org is sometimes down for maintenance, and
# gutenberg.pglaf.org is the Project Gutenberg Literary Archive Foundation's own server.
SOURCES = ("https://www.gutenberg.org/cache/epub", "https://gutenberg.pglaf.org/cache/epub")
SECONDS_BETWEEN_REQUESTS = 1.0

_START = re.compile(rb"^\*\*\* ?START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
_END = re.compile(rb"^\*\*\* ?END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)


@dataclass(frozen=True)
class Book:
    key: str
    ebook_number: str
    title: str
    author: str
    first_published: date

    @property
    def filename(self) -> str:
        return f"pg{self.ebook_number}.txt"

    @property
    def source_url(self) -> str:
        return f"https://www.gutenberg.org/ebooks/{self.ebook_number}"


BOOKS = (
    Book("pride-and-prejudice", "1342", "Pride and Prejudice", "Jane Austen", date(1813, 1, 28)),
    Book(
        "frankenstein",
        "84",
        "Frankenstein; or, The Modern Prometheus",
        "Mary Wollstonecraft Shelley",
        date(1818, 1, 1),
    ),
    Book(
        "sherlock-holmes",
        "1661",
        "The Adventures of Sherlock Holmes",
        "Arthur Conan Doyle",
        date(1892, 10, 14),
    ),
)


def check_ebook(book: Book, content: bytes) -> None:
    """Refuse a download that is not the book's Gutenberg file, such as a maintenance page."""
    if f"[eBook #{book.ebook_number}]".encode() not in content:
        raise ValueError(f"{book.filename} does not name itself eBook #{book.ebook_number}")
    start = _START.search(content)
    if not start:
        raise ValueError(f"{book.filename} has no START OF THE PROJECT GUTENBERG EBOOK line")
    if not _END.search(content, start.end()):
        raise ValueError(f"{book.filename} has no END OF THE PROJECT GUTENBERG EBOOK line")


def build_manifest(contents: Mapping[str, bytes]) -> Manifest:
    """The collection's manifest, given each book's downloaded bytes by ebook number."""
    return Manifest(
        collection=COLLECTION,
        description="Three novels from Project Gutenberg: long, weakly structured prose for "
        "testing chunking. Dated by first publication.",
        source="Project Gutenberg",
        licence="Public domain in the United States. Each file keeps the Project Gutenberg "
        "licence; only the text between its START and END lines is indexed.",
        documents=tuple(
            ManifestDocument(
                key=book.key,
                filename=book.filename,
                title=book.title,
                doc_type="book",
                doc_date=book.first_published,
                metadata={"author": book.author},
                source_url=book.source_url,
                provenance={"ebook_number": book.ebook_number},
                sha256=hashlib.sha256(contents[book.ebook_number]).hexdigest(),
            )
            for book in BOOKS
        ),
    )


def download(client: httpx.Client, book: Book) -> bytes:
    """The book's plain-text file from the first source that serves it."""
    errors = []
    for source in SOURCES:
        url = f"{source}/{book.ebook_number}/{book.filename}"
        try:
            response = client.get(url)
            response.raise_for_status()
            check_ebook(book, response.content)
        except (httpx.HTTPError, ValueError) as error:
            errors.append(f"{url}: {error}")
            continue
        finally:
            time.sleep(SECONDS_BETWEEN_REQUESTS)
        return response.content
    raise RuntimeError(f"No source served {book.filename}:\n" + "\n".join(errors))


def fetch_books(client: httpx.Client, folder: Path) -> Manifest:
    """Download every book into folder and write its manifest there."""
    folder.mkdir(parents=True, exist_ok=True)
    contents = {}
    for book in BOOKS:
        content = download(client, book)
        (folder / book.filename).write_bytes(content)
        contents[book.ebook_number] = content
        print(f"{book.filename}: {len(content):,} bytes")
    manifest = build_manifest(contents)
    (folder / MANIFEST).write_text(write_manifest(manifest))
    return manifest


def main() -> None:
    settings = Settings()
    with http_client() as client:
        fetch_books(client, settings.data_dir / "raw" / COLLECTION)


if __name__ == "__main__":
    main()
