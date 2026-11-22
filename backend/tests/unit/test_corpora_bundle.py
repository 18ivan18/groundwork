import hashlib
import os
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest
from collection_builder import doc, write_collection

from groundwork.ingestion.corpora_bundle import pack
from groundwork.ingestion.corpus import ManifestError, corpus_entries

TEXT_SHA256 = hashlib.sha256(b"text").hexdigest()


def two_collections(raw: Path) -> None:
    write_collection(raw, "northwind", [doc("a.md", sha256=TEXT_SHA256)])
    write_collection(
        raw, "books", [doc("b.txt", sha256=TEXT_SHA256), doc("c.txt", sha256=TEXT_SHA256)]
    )


def test_packing_the_same_corpora_twice_gives_the_same_bytes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    two_collections(raw)

    first = pack(raw, tmp_path / "first.tar.gz")
    # File times and permissions are not part of a corpora version.
    os.utime(raw / "books" / "b.txt", (0, 1_000_000))
    (raw / "northwind" / "a.md").chmod(0o600)
    second = pack(raw, tmp_path / "second.tar.gz")

    assert first == second
    assert (tmp_path / "first.tar.gz").read_bytes() == (tmp_path / "second.tar.gz").read_bytes()
    assert first == hashlib.sha256((tmp_path / "first.tar.gz").read_bytes()).hexdigest()


def test_the_archive_unpacks_into_the_collections_with_their_manifests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    two_collections(raw)
    (raw / "books" / ".DS_Store").write_bytes(b"not listed")

    pack(raw, tmp_path / "corpora.tar.gz")
    with tarfile.open(tmp_path / "corpora.tar.gz") as tar:
        tar.extractall(tmp_path / "unpacked", filter="data")

    unpacked = tmp_path / "unpacked"
    assert [(entry.key, entry.meta) for entry in corpus_entries(unpacked)] == [
        (entry.key, entry.meta) for entry in corpus_entries(raw)
    ]
    files = [path for path in unpacked.rglob("*") if path.is_file()]
    assert sorted(str(path.relative_to(unpacked)) for path in files) == [
        "books/b.txt",
        "books/c.txt",
        "books/manifest.json",
        "northwind/a.md",
        "northwind/manifest.json",
    ]


def drop_a_field(folder: Path) -> None:
    manifest = (folder / "manifest.json").read_text()
    (folder / "manifest.json").write_text(manifest.replace('"licence": "CC0", ', ""))


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (drop_a_field, "licence"),
        (lambda folder: (folder / "d.md").unlink(), r"d\.md is missing"),
        (lambda folder: (folder / "d.md").write_text("edited"), "does not match its sha256"),
    ],
    ids=["invalid manifest", "missing file", "changed file"],
)
def test_a_damaged_collection_is_not_packed(
    tmp_path: Path, damage: Callable[[Path], None], problem: str
) -> None:
    raw = tmp_path / "raw"
    two_collections(raw)
    damage(write_collection(raw, "faulty", [doc("d.md", sha256=TEXT_SHA256)]))

    with pytest.raises(ManifestError, match=f"(?s)faulty.*{problem}"):
        pack(raw, tmp_path / "corpora.tar.gz")
    assert list(tmp_path.iterdir()) == [raw]


def test_a_document_without_a_sha256_is_not_packed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_collection(raw, "unhashed", [doc("a.md", sha256=TEXT_SHA256), doc("b.md")])

    with pytest.raises(ManifestError, match=r"unhashed.*b\.md has no sha256"):
        pack(raw, tmp_path / "corpora.tar.gz")
