import hashlib
import shlex
import shutil
from pathlib import Path

from build_corpora import compare, publish_command
from collection_builder import doc, write_collection

from groundwork.ingestion.corpora_bundle import pack

TEXT_SHA256 = hashlib.sha256(b"text").hexdigest()


def packed(tmp_path: Path) -> tuple[Path, Path]:
    """Two collections under source/, packed into corpora.tar.gz."""
    source = tmp_path / "source"
    write_collection(source, "northwind", [doc("a.md", sha256=TEXT_SHA256)])
    write_collection(source, "books", [doc("b.txt", sha256=TEXT_SHA256)])
    archive = tmp_path / "corpora.tar.gz"
    pack(source, archive)
    return source, archive


def test_a_packed_collection_missing_locally_is_only_in_the_archive(tmp_path: Path) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "raw"
    shutil.copytree(source / "books", raw / "books")

    comparison = compare(archive, raw)

    assert comparison.identical == ["books/b.txt", "books/manifest.json"]
    assert comparison.only_packed == ["northwind/a.md", "northwind/manifest.json"]
    assert comparison.different == comparison.only_local == []
    assert not comparison.byte_identical


def test_the_publish_command_names_the_collections_their_licences_and_the_sha256(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write_collection(source, "northwind", [doc("a.md", sha256=TEXT_SHA256)], licence="Original")
    write_collection(
        source,
        "books",
        [doc("b.txt", sha256=TEXT_SHA256), doc("c.txt", sha256=TEXT_SHA256)],
        licence="Public domain in the US",
    )
    archive = tmp_path / "corpora-v2.tar.gz"
    sha256 = pack(source, archive)

    command = shlex.split(publish_command(archive, 2, sha256))

    assert command[:4] == ["gh", "release", "create", "corpora-v2"]
    assert command[4] == str(archive.resolve())
    assert command[command.index("--repo") + 1] == "18ivan18/groundwork"
    assert command[command.index("--title") + 1] == "Corpora v2"
    notes = command[command.index("--notes") + 1].splitlines()
    assert "- books (2 documents): Public domain in the US" in notes
    assert "- northwind (1 document): Original" in notes
    assert f"sha256: {sha256}" in notes
