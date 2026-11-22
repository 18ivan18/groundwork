import hashlib
import io
import json
import os
import tarfile
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from collection_builder import doc, write_collection

from groundwork.domain.models import DocumentMeta
from groundwork.ingestion.corpora_bundle import (
    CorporaError,
    CorporaLock,
    ensure,
    ensure_locked,
    pack,
    read_lock,
)
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


def corpora_lock(archive: Path, sha256: str | None = None, version: int = 1) -> CorporaLock:
    return CorporaLock(
        version=version,
        url=f"https://example.test/{archive.name}",
        sha256=sha256 or hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


class Server:
    """Serves one archive at any URL and records the requests made."""

    def __init__(self, archive: Path) -> None:
        self.body = archive.read_bytes()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, content=self.body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def packed(tmp_path: Path) -> tuple[Path, Path]:
    """Two collections under source/, packed into corpora.tar.gz."""
    source = tmp_path / "source"
    two_collections(source)
    archive = tmp_path / "corpora.tar.gz"
    pack(source, archive)
    return source, archive


def entries(raw: Path) -> list[tuple[str, DocumentMeta]]:
    return [(entry.key, entry.meta) for entry in corpus_entries(raw)]


def test_ensure_installs_the_locked_corpora_version(tmp_path: Path) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"

    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)

    assert entries(raw) == entries(source)


def snapshot(raw: Path) -> dict[str, bytes] | None:
    """Every file under raw by relative path, or None if raw is missing."""
    if not raw.exists():
        return None
    return {str(p.relative_to(raw)): p.read_bytes() for p in raw.rglob("*") if p.is_file()}


def installed(raw: Path, previous: bool) -> dict[str, bytes] | None:
    """A raw directory that is missing, or holds an earlier corpora version than two_collections."""
    if previous:
        write_collection(raw, "previous", [doc("old.md", sha256=TEXT_SHA256)])
    return snapshot(raw)


def leftovers(raw: Path) -> list[str]:
    """Anything beside raw in its parent directory: temporary files ensure failed to remove."""
    return [path.name for path in raw.parent.iterdir() if path != raw]


@pytest.mark.parametrize("previous", [False, True], ids=["missing", "previous version"])
def test_a_download_with_the_wrong_sha256_is_refused(tmp_path: Path, previous: bool) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    before = installed(raw, previous)
    wrong = "0" * 64
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()

    with (
        Server(archive).client() as client,
        pytest.raises(CorporaError, match=f"(?s){wrong}.*{actual}"),
    ):
        ensure(raw, corpora_lock(archive, sha256=wrong), client)

    assert snapshot(raw) == before
    assert leftovers(raw) == []


class BrokenServer(Server):
    """Sends the first half of the archive, then the connection drops."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        def body() -> Iterator[bytes]:
            yield self.body[: len(self.body) // 2]
            raise httpx.ReadError("connection dropped", request=request)

        return httpx.Response(200, content=body())


@pytest.mark.parametrize("previous", [False, True], ids=["missing", "previous version"])
def test_an_interrupted_download_leaves_no_partial_corpus(tmp_path: Path, previous: bool) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    before = installed(raw, previous)

    with BrokenServer(archive).client() as client, pytest.raises(httpx.ReadError):
        ensure(raw, corpora_lock(archive), client)

    assert snapshot(raw) == before
    assert leftovers(raw) == []


def test_corpora_that_match_the_lock_are_not_downloaded_again(tmp_path: Path) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)

    again = Server(archive)
    with again.client() as client:
        ensure(raw, corpora_lock(archive), client)

    assert again.requests == []
    assert entries(raw) == entries(source)


# Version 2 replacing version 1, and the same version number with other bytes.
@pytest.mark.parametrize("previous_version", [1, 2], ids=["older version", "other bytes"])
def test_corpora_installed_from_another_lock_are_replaced(
    tmp_path: Path, previous_version: int
) -> None:
    previous = tmp_path / "previous"
    write_collection(previous, "previous", [doc("old.md", sha256=TEXT_SHA256)])
    pack(previous, tmp_path / "previous.tar.gz")
    raw = tmp_path / "data" / "raw"
    with Server(tmp_path / "previous.tar.gz").client() as client:
        ensure(raw, corpora_lock(tmp_path / "previous.tar.gz", version=previous_version), client)
    source, archive = packed(tmp_path)

    server = Server(archive)
    with server.client() as client:
        ensure(raw, corpora_lock(archive, version=2), client)

    assert len(server.requests) == 1
    assert entries(raw) == entries(source)
    assert leftovers(raw) == []


def archive_with(path: Path, member: tarfile.TarInfo) -> Path:
    """A tarball holding a valid collection and member, which may carry no content."""
    with tarfile.open(path, "w:gz") as tar:
        tar.add(packed(path.parent)[0] / "books", "books")
        member.size = 0 if member.isfile() else member.size
        tar.addfile(member, io.BytesIO(b""))
    return path


def link(name: str, target: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.SYMTYPE
    member.linkname = target
    return member


@pytest.mark.parametrize(
    "member",
    [
        tarfile.TarInfo("../outside.md"),
        tarfile.TarInfo("books/../../outside.md"),
        tarfile.TarInfo("/outside.md"),
        link("books/outside.md", "../../outside.md"),
    ],
    ids=["parent", "nested parent", "absolute", "symlink"],
)
def test_a_member_outside_the_target_is_refused(tmp_path: Path, member: tarfile.TarInfo) -> None:
    archive = archive_with(tmp_path / "evil.tar.gz", member)
    raw = tmp_path / "data" / "raw"

    with Server(archive).client() as client, pytest.raises(CorporaError, match="outside"):
        ensure(raw, corpora_lock(archive), client)

    assert not raw.exists()
    assert leftovers(raw) == []
    assert not list(tmp_path.rglob("outside.md"))


SHA256 = "ab" * 32
LOCK = {"version": 1, "url": "https://example.test/corpora-v1.tar.gz", "sha256": SHA256}


def test_a_lock_file_reads_into_a_corpora_lock(tmp_path: Path) -> None:
    (tmp_path / "corpora.lock.json").write_text(json.dumps(LOCK))

    lock = read_lock(tmp_path / "corpora.lock.json")

    assert (lock.version, lock.url, lock.sha256) == (
        1,
        "https://example.test/corpora-v1.tar.gz",
        SHA256,
    )


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({**LOCK, "sha256": "not-a-hash"}),
        json.dumps({key: value for key, value in LOCK.items() if key != "url"}),
        json.dumps({**LOCK, "collections": ["books"]}),
        json.dumps({**LOCK, "version": 0}),
        "{",
    ],
    ids=["bad sha256", "missing url", "unknown field", "version 0", "not json"],
)
def test_a_malformed_lock_file_is_refused(tmp_path: Path, content: str) -> None:
    (tmp_path / "corpora.lock.json").write_text(content)

    with pytest.raises(CorporaError, match=r"corpora\.lock\.json"):
        read_lock(tmp_path / "corpora.lock.json")


def test_without_a_lock_file_the_corpora_are_used_as_they_are(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    before = installed(raw, previous=True)
    _, archive = packed(tmp_path)
    server = Server(archive)

    with server.client() as client:
        message = ensure_locked(raw, tmp_path / "corpora.lock.json", client)

    assert "No corpora lock" in message
    assert str(tmp_path / "corpora.lock.json") in message
    assert server.requests == []
    assert snapshot(raw) == before


def test_with_a_lock_file_the_locked_version_is_installed_once(tmp_path: Path) -> None:
    source, archive = packed(tmp_path)
    lock_file = tmp_path / "corpora.lock.json"
    lock_file.write_text(corpora_lock(archive, version=3).model_dump_json())
    raw = tmp_path / "data" / "raw"
    server = Server(archive)

    with server.client() as client:
        first = ensure_locked(raw, lock_file, client)
        second = ensure_locked(raw, lock_file, client)

    assert "Installed corpora-v3" in first
    assert "corpora-v3 already" in second
    assert len(server.requests) == 1
    assert entries(raw) == entries(source)
