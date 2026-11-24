import hashlib
import io
import json
import os
import shutil
import tarfile
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from collection_builder import doc, write_collection

from groundwork.config import Settings
from groundwork.domain.models import DocumentMeta
from groundwork.ingestion.corpora_bundle import (
    CorporaError,
    CorporaLock,
    corpora_version,
    ensure,
    ensure_corpora,
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


@pytest.mark.parametrize("exists", [False, True], ids=["missing", "empty"])
def test_ensure_installs_the_locked_corpora_version(tmp_path: Path, exists: bool) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    if exists:
        raw.mkdir(parents=True)

    with Server(archive).client() as client:
        outcome = ensure(raw, corpora_lock(archive), client)

    assert outcome == "installed"
    assert entries(raw) == entries(source)


class RedirectingServer(Server):
    """Redirects the Release URL to where the asset is, as GitHub does."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.test":
            self.requests.append(request)
            return httpx.Response(302, headers={"Location": "https://assets.example.test/a"})
        return super().__call__(request)


def test_ensure_follows_the_release_redirect_whatever_the_client(tmp_path: Path) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"

    # A client that does not follow redirects by default.
    with RedirectingServer(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)

    assert entries(raw) == entries(source)


def test_temporary_folders_left_by_a_killed_run_are_removed(tmp_path: Path) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)
    stale = raw.parent / ".raw-corpora-k1lled"
    (stale / "unpacked").mkdir(parents=True)
    (stale / "corpora-v1.tar.gz").write_bytes(b"half")
    kept = raw.parent / ".raw-backup"
    kept.mkdir()

    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)

    assert not stale.exists()
    assert kept.exists()


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


def test_a_missing_lock_file_is_an_error(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    before = installed(raw, previous=True)
    _, archive = packed(tmp_path)
    server = Server(archive)
    lock_file = tmp_path / "conf" / "corpora.lock.json"

    with (
        server.client() as client,
        pytest.raises(CorporaError, match=f"(?s){lock_file}.*GROUNDWORK_CORPORA=local"),
    ):
        ensure_locked(raw, lock_file, client)

    assert server.requests == []
    assert snapshot(raw) == before


@pytest.mark.parametrize("locked", [False, True], ids=["no lock", "lock"])
def test_local_corpora_are_used_as_they_are_and_said_to_be_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locked: bool
) -> None:
    monkeypatch.setenv("GROUNDWORK_CORPORA", "local")
    raw = tmp_path / "data" / "raw"
    before = installed(raw, previous=True)
    _, archive = packed(tmp_path)
    lock_file = tmp_path / "corpora.lock.json"
    if locked:
        lock_file.write_text(corpora_lock(archive).model_dump_json())
    server = Server(archive)

    with server.client() as client:
        message = ensure_locked(raw, lock_file, client)

    assert "LOCAL" in message
    assert "unchecked" in message
    assert "corpora version" in message
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


def test_scripts_ensure_the_corpora_locked_in_the_working_directorys_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # backend/ in a checkout, /app in the Docker image: both hold conf/corpora.lock.json.
    source, archive = packed(tmp_path)
    workdir = tmp_path / "app"
    (workdir / "conf").mkdir(parents=True)
    (workdir / "conf" / "corpora.lock.json").write_text(corpora_lock(archive).model_dump_json())
    monkeypatch.chdir(workdir)
    settings = Settings(_env_file=None, data_dir=tmp_path / "volume")

    with Server(archive).client() as client:
        raw = ensure_corpora(settings, client)

    assert raw == tmp_path / "volume" / "raw"
    assert entries(raw) == entries(source)
    assert "Installed corpora-v1" in capsys.readouterr().out


def test_the_corpora_version_of_a_run_is_the_installed_one(tmp_path: Path) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive, version=3), client)

    assert corpora_version(raw) == "corpora-v3"


def test_a_run_on_local_corpora_has_no_version_even_when_one_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)
    monkeypatch.setenv("GROUNDWORK_CORPORA", "local")

    assert corpora_version(raw) is None


def test_corpora_no_lock_installed_are_not_given_a_version(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    two_collections(raw)

    with pytest.raises(CorporaError, match=r"no \.corpora-version\.json"):
        corpora_version(raw)


# Local changes: Northwind's only source is the maintainer's data/raw, so ensure never loses them.


def edit(raw: Path) -> None:
    (raw / "northwind" / "a.md").write_text("edited locally")


def add(raw: Path) -> None:
    (raw / "books" / "new.txt").write_text("added locally")


def remove(raw: Path) -> None:
    (raw / "books" / "c.txt").unlink()


@pytest.mark.parametrize(
    ("change", "path"),
    [(edit, "northwind/a.md"), (add, "books/new.txt"), (remove, "books/c.txt")],
    ids=["edited", "added", "removed"],
)
@pytest.mark.parametrize("next_version", [1, 2], ids=["same lock", "new lock"])
def test_corpora_with_local_changes_are_never_replaced(
    tmp_path: Path, change: Callable[[Path], None], path: str, next_version: int
) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)
    change(raw)
    before = snapshot(raw)
    server = Server(archive)

    with (
        server.client() as client,
        pytest.raises(CorporaError, match=f"(?s)local changes.*{path}.*new corpora version"),
    ):
        ensure(raw, corpora_lock(archive, version=next_version), client)

    assert snapshot(raw) == before
    assert server.requests == []
    assert leftovers(raw) == []


def test_hidden_files_are_not_local_changes(tmp_path: Path) -> None:
    _, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    with Server(archive).client() as client:
        ensure(raw, corpora_lock(archive), client)
    (raw / "books" / ".DS_Store").write_bytes(b"finder")

    again = Server(archive)
    with again.client() as client:
        ensure(raw, corpora_lock(archive), client)

    assert again.requests == []


# A raw directory from before markers held file hashes: every checkout and Docker volume then.


def test_unmarked_corpora_identical_to_the_locked_version_are_adopted(tmp_path: Path) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    shutil.copytree(source, raw)
    (raw / ".keep").write_text("left in place")
    before = snapshot(raw)

    with Server(archive).client() as client:
        outcome = ensure(raw, corpora_lock(archive), client)

    assert outcome == "adopted"
    assert (raw / ".keep").exists()
    assert {
        k: v for k, v in (snapshot(raw) or {}).items() if k != ".corpora-version.json"
    } == before
    assert leftovers(raw) == []
    again = Server(archive)
    with again.client() as client:
        assert ensure(raw, corpora_lock(archive), client) == "current"
    assert again.requests == []


@pytest.mark.parametrize(
    ("change", "line"),
    [
        (edit, "differs from corpora-v1: northwind/a.md"),
        (add, "only in the raw directory: books/new.txt"),
        (remove, "only in corpora-v1: books/c.txt"),
    ],
    ids=["edited", "added", "removed"],
)
def test_unmarked_corpora_that_differ_from_the_locked_version_are_left_alone(
    tmp_path: Path, change: Callable[[Path], None], line: str
) -> None:
    source, archive = packed(tmp_path)
    raw = tmp_path / "data" / "raw"
    shutil.copytree(source, raw)
    change(raw)
    before = snapshot(raw)

    with (
        Server(archive).client() as client,
        pytest.raises(CorporaError, match=f"(?s){line}.*new corpora version"),
    ):
        ensure(raw, corpora_lock(archive), client)

    assert snapshot(raw) == before
    assert leftovers(raw) == []
