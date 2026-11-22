"""Corpora versions: every collection under data/raw/ packed together into one tarball, published
as the asset of a GitHub Release.

The bundle format:

- The archive is `corpora-v<N>.tar.gz`, the one asset of the Release tagged `corpora-v<N>`.
- Its top level holds one directory per collection, named like the collection, with the
  collection's manifest.json and the files the manifest lists, and nothing else:
  `<collection>/manifest.json`, `<collection>/<filename>`. Unpacked into data/raw/, it gives the
  folders the corpus reader reads.
- The archive is reproducible: the same files always give the same bytes and sha256. Members are
  regular files only, sorted by path, with mtime 0, owner and group 0 and unnamed, and mode 0644;
  the gzip header has mtime 0 and no file name.

`pack` makes a corpora version. Publishing it is a manual step.

The lock, backend/conf/corpora.lock.json and committed, names the corpora version a commit expects:

    {"version": 1, "url": "https://github.com/.../corpora-v1.tar.gz", "sha256": "<64 hex>"}

`ensure` makes the raw directory hold the locked version: it downloads, verifies and unpacks the
tarball, and writes the lock it installed from into the raw directory as .corpora-version.json, so
the next run can tell without a request that nothing has changed. The ingest and eval scripts call
it through `ensure_locked` before they read the corpora. Without a lock file they use the raw
directory as it is.
"""

import gzip
import hashlib
import io
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from groundwork.ingestion.corpus import MANIFEST, ManifestError, read_manifest

ARCHIVE_SUFFIX = ".tar.gz"
# Relative to backend/. Scripts resolve it from their own location, not the working directory.
LOCK_FILE = Path("conf/corpora.lock.json")
# Written by ensure into the raw directory: the lock its corpora were installed from.
VERSION_MARKER = ".corpora-version.json"
_MODE = 0o644


def release_tag(version: int) -> str:
    """The GitHub Release tag of a corpora version, `corpora-v<N>`."""
    return f"corpora-v{version}"


def archive_name(version: int) -> str:
    """The file name of a corpora version's tarball, the Release's one asset."""
    return release_tag(version) + ARCHIVE_SUFFIX


def _members(raw_dir: Path) -> dict[str, Path]:
    """Archive path -> file, for every collection under raw_dir. Every manifest must be valid and
    every document present and hashed, matching its sha256."""
    members = {}
    for folder in sorted(path.parent for path in raw_dir.glob(f"*/{MANIFEST}")):
        try:
            manifest = read_manifest(folder)
        except ManifestError as error:
            raise ManifestError(f"Collection {folder.name} can't be packed: {error}") from error
        if unhashed := [doc.filename for doc in manifest.documents if not doc.sha256]:
            raise ManifestError(
                f"Collection {folder.name} can't be packed: {', '.join(unhashed)} has no sha256"
            )
        members[f"{folder.name}/{MANIFEST}"] = folder / MANIFEST
        for document in manifest.documents:
            members[f"{folder.name}/{document.filename}"] = folder / document.filename
    return dict(sorted(members.items()))


def pack(raw_dir: Path, archive: Path) -> str:
    """Write every collection under raw_dir into archive, reproducibly, and return its sha256.

    Raises ManifestError, naming the collection, and writes nothing if any manifest is invalid or
    any listed file is missing, unhashed or different from its sha256."""
    members = _members(raw_dir)
    partial = archive.with_name(archive.name + ".partial")
    try:
        with (
            partial.open("wb") as file,
            # No file name and a fixed mtime in the gzip header.
            gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar,
        ):
            for name, path in members.items():
                content = path.read_bytes()
                # TarInfo defaults: mtime 0, uid and gid 0, no owner or group names.
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = _MODE
                tar.addfile(info, io.BytesIO(content))
        os.replace(partial, archive)
    finally:
        partial.unlink(missing_ok=True)
    return hashlib.sha256(archive.read_bytes()).hexdigest()


class CorporaLock(BaseModel):
    """The corpora version a commit expects: its number, where its tarball is and its sha256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Annotated[int, Field(ge=1)]
    url: Annotated[str, StringConstraints(min_length=1)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CorporaError(Exception):
    """A corpora lock is malformed, or the locked corpora version could not be installed."""


def read_lock(path: Path) -> CorporaLock:
    """The corpora lock in path. Raises CorporaError, naming the file, if it is malformed."""
    try:
        return CorporaLock.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise CorporaError(f"{path} is not a valid corpora lock: {error}") from error


def ensure_locked(raw_dir: Path, lock_file: Path, client: httpx.Client | None = None) -> str:
    """Ensure raw_dir holds the corpora version locked in lock_file, and return a line saying what
    was done, for a script to print.

    Without a lock file, raw_dir is used as it is and nothing is checked or downloaded: the case
    until the first corpora version is published and its lock committed. client defaults to a new
    one that follows redirects."""
    if not lock_file.exists():
        return f"No corpora lock at {lock_file}: using {raw_dir} as it is, unchecked."
    lock = read_lock(lock_file)
    tag = release_tag(lock.version)
    if client is None:
        with httpx.Client(timeout=60, follow_redirects=True) as own_client:
            installed = ensure(raw_dir, lock, own_client)
    else:
        installed = ensure(raw_dir, lock, client)
    if installed:
        return f"Installed {tag} from {lock.url} into {raw_dir}."
    return f"Corpora: {tag} already in {raw_dir}."


def ensure(raw_dir: Path, lock: CorporaLock, client: httpx.Client) -> bool:
    """Make raw_dir hold the locked corpora version, and return whether it had to be installed.

    Makes no request if raw_dir already holds it. Otherwise downloads the tarball, checks its
    sha256 and unpacks it beside raw_dir, then swaps it in for whatever raw_dir held. On any
    failure it raises and leaves raw_dir as it was: missing, or the version it held before."""
    if (installed := installed_lock(raw_dir)) and (installed.version, installed.sha256) == (
        lock.version,
        lock.sha256,
    ):
        return False
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    # Next to raw_dir, so the swap is a rename on one file system.
    with tempfile.TemporaryDirectory(dir=raw_dir.parent, prefix=f".{raw_dir.name}-") as temporary:
        work = Path(temporary)
        archive = work / archive_name(lock.version)
        sha256 = _download(client, lock.url, archive)
        if sha256 != lock.sha256:
            raise CorporaError(
                f"{lock.url} does not match the lock: expected sha256 {lock.sha256}, got {sha256}"
            )
        staging = work / "unpacked"
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                _check_member(member, raw_dir)
            # The data filter refuses the same and more, should the check above miss a case.
            tar.extractall(staging, filter="data")
        (staging / VERSION_MARKER).write_text(lock.model_dump_json() + "\n")
        _swap(staging, raw_dir, work / "replaced")
    return True


def installed_lock(raw_dir: Path) -> CorporaLock | None:
    """The lock raw_dir's corpora were installed from by ensure, or None if they weren't."""
    try:
        return CorporaLock.model_validate_json((raw_dir / VERSION_MARKER).read_text())
    except (OSError, ValidationError):
        return None


def _download(client: httpx.Client, url: str, path: Path) -> str:
    """Stream url into path and return the sha256 of its bytes."""
    digest = hashlib.sha256()
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                file.write(chunk)
    return digest.hexdigest()


def _check_member(member: tarfile.TarInfo, raw_dir: Path) -> None:
    """Refuse a member that is not a plain file or folder, or whose path leaves the target."""
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise CorporaError(f"Archive member {member.name} would land outside {raw_dir}")
    if not (member.isfile() or member.isdir()):
        raise CorporaError(
            f"Archive member {member.name} is a link or special file, which could reach outside "
            f"{raw_dir}"
        )


def _swap(new: Path, target: Path, old: Path) -> None:
    """Put new in target's place, moving whatever was there to old, and back if that fails."""
    if target.exists():
        target.rename(old)
    try:
        new.rename(target)
    except BaseException:
        if old.exists():
            old.rename(target)
        raise
