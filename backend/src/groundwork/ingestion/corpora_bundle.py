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
tarball, and writes a marker, .corpora-version.json, into the raw directory: the lock it installed
from and the sha256 of every file it installed. The next run makes no request if the marker names
the lock and the files are unchanged.

Local data is never lost: Northwind's only source is the maintainer's data/raw/ (D27). `ensure`
replaces a raw directory only when it is missing, empty or exactly what its marker records, which
its Release restores. If the files differ from the marker, or there is no marker and they differ
from the locked version, it raises CorporaError listing the differences and touches nothing. A raw
directory without a marker that matches the locked version byte for byte is adopted: the marker is
written beside it.

The ingest and eval scripts call `ensure_corpora` before they read the corpora. A missing lock file
is an error. GROUNDWORK_CORPORA=local in the environment skips all of it and uses the raw directory
as it is, unchecked, for a maintainer iterating on Northwind; results from such a run must not be
labelled with a corpora version. `corpora_version` says which label a run's results get.
"""

import gzip
import hashlib
import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from groundwork.config import Settings
from groundwork.ingestion.corpus import MANIFEST, ManifestError, read_manifest

ARCHIVE_SUFFIX = ".tar.gz"
# Relative to the working directory, like USERS_FILE: backend/ in a checkout, /app in the image.
LOCK_FILE = Path("conf/corpora.lock.json")
# Written by ensure into the raw directory: the lock its corpora were installed from and the
# sha256 of every file it installed.
VERSION_MARKER = ".corpora-version.json"
# Set to "local" to skip ensure and use the raw directory as it is, unchecked.
LOCAL_CORPORA_ENV = "GROUNDWORK_CORPORA"
_MODE = 0o644


def http_client() -> httpx.Client:
    """The HTTP client for downloading corpora and their upstream files: follows redirects."""
    return httpx.Client(timeout=60, follow_redirects=True)


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


def ensure_corpora(settings: Settings, client: httpx.Client | None = None) -> Path:
    """For a script about to read the corpora: ensure settings' raw directory holds the version
    locked in conf/corpora.lock.json under the working directory, print what was done, and return
    the raw directory."""
    raw_dir = settings.data_dir / "raw"
    print(ensure_locked(raw_dir, LOCK_FILE, client))
    return raw_dir


def ensure_locked(raw_dir: Path, lock_file: Path, client: httpx.Client | None = None) -> str:
    """Ensure raw_dir holds the corpora version locked in lock_file, and return a line saying what
    was done, for a script to print.

    With GROUNDWORK_CORPORA=local in the environment nothing is checked or downloaded: raw_dir is
    used as it is, for a maintainer iterating on Northwind. Otherwise a missing lock file is a
    CorporaError. client defaults to a new one from http_client."""
    if _corpora_are_local():
        return (
            f"!!! CORPORA ARE LOCAL ({LOCAL_CORPORA_ENV}=local): using {raw_dir} as it is, "
            "unchecked. Results from this run must not be labelled with a corpora version."
        )
    if not lock_file.exists():
        raise CorporaError(
            f"No corpora lock at {lock_file}. Run from backend/ (or /app in the Docker image), or "
            f"set {LOCAL_CORPORA_ENV}=local to use {raw_dir} as it is, unchecked."
        )
    lock = read_lock(lock_file)
    tag = release_tag(lock.version)
    if client is None:
        with http_client() as own_client:
            outcome = ensure(raw_dir, lock, own_client)
    else:
        outcome = ensure(raw_dir, lock, client)
    if outcome == "installed":
        return f"Installed {tag} from {lock.url} into {raw_dir}."
    if outcome == "adopted":
        return f"Corpora: {raw_dir} already held {tag} byte for byte; recorded it as installed."
    return f"Corpora: {tag} already in {raw_dir}."


def _corpora_are_local() -> bool:
    return os.environ.get(LOCAL_CORPORA_ENV) == "local"


def corpora_version(raw_dir: Path) -> str | None:
    """The corpora version that results measured on raw_dir belong to, such as "corpora-v1", from
    the lock it was installed from. None when the corpora are local (GROUNDWORK_CORPORA=local) and
    so unchecked, whatever was once installed. Call it after ensure_corpora, which checks that
    raw_dir still holds what its marker records."""
    if _corpora_are_local():
        return None
    installed = installed_corpora(raw_dir)
    if installed is None:
        raise CorporaError(f"{raw_dir} has no {VERSION_MARKER} to name its corpora version")
    return release_tag(installed.lock.version)


class InstalledCorpora(BaseModel):
    """The marker ensure writes into a raw directory: the lock it installed from and the sha256 of
    every file it installed, by path relative to the raw directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lock: CorporaLock
    files: dict[str, str]


def ensure(
    raw_dir: Path, lock: CorporaLock, client: httpx.Client
) -> Literal["current", "installed", "adopted"]:
    """Make raw_dir hold the locked corpora version, and say how: it already did ("current"), it
    was downloaded and installed ("installed"), or raw_dir already held its files but had no
    marker, which is now written ("adopted").

    raw_dir already holds the locked version when its marker names the lock, every file the marker
    records is unchanged and no other file is beside them; then no request is made. Hidden files
    don't count.

    Local data is never lost: raw_dir is replaced only when it is missing or empty, or holds
    exactly what its marker records, which its Release can restore. Otherwise ensure raises
    CorporaError, listing the differences, and leaves raw_dir as it is: local changes to installed
    corpora, whatever the lock, and a raw_dir without a marker whose files differ from the locked
    version's.

    To install, it downloads the tarball, checks its sha256 and unpacks it beside raw_dir, then
    swaps it in. On any failure it raises and leaves raw_dir as it was. It first removes the
    temporary folders a killed run left beside raw_dir."""
    _remove_stale_temporaries(raw_dir)
    installed = installed_corpora(raw_dir)
    if installed is not None:
        _refuse_local_changes(raw_dir, installed)
        if (installed.lock.version, installed.lock.sha256) == (lock.version, lock.sha256):
            return "current"
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    # Next to raw_dir, so the swap is a rename on one file system.
    with tempfile.TemporaryDirectory(dir=raw_dir.parent, prefix=_temporary_prefix(raw_dir)) as tmp:
        work = Path(tmp)
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
        marker = InstalledCorpora(lock=lock, files=file_hashes(staging))
        if installed is None and (local := file_hashes(raw_dir)):
            if differences := _differences(marker.files, local, release_tag(lock.version)):
                raise CorporaError(
                    f"{raw_dir} has no {VERSION_MARKER} and differs from "
                    f"{release_tag(lock.version)}, so it is left as it is:\n  "
                    + "\n  ".join(differences)
                    + _how_to_proceed(raw_dir)
                )
            _write_marker(raw_dir, marker)
            return "adopted"
        _write_marker(staging, marker)
        if installed is not None:
            # Edits made during the download count too.
            _refuse_local_changes(raw_dir, installed)
        _swap(staging, raw_dir, work / "replaced")
    return "installed"


def installed_corpora(raw_dir: Path) -> InstalledCorpora | None:
    """The marker ensure wrote into raw_dir, or None if it has none it can read."""
    try:
        return InstalledCorpora.model_validate_json((raw_dir / VERSION_MARKER).read_text())
    except (OSError, ValidationError):
        return None


def _write_marker(folder: Path, marker: InstalledCorpora) -> None:
    (folder / VERSION_MARKER).write_text(marker.model_dump_json(indent=2) + "\n")


def file_hashes(folder: Path) -> dict[str, str]:
    """The sha256 of every file under folder, by relative POSIX path. Hidden files and folders,
    such as the marker or .DS_Store, are left out."""
    hashes = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _differences(expected: dict[str, str], local: dict[str, str], source: str) -> list[str]:
    """One line per path whose local file differs from the one expected from source."""
    lines = {path: f"only in {source}" for path in expected.keys() - local.keys()}
    lines |= {path: "only in the raw directory" for path in local.keys() - expected.keys()}
    lines |= {
        path: f"differs from {source}"
        for path in expected.keys() & local.keys()
        if expected[path] != local[path]
    }
    return [f"{lines[path]}: {path}" for path in sorted(lines)]


def _refuse_local_changes(raw_dir: Path, installed: InstalledCorpora) -> None:
    """Raise CorporaError, listing them, if raw_dir's files differ from those ensure installed."""
    tag = release_tag(installed.lock.version)
    if differences := _differences(installed.files, file_hashes(raw_dir), tag):
        raise CorporaError(
            f"{raw_dir} has local changes against {tag}, which it was installed from, so it is "
            "left as it is:\n  " + "\n  ".join(differences) + _how_to_proceed(raw_dir)
        )


def _how_to_proceed(raw_dir: Path) -> str:
    return (
        f"\nMove the edits out of {raw_dir} and run again, or pack and publish them as a new "
        "corpora version (scripts/build_corpora.py) and lock it. To use them unchecked meanwhile, "
        f"set {LOCAL_CORPORA_ENV}=local."
    )


def _temporary_prefix(raw_dir: Path) -> str:
    return f".{raw_dir.name}-corpora-"


def _remove_stale_temporaries(raw_dir: Path) -> None:
    """Remove the temporary folders a killed run of ensure left beside raw_dir."""
    for stale in raw_dir.parent.glob(_temporary_prefix(raw_dir) + "*"):
        shutil.rmtree(stale, ignore_errors=True)


def _download(client: httpx.Client, url: str, path: Path) -> str:
    """Stream url into path and return the sha256 of its bytes."""
    digest = hashlib.sha256()
    # Per request, not left to the client: a Release asset URL always redirects, and a caller's
    # client may not follow redirects.
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
