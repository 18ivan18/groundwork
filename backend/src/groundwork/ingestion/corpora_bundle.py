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
"""

import gzip
import hashlib
import io
import os
import tarfile
from pathlib import Path

from groundwork.ingestion.corpus import MANIFEST, ManifestError, read_manifest

ARCHIVE_SUFFIX = ".tar.gz"
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
