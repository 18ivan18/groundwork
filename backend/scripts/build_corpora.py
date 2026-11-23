"""Build a corpora version: pack the four collections into corpora-v<N>.tar.gz and print what to
publish.

By default the SEC, Kubernetes and Gutenberg collections are fetched from upstream into a staging
directory, so the local raw directory is never overwritten. Northwind is fictional and has no
upstream; it is copied from data/raw/northwind/, so edit it there before packing a new version.

With --from-local nothing is fetched: all four collections are packed from data/raw/ exactly as
they are. corpora-v1 was made this way, from the bytes once committed in data/raw/ that the eval
baseline was measured on. A fetched build differs from them (the SEC fetcher strips a script tag
EDGAR injects into the HTML filings), so a fetched corpora version needs an eval rerun.

The build then compares every packed file with the file at the same path under data/raw/ and says
whether they are byte-identical: if they are, eval numbers measured on the local corpora hold for
the new corpora version without a rerun.

Publishing is a manual step: the script prints the `gh release create` command and does not run it.

Run from backend/:
    uv run python scripts/build_corpora.py 1 --from-local [--out DIR]
    uv run python scripts/build_corpora.py N [--out DIR]   # needs the network and SEC_USER_AGENT
"""

import argparse
import hashlib
import shlex
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fetch_books import COLLECTION as BOOKS_COLLECTION
from fetch_books import fetch_books
from fetch_engineering_docs import COLLECTION as ENGINEERING_COLLECTION
from fetch_engineering_docs import fetch_engineering_docs
from fetch_filings import MISSING_USER_AGENT, fetch_filings, sec_client

from groundwork.config import Settings
from groundwork.ingestion.corpora_bundle import (
    archive_name,
    file_hashes,
    http_client,
    pack,
    release_tag,
)
from groundwork.ingestion.corpus import MANIFEST, NORTHWIND_COLLECTION, SEC_COLLECTION, Manifest

REPOSITORY = "18ivan18/groundwork"


@dataclass(frozen=True)
class Comparison:
    """Packed files against the files at the same paths under a raw directory."""

    identical: list[str]
    different: list[str]
    only_packed: list[str]
    only_local: list[str]

    @property
    def byte_identical(self) -> bool:
        return not (self.different or self.only_packed or self.only_local)


def stage(raw_dir: Path, staging: Path, sec_user_agent: str) -> None:
    """Fetch the third-party collections into staging and copy Northwind beside them."""
    with sec_client(sec_user_agent) as client:
        fetch_filings(client, staging / SEC_COLLECTION)
    with http_client() as client:
        fetch_engineering_docs(client, staging / ENGINEERING_COLLECTION)
        fetch_books(client, staging / BOOKS_COLLECTION)
    shutil.copytree(raw_dir / NORTHWIND_COLLECTION, staging / NORTHWIND_COLLECTION)


def compare(archive: Path, raw_dir: Path) -> Comparison:
    """Every file in archive against raw_dir, by sha256. Local files count only in the packed
    collections' folders, hidden files not at all."""
    packed: dict[str, str] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            if file is not None:
                packed[member.name] = hashlib.sha256(file.read()).hexdigest()
    collections = {name.split("/")[0] for name in packed}
    # A collection missing locally has every packed file only in the archive.
    local = {
        f"{collection}/{name}": sha256
        for collection in collections
        for name, sha256 in file_hashes(raw_dir / collection).items()
    }
    shared = sorted(packed.keys() & local.keys())
    return Comparison(
        identical=[name for name in shared if packed[name] == local[name]],
        different=[name for name in shared if packed[name] != local[name]],
        only_packed=sorted(packed.keys() - local.keys()),
        only_local=sorted(local.keys() - packed.keys()),
    )


def report(comparison: Comparison, raw_dir: Path) -> None:
    if comparison.byte_identical:
        print(f"Byte-identical: all {len(comparison.identical)} packed files match {raw_dir}.")
        return
    print(f"NOT byte-identical with {raw_dir}:")
    print(f"  {len(comparison.identical)} files identical")
    for label, names in (
        ("differ", comparison.different),
        ("only in the archive", comparison.only_packed),
        (f"only in {raw_dir}", comparison.only_local),
    ):
        for name in names:
            print(f"  {label}: {name}")


def release_notes(archive: Path, sha256: str) -> str:
    """The Release notes: each packed collection with its document count and licence, as its
    manifest states it, then the tarball's sha256."""
    manifests = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if PurePosixPath(member.name).name == MANIFEST and (file := tar.extractfile(member)):
                manifests.append(Manifest.model_validate_json(file.read()))
    lines = [f"{len(manifests)} collections for Groundwork's demo and eval.", ""]
    for manifest in sorted(manifests, key=lambda manifest: manifest.collection):
        count = len(manifest.documents)
        documents = "document" if count == 1 else "documents"
        lines.append(f"- {manifest.collection} ({count} {documents}): {manifest.licence}")
    lines += ["", f"sha256: {sha256}"]
    return "\n".join(lines)


def publish_command(archive: Path, version: int, sha256: str) -> str:
    """The `gh release create` command that publishes archive as corpora version `version`."""
    return shlex.join(
        [
            "gh",
            "release",
            "create",
            release_tag(version),
            str(archive.resolve()),
            "--repo",
            REPOSITORY,
            "--title",
            f"Corpora v{version}",
            "--notes",
            release_notes(archive, sha256),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", type=int, help="the corpora version number N")
    parser.add_argument(
        "--out", type=Path, help="directory for the tarball (default: a new temporary directory)"
    )
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="fetch nothing; pack all four collections from data/raw/ exactly as they are",
    )
    args = parser.parse_args()

    settings = Settings()
    if not args.from_local and not settings.sec_user_agent:
        sys.exit(MISSING_USER_AGENT)
    raw_dir = settings.data_dir / "raw"
    out: Path = args.out or Path(tempfile.mkdtemp(prefix="corpora-"))
    out.mkdir(parents=True, exist_ok=True)
    archive = out / archive_name(args.version)

    if args.from_local:
        sha256 = pack(raw_dir, archive)
    else:
        with tempfile.TemporaryDirectory(prefix="corpora-staging-") as staging:
            stage(raw_dir, Path(staging), settings.sec_user_agent)
            sha256 = pack(Path(staging), archive)

    print()
    print(f"Tarball: {archive.resolve()}")
    print(f"sha256:  {sha256}")
    report(compare(archive, raw_dir), raw_dir)
    print()
    print("To publish (manual):")
    print(publish_command(archive, args.version, sha256))


if __name__ == "__main__":
    main()
