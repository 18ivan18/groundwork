"""Build a corpora version: fetch the third-party collections, take Northwind from the local raw
directory, pack all four into corpora-v<N>.tar.gz and print what to publish.

The SEC, Kubernetes and Gutenberg collections are fetched from upstream into a staging directory,
so the local raw directory is never overwritten. Northwind is fictional and has no upstream; it is
copied from data/raw/northwind/, so edit it there before packing a new version.

The build then compares every packed file with the file at the same path under data/raw/ and says
whether they are byte-identical: if they are, eval numbers measured on the local corpora hold for
the new corpora version without a rerun.

Publishing is a manual step: the script prints the `gh release create` command and does not run it.

Run from backend/ (needs the network and SEC_USER_AGENT):
    uv run python scripts/build_corpora.py 1 [--out DIR]
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from fetch_books import COLLECTION as BOOKS_COLLECTION
from fetch_books import fetch_books
from fetch_engineering_docs import COLLECTION as ENGINEERING_COLLECTION
from fetch_engineering_docs import fetch_engineering_docs
from fetch_filings import fetch_filings, sec_client

from groundwork.config import Settings
from groundwork.ingestion.corpora_bundle import archive_name, pack, release_tag
from groundwork.ingestion.corpus import SEC_COLLECTION

NORTHWIND_COLLECTION = "northwind"
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
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        fetch_engineering_docs(client, staging / ENGINEERING_COLLECTION)
        fetch_books(client, staging / BOOKS_COLLECTION)
    shutil.copytree(raw_dir / NORTHWIND_COLLECTION, staging / NORTHWIND_COLLECTION)


def compare(archive: Path, raw_dir: Path) -> Comparison:
    """Every file in archive against raw_dir, by sha256. Local files count only in the packed
    collections' folders."""
    packed: dict[str, str] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            file = tar.extractfile(member)
            if file is not None:
                packed[member.name] = hashlib.sha256(file.read()).hexdigest()
    collections = {name.split("/")[0] for name in packed}
    local = {
        str(path.relative_to(raw_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for collection in collections
        for path in (raw_dir / collection).iterdir()
        if path.is_file() and not path.name.startswith(".")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", type=int, help="the corpora version number N")
    parser.add_argument(
        "--out", type=Path, help="directory for the tarball (default: a new temporary directory)"
    )
    args = parser.parse_args()

    settings = Settings()
    if not settings.sec_user_agent:
        sys.exit('SEC_USER_AGENT is required by EDGAR. Set it in .env: "Your Name your@email"')
    raw_dir = settings.data_dir / "raw"
    out: Path = args.out or Path(tempfile.mkdtemp(prefix="corpora-"))
    out.mkdir(parents=True, exist_ok=True)
    archive = out / archive_name(args.version)

    with tempfile.TemporaryDirectory(prefix="corpora-staging-") as staging:
        stage(raw_dir, Path(staging), settings.sec_user_agent)
        sha256 = pack(Path(staging), archive)

    print()
    print(f"Tarball: {archive.resolve()}")
    print(f"sha256:  {sha256}")
    report(compare(archive, raw_dir), raw_dir)
    tag = release_tag(args.version)
    print()
    print("To publish (manual):")
    print(f"  gh release create {tag} {archive.resolve()} --repo {REPOSITORY} --title {tag}")


if __name__ == "__main__":
    main()
