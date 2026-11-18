"""Read the committed corpora: one folder per collection under data/raw/, each described by its
manifest.json.

A manifest names the collection's source and licence and lists its documents with the metadata
they are indexed with: title, type, date, access groups and free-form metadata. `planted` lists
the test cases a document was written to contain (a contradiction, an injection), so evals and
readers know it is bait.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from groundwork.domain.models import EVERYONE, DocumentMeta
from groundwork.ingestion.loaders import is_supported

MANIFEST = "manifest.json"
SEC_COLLECTION = "sec-10k"

_Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=40)]
_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# Document keys come from file paths, so they may run longer than collection names.
_Key = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=100)]


class ManifestError(ValueError):
    """A manifest is malformed or does not match the files beside it."""


class ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Names the document in eval questions, unique within the collection.
    key: _Key
    filename: _Text
    title: _Text
    doc_type: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]
    doc_date: date | None = None
    access_groups: tuple[_Slug, ...] = (EVERYONE,)
    owner_id: str | None = None
    # Filterable, as `meta.<key>` in the index.
    metadata: dict[str, str] = Field(default_factory=dict)
    # Where the file came from. Recorded, not indexed.
    source_url: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    planted: tuple[_Text, ...] = ()

    @model_validator(mode="after")
    def _checks(self) -> "ManifestDocument":
        if not self.access_groups:
            raise ValueError("access_groups must name at least one group")
        if "/" in self.filename or "\\" in self.filename or self.filename.startswith("."):
            raise ValueError("filename must be a plain file name in the collection folder")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: _Slug
    description: _Text
    source: _Text
    licence: _Text
    # The upstream commit or version the files were taken from, when there is one.
    revision: str | None = None
    documents: tuple[ManifestDocument, ...]

    @model_validator(mode="after")
    def _unique_files(self) -> "Manifest":
        for field in ("key", "filename"):
            values = [getattr(document, field) for document in self.documents]
            if duplicates := sorted({value for value in values if values.count(value) > 1}):
                raise ValueError(f"{field}s listed twice: {', '.join(duplicates)}")
        return self


def read_manifest(folder: Path, *, check_files: bool = True) -> Manifest:
    """Parse and validate a collection's manifest. With check_files, every listed file must exist
    and match its sha256."""
    path = folder / MANIFEST
    try:
        manifest = Manifest.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ManifestError(f"{path}: {error}") from error
    if manifest.collection != folder.name:
        raise ManifestError(f"{path}: collection {manifest.collection!r} is not the folder name")
    if check_files:
        for document in manifest.documents:
            file = folder / document.filename
            if not file.is_file():
                raise ManifestError(f"{path}: {document.filename} is missing")
            if document.sha256 and hashlib.sha256(file.read_bytes()).hexdigest() != document.sha256:
                raise ManifestError(f"{path}: {document.filename} does not match its sha256")
    return manifest


def write_manifest(manifest: Manifest) -> str:
    """The manifest as committed: two-space JSON, defaults left out, a newline at the end."""
    data = manifest.model_dump(mode="json", exclude_defaults=True)
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def document_meta(collection: str, document: ManifestDocument) -> DocumentMeta:
    return DocumentMeta(
        title=document.title,
        collection_id=collection,
        doc_type=document.doc_type,
        doc_date=document.doc_date,
        filename=document.filename,
        metadata=dict(document.metadata),
        access_groups=document.access_groups,
        owner_id=document.owner_id,
    )


def load_corpus(
    raw_dir: Path, collections: frozenset[str] | None = None
) -> list[tuple[Path, DocumentMeta]]:
    """The documents of every collection under raw_dir, or of the named ones, that have a loader.
    Files in other formats wait until their format has one."""
    folders = sorted(path.parent for path in raw_dir.glob(f"*/{MANIFEST}"))
    if collections is not None:
        if unknown := collections - {folder.name for folder in folders}:
            raise ManifestError(f"No manifest for {', '.join(sorted(unknown))} in {raw_dir}")
        folders = [folder for folder in folders if folder.name in collections]
    corpus = []
    for folder in folders:
        manifest = read_manifest(folder)
        corpus += [
            (folder / document.filename, document_meta(manifest.collection, document))
            for document in manifest.documents
            if is_supported(document.filename)
        ]
    return corpus
