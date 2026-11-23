"""Write small collections into a temporary directory, for tests of the corpus reader and the
corpora bundle."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_collection(
    raw: Path,
    name: str,
    documents: list[dict[str, Any]],
    *,
    texts: Mapping[str, str] | None = None,
    **fields: Any,
) -> Path:
    """A collection folder under raw: each document's file beside a manifest. A file holds its
    entry in `texts`, or "text"."""
    texts = texts or {}
    folder = raw / name
    folder.mkdir(parents=True)
    for document in documents:
        (folder / document["filename"]).write_text(texts.get(document["filename"], "text"))
    manifest = {
        "collection": name,
        "description": "Test documents.",
        "source": "Tests",
        "licence": "CC0",
        "documents": documents,
        **fields,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def doc(filename: str = "a.md", **fields: Any) -> dict[str, Any]:
    """A manifest entry, keyed by its file name."""
    key = filename.split(".")[0].lower() + "-" + filename.split(".")[-1].lower()
    return {"key": key, "filename": filename, "title": "A", "doc_type": "runbook", **fields}
