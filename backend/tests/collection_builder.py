"""Write small collections into a temporary directory, for tests of the corpus reader and the
corpora bundle."""

import json
from pathlib import Path
from typing import Any


def write_collection(raw: Path, name: str, documents: list[dict[str, Any]], **fields: Any) -> Path:
    """A collection folder under raw: each document's file holds "text", beside a manifest."""
    folder = raw / name
    folder.mkdir(parents=True)
    for document in documents:
        (folder / document["filename"]).write_text("text")
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
