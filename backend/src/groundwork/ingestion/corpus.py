"""Read the committed corpus described by data/raw/manifest.json."""

import json
from datetime import date
from pathlib import Path
from typing import Any

from groundwork.domain.models import DocumentMeta
from groundwork.ingestion.loaders import is_supported

SEC_COLLECTION = "sec-10k"

# 10-K405 was a variant of the 10-K, so it is found under the same document type. The original
# form stays in the metadata.
_DOC_TYPE_BY_FORM = {"10-K405": "10-K"}


def document_meta(entry: dict[str, Any]) -> DocumentMeta:
    doc_type = _DOC_TYPE_BY_FORM.get(entry["form"], entry["form"])
    return DocumentMeta(
        title=f"{entry['company']} · Form {doc_type}",
        collection_id=SEC_COLLECTION,
        doc_type=doc_type,
        doc_date=date.fromisoformat(entry["period_end"]),
        filename=entry["filename"],
        metadata={"company": entry["company"], "ticker": entry["ticker"], "form": entry["form"]},
    )


def load_corpus(raw_dir: Path) -> list[tuple[Path, DocumentMeta]]:
    """List the manifest's files that have a loader. Others wait until their format has one."""
    entries = json.loads((raw_dir / "manifest.json").read_text())
    return [
        (raw_dir / entry["filename"], document_meta(entry))
        for entry in entries
        if is_supported(entry["filename"])
    ]
