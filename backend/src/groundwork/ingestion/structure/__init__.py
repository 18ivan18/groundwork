"""Structure detection: split a loaded document into sections with heading paths.

The detector is chosen per document. The document type wins over the file format, because the
type says how the text is organised: an SEC filing is split at its `Item N.` headings whether it
arrived as HTML or text, and the `Item` rule is never run on anything else.
"""

from pathlib import Path

from groundwork.ingestion.structure.base import Section, StructureDetector, whole_document
from groundwork.ingestion.structure.books import detect_book_chapters
from groundwork.ingestion.structure.headings import detect_from_headings
from groundwork.ingestion.structure.sec import detect_sec_items

SEC_DOC_TYPES = frozenset({"10-K", "10-Q"})
BOOK_DOC_TYPES = frozenset({"book"})
_BY_SUFFIX: dict[str, StructureDetector] = {
    ".md": detect_from_headings,
    ".htm": detect_from_headings,
    ".html": detect_from_headings,
}


def detector_for(doc_type: str, filename: str) -> StructureDetector:
    if doc_type in SEC_DOC_TYPES:
        return detect_sec_items
    if doc_type in BOOK_DOC_TYPES:
        return detect_book_chapters
    return _BY_SUFFIX.get(Path(filename).suffix.lower(), whole_document)


__all__ = ["Section", "StructureDetector", "detector_for", "whole_document"]
