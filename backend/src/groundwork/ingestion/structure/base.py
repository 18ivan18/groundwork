"""What every structure detector returns."""

from dataclasses import dataclass
from typing import Protocol

from groundwork.ingestion.loaders.base import LoadedDocument


@dataclass(frozen=True)
class Section:
    """A slice `start:end` of the document text, with the headings it sits under, outermost
    first, such as ("Item 7 — Management's Discussion", ) or ("Install", "Linux"). An empty path
    means the text is under no heading."""

    path: tuple[str, ...]
    start: int
    end: int


class StructureDetector(Protocol):
    """Splits a loaded document into sections. Chunks never cross a section boundary, and text
    outside every section is not indexed."""

    def __call__(self, loaded: LoadedDocument) -> list[Section]: ...


def whole_document(loaded: LoadedDocument) -> list[Section]:
    """The plain-text fallback: one section with no heading."""
    return [Section(path=(), start=0, end=len(loaded.text))]
