"""Sections from the headings a format marks up: Markdown `#` lines and HTML h1-h4."""

from groundwork.ingestion.loaders.base import LoadedDocument
from groundwork.ingestion.structure.base import Section, whole_document


def detect_from_headings(loaded: LoadedDocument) -> list[Section]:
    """Each heading starts a section that runs to the next heading, under the path of headings
    above it: a level-3 heading after `# A` and `## B` has the path (A, B, its title). Text before
    the first heading has an empty path. A document with no headings is one section."""
    if not loaded.headings:
        return whole_document(loaded)
    sections = []
    first = loaded.headings[0].start
    if loaded.text[:first].strip():
        sections.append(Section(path=(), start=0, end=first))
    # (level, title) of the headings that enclose the current one.
    stack: list[tuple[int, str]] = []
    ends = [heading.start for heading in loaded.headings[1:]] + [len(loaded.text)]
    for heading, end in zip(loaded.headings, ends, strict=True):
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.title))
        sections.append(
            Section(path=tuple(title for _, title in stack), start=heading.start, end=end)
        )
    return sections
