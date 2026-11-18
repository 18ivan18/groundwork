"""Sections from the headings a format marks up: Markdown `#` lines and HTML h1-h4."""

from groundwork.ingestion.loaders.base import LoadedDocument
from groundwork.ingestion.structure.base import Section, whole_document


def _is_title(loaded: LoadedDocument) -> bool:
    """A document that opens with its only level-1 heading uses it as its title, which the
    document's own title already carries, so it is left out of the section paths."""
    first = loaded.headings[0]
    return (
        first.level == 1
        and not loaded.text[: first.start].strip()
        and sum(heading.level == 1 for heading in loaded.headings) == 1
    )


def detect_from_headings(loaded: LoadedDocument) -> list[Section]:
    """Each heading starts a section that runs to the next heading, under the path of headings
    above it: a level-3 heading after `# A` and `## B` has the path (A, B, its title). Text before
    the first heading has an empty path, and so does a title heading (see `_is_title`) and the text
    under it up to the next heading. A document with no headings is one section."""
    if not loaded.headings:
        return whole_document(loaded)
    sections = []
    first = loaded.headings[0].start
    if loaded.text[:first].strip():
        sections.append(Section(path=(), start=0, end=first))
    # (level, title) of the headings that enclose the current one.
    stack: list[tuple[int, str]] = []
    ends = [heading.start for heading in loaded.headings[1:]] + [len(loaded.text)]
    title = _is_title(loaded)
    for index, (heading, end) in enumerate(zip(loaded.headings, ends, strict=True)):
        if index == 0 and title:
            sections.append(Section(path=(), start=heading.start, end=end))
            continue
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.title))
        sections.append(
            Section(path=tuple(title for _, title in stack), start=heading.start, end=end)
        )
    return sections
