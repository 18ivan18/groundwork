"""Chapters of Project Gutenberg books.

Only the text between Gutenberg's `*** START OF` and `*** END OF` lines is placed in sections, so
the licence and the catalogue header stay in the file but are never indexed.
"""

import re

from groundwork.ingestion.loaders.base import LoadedDocument
from groundwork.ingestion.structure.base import Section

FRONT_MATTER = "Front matter"

_START = re.compile(r"^\*\*\* ?START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
_END = re.compile(r"^\*\*\* ?END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE)
_NUMERAL = r"(?:[IVXLC]+|\d+)"
# Flush left, on its own line: tables of contents indent their entries. "Chapter I.]" closes an
# illustration caption in some editions, which is where their first chapter heading sits.
_CHAPTER = re.compile(rf"^(?:CHAPTER|Chapter|LETTER|Letter) {_NUMERAL}\.?\]?[ \t]*$", re.MULTILINE)
# Story collections: "I. A SCANDAL IN BOHEMIA", in capitals, then parts numbered "I." alone.
_STORY = re.compile(rf"^{_NUMERAL}\. [A-Z][A-Z\u2019'\-, ]+$", re.MULTILINE)
_PART = re.compile(rf"^{_NUMERAL}\.$", re.MULTILINE)


def _body(text: str) -> tuple[int, int]:
    start, end = 0, len(text)
    if match := _START.search(text):
        start = match.end()
    if match := _END.search(text, start):
        end = match.start()
    return start, end


def _title(line: str) -> str:
    return line.strip().rstrip("].").strip()


def detect_book_chapters(loaded: LoadedDocument) -> list[Section]:
    """Chapters (or letters) as sections or, in a story collection, stories and their parts.

    The heading line opens its section. Text before the first heading is front matter, such as
    the title page and contents. A book with no recognised headings is one untitled section.
    """
    text = loaded.text
    body_start, body_end = _body(text)
    # (start, path) of each heading inside the body.
    marks: list[tuple[int, tuple[str, ...]]] = [
        (match.start(), (_title(match[0]),))
        for match in _CHAPTER.finditer(text, body_start, body_end)
    ]
    # Only a book without chapters is read as a story collection, so a stray capitalised line
    # in a novel can't switch it to stories.
    if not marks:
        stories = list(_STORY.finditer(text, body_start, body_end))
        marks = [(match.start(), (_title(match[0]),)) for match in stories]
        for match in _PART.finditer(text, body_start, body_end):
            enclosing = [story for story in stories if story.start() < match.start()]
            if enclosing:
                path: tuple[str, ...] = (_title(enclosing[-1][0]), f"Part {_title(match[0])}")
                marks.append((match.start(), path))
    if not marks:
        return [Section(path=(), start=body_start, end=body_end)]
    marks.sort()
    sections = []
    if text[body_start : marks[0][0]].strip():
        sections.append(Section(path=(FRONT_MATTER,), start=body_start, end=marks[0][0]))
    ends = [start for start, _ in marks[1:]] + [body_end]
    for (start, path), end in zip(marks, ends, strict=True):
        sections.append(Section(path=path, start=start, end=end))
    return sections
