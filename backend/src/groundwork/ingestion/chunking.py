"""Split sections into overlapping windows that end on sentence or paragraph boundaries."""

import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from groundwork.domain.models import Chunk, make_chunk_id
from groundwork.ingestion.loaders.base import PageSpan
from groundwork.ingestion.structure import Section

# The last window of a long section is often a few words that only make sense with the window
# before it, which the overlap already carries: below MIN_CHUNK_TOKENS it is dropped.
MIN_CHUNK_TOKENS = 30
# A short section that fits in one window is kept whole, down to MIN_SECTION_TOKENS: runbooks and
# policies have sections of a sentence or two. Below that it is a heading with "None." or a page
# footer.
MIN_SECTION_TOKENS = 8

# A period after these does not end a sentence. Without this check a chunk can end on "Mr.",
# cutting the name away from what the passage says about that person.
_ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Messrs", "Dr", "Jr", "Sr", "Inc", "Co", "Corp", "Ltd", "No")
_NOT_ABBREVIATION = "".join(rf"(?<!\b{word}\.)" for word in _ABBREVIATIONS) + r"(?<!\b[A-Z]\.)"
# A blank line, or the whitespace after a sentence. Lines in filings are hard-wrapped, so a
# single newline says nothing about structure.
_BOUNDARY = re.compile(rf"\n[ \t]*\n\s*|(?<=[.!?]){_NOT_ABBREVIATION}\s+(?=[\"'(]?[A-Z])")
# Tried in order on a unit too big for one chunk: table rows first, then words.
_FALLBACK_BOUNDARIES = (re.compile(r"\n"), re.compile(r"\s+"))


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class _Unit:
    """The smallest piece of text a chunk can start or end on."""

    start: int
    end: int
    tokens: int


def _split(text: str, start: int, end: int, boundary: re.Pattern[str]) -> list[tuple[int, int]]:
    spans = []
    position = start
    for match in boundary.finditer(text, start, end):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _fit(
    text: str,
    start: int,
    end: int,
    counter: TokenCounter,
    limit: int,
    fallbacks: tuple[re.Pattern[str], ...],
) -> list[_Unit]:
    """Return the span as one unit, splitting it further if it exceeds the limit."""
    tokens = counter.count(text[start:end])
    if tokens <= limit or not fallbacks:
        return [_Unit(start, end, tokens)]
    units = []
    for s, e in _split(text, start, end, fallbacks[0]):
        units.extend(_fit(text, s, e, counter, limit, fallbacks[1:]))
    return units


def _units(text: str, section: Section, counter: TokenCounter, limit: int) -> list[_Unit]:
    units = []
    for start, end in _split(text, section.start, section.end, _BOUNDARY):
        units.extend(_fit(text, start, end, counter, limit, _FALLBACK_BOUNDARIES))
    return units


def _pack(units: list[_Unit], chunk_tokens: int, overlap: int) -> list[list[_Unit]]:
    """Fill windows greedily, starting each one `overlap` tokens before the last one ended."""
    windows: list[list[_Unit]] = []
    first = 0
    while first < len(units):
        total = 0
        last = first
        # `last == first` admits a unit that alone exceeds the limit, so the loop always advances.
        while last < len(units) and (last == first or total + units[last].tokens <= chunk_tokens):
            total += units[last].tokens
            last += 1
        windows.append(units[first:last])
        if last == len(units):
            break

        next_first = last
        kept = 0
        # `next_first - 1 > first` guarantees the next window starts later than this one.
        while next_first - 1 > first and kept + units[next_first - 1].tokens <= overlap:
            next_first -= 1
            kept += units[next_first].tokens
        first = next_first
    return windows


def page_at(page_spans: Sequence[PageSpan], position: int) -> int | None:
    """The page a character is on, or None for formats without pages. A position in the break
    between two pages counts as the earlier page."""
    index = bisect_right([span.char_start for span in page_spans], position) - 1
    return page_spans[index].page if index >= 0 else None


def chunk_document(
    doc_id: str,
    text: str,
    sections: list[Section],
    counter: TokenCounter,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
    page_spans: Sequence[PageSpan] = (),
) -> list[Chunk]:
    """Chunk each section separately, so that no chunk mixes two sections. A chunk's page is the
    one it starts on."""
    chunks: list[Chunk] = []
    for section in sections:
        units = _units(text, section, counter, chunk_tokens)
        windows = _pack(units, chunk_tokens, chunk_overlap)
        minimum = MIN_SECTION_TOKENS if len(windows) == 1 else MIN_CHUNK_TOKENS
        for window in windows:
            if sum(unit.tokens for unit in window) < minimum:
                continue
            start, end = window[0].start, window[-1].end
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, len(chunks)),
                    doc_id=doc_id,
                    text=text[start:end],
                    section_path=section.path,
                    page=page_at(page_spans, start),
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks
