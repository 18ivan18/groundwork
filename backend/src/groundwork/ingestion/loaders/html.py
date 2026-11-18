"""HTML, including SEC inline XBRL filings.

Inline XBRL is XHTML with financial tags woven in. The tags themselves are harmless inline
elements, but a filing also carries an `ix:header` block of hidden facts and contexts that must
not reach the passages. Hidden elements go before any text is read.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Comment, NavigableString, Tag

from groundwork.ingestion.loaders.base import CELL_SEPARATOR, Heading, LoadedDocument

# Elements that start a new paragraph. Every other element is inline, and its text joins the
# text around it with no space added: filings split single words across spans ("ITEM 1. B",
# "USINESS"), so inserting spaces between elements would break them.
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "center", "dd", "div", "dl", "dt",
        "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "li", "main", "nav", "ol", "p", "pre", "section", "ul",
    }
)  # fmt: skip
_NEVER_SHOWN_TAGS = ("head", "script", "style", "template", "noscript", "ix:header")
_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# Financial tables put "$" and the closing ")" or "%" in cells of their own. Rejoined to the
# number, so "$ | (1,234 | )" reads "$(1,234)" and columns line up with the header row.
_LEADING_SYMBOL = re.compile(r"^[$€£¥]$")
_TRAILING_SYMBOL = re.compile(r"^[)%]+$")
# A paragraph that is only a page number. Filings print one at every page break.
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
# Lists are often laid out as a bullet in one element and its text in the next, either as two
# paragraphs or as two spans spaced apart by CSS alone.
_BULLET = re.compile(r"^[•●◦▪■]$")
_BULLET_TOUCHING_TEXT = re.compile(r"^([•●◦▪■])(?=\S)", re.MULTILINE)
# Headings that structure detection uses. h5 and h6 are usually styled labels, not sections.
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
_PARAGRAPH_BREAK = "\n\n"


@dataclass(frozen=True)
class _Block:
    text: str
    # The heading level when the block is an h1-h4 element, else None.
    heading: int | None = None


class _TextBuilder:
    """Collects text into paragraphs, collapsing whitespace the way a browser does."""

    def __init__(self) -> None:
        self.blocks: list[_Block] = []
        self.heading: int | None = None
        self._inline: list[str] = []

    def add_inline(self, text: str) -> None:
        self._inline.append(text)

    def end_block(self) -> None:
        # str.split() also splits on non-breaking spaces, which filings use between "Item" and
        # its number. Section detection only sees ordinary spaces as a result.
        lines = (" ".join(line.split()) for line in "".join(self._inline).split("\n"))
        self._inline = []
        block = "\n".join(line for line in lines if line)
        if block:
            self.blocks.append(_Block(block, self.heading))

    def add_block(self, block: str) -> None:
        self.end_block()
        if block:
            self.blocks.append(_Block(block))


def _is_hidden(tag: Tag) -> bool:
    style = tag.get("style")
    return tag.has_attr("hidden") or (isinstance(style, str) and bool(_DISPLAY_NONE.search(style)))


def _remove_hidden(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_NEVER_SHOWN_TAGS):
        tag.decompose()
    # Collected first: removing a tag while iterating would skip its siblings.
    for tag in [tag for tag in soup.find_all(True) if _is_hidden(tag)]:
        if not tag.decomposed:
            tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _walk(node: Tag, builder: _TextBuilder) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            # Comments are removed already; the XML declaration and doctype are subclasses too.
            if type(child) is NavigableString:
                builder.add_inline(str(child))
        elif isinstance(child, Tag):
            if child.name == "table":
                builder.add_block(_render_table(child))
            elif child.name == "br":
                builder.add_inline("\n")
            elif child.name in _HEADING_LEVELS:
                builder.end_block()
                builder.heading = _HEADING_LEVELS[child.name]
                _walk(child, builder)
                builder.end_block()
                builder.heading = None
            elif child.name in _BLOCK_TAGS:
                builder.end_block()
                _walk(child, builder)
                builder.end_block()
            else:
                _walk(child, builder)


def _cell_text(cell: Tag) -> str:
    builder = _TextBuilder()
    _walk(cell, builder)
    builder.end_block()
    return " ".join(" ".join(block.text for block in builder.blocks).split())


def _merge_symbols(cells: list[str]) -> list[str]:
    merged: list[str] = []
    prefix = ""
    for cell in cells:
        if _LEADING_SYMBOL.match(cell):
            prefix += cell
        elif _TRAILING_SYMBOL.match(cell) and merged and not prefix:
            merged[-1] += cell
        else:
            merged.append(prefix + cell)
            prefix = ""
    if prefix:
        merged.append(prefix)
    return merged


def _render_table(table: Tag) -> str:
    """One line per row, cells joined by CELL_SEPARATOR. Empty spacer cells are dropped."""
    rows = []
    for row in table.find_all("tr"):
        # Rows of a nested table belong to it, and are rendered inside their cell.
        if row.find_parent("table") is not table:
            continue
        cells = [_cell_text(cell) for cell in row.find_all(["td", "th"], recursive=False)]
        cells = _merge_symbols([cell for cell in cells if cell])
        if cells:
            rows.append(CELL_SEPARATOR.join(cells))
    return "\n".join(rows)


def _tidy(blocks: list[_Block]) -> list[_Block]:
    """Drop page numbers and join each lone bullet to the paragraph it marks."""
    tidied: list[_Block] = []
    bullet = ""
    for block in blocks:
        if _PAGE_NUMBER.match(block.text):
            continue
        if _BULLET.match(block.text):
            bullet = f"{block.text} "
            continue
        text = bullet + _BULLET_TOUCHING_TEXT.sub(r"\1 ", block.text)
        tidied.append(_Block(text, block.heading))
        bullet = ""
    return tidied


def _parse(raw: bytes) -> tuple[str, tuple[Heading, ...]]:
    with warnings.catch_warnings():
        # Inline XBRL opens with an XML declaration. The HTML parser is still the right one: it
        # tolerates the malformed markup that ordinary uploaded HTML often has.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw, "lxml")
    _remove_hidden(soup)
    builder = _TextBuilder()
    _walk(soup, builder)
    builder.end_block()
    blocks = _tidy(builder.blocks)
    headings = []
    position = 0
    for block in blocks:
        if block.heading is not None:
            # A heading with a <br> spans lines; its title reads as one.
            headings.append(Heading(block.heading, " ".join(block.text.split()), position))
        position += len(block.text) + len(_PARAGRAPH_BREAK)
    return _PARAGRAPH_BREAK.join(block.text for block in blocks), tuple(headings)


def html_to_text(raw: bytes) -> str:
    return _parse(raw)[0]


def load_html(raw: bytes) -> LoadedDocument:
    text, headings = _parse(raw)
    return LoadedDocument(text=text, headings=headings)
