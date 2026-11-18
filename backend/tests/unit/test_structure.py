from itertools import pairwise

from groundwork.ingestion.loaders import LoadedDocument, load_html, load_markdown
from groundwork.ingestion.structure import Section, detector_for, whole_document
from groundwork.ingestion.structure.books import FRONT_MATTER, detect_book_chapters
from groundwork.ingestion.structure.headings import detect_from_headings
from groundwork.ingestion.structure.sec import detect_sec_items


def paths(sections: list[Section]) -> list[tuple[str, ...]]:
    return [section.path for section in sections]


def covers_without_gaps(sections: list[Section], start: int, end: int) -> bool:
    return (
        sections[0].start == start
        and sections[-1].end == end
        and all(a.end == b.start for a, b in pairwise(sections))
    )


# Markdown

MARKDOWN = """Intro before any heading.

# Install

Pick a platform.

## Linux

Use the package.

```bash
# not a heading: a shell comment
```

## macOS ##

Use brew.

# Upgrade

Read the notes.
"""


def test_markdown_headings_nest_by_level() -> None:
    loaded = load_markdown(MARKDOWN.encode())

    sections = detect_from_headings(loaded)

    assert paths(sections) == [
        (),
        ("Install",),
        ("Install", "Linux"),
        ("Install", "macOS"),
        ("Upgrade",),
    ]
    assert covers_without_gaps(sections, 0, len(loaded.text))


def test_a_heading_line_opens_its_section() -> None:
    loaded = load_markdown(MARKDOWN.encode())

    linux = detect_from_headings(loaded)[2]

    assert loaded.text[linux.start :].startswith("## Linux\n")


def test_hashes_inside_a_fenced_block_are_not_headings() -> None:
    loaded = load_markdown(MARKDOWN.encode())

    assert [heading.title for heading in loaded.headings] == [
        "Install",
        "Linux",
        "macOS",
        "Upgrade",
    ]


def test_a_tilde_fence_closes_only_on_a_tilde_fence() -> None:
    loaded = load_markdown(b"~~~\n```\n# code\n~~~\n# Real\n")

    assert [heading.title for heading in loaded.headings] == ["Real"]


def test_markdown_keeps_its_text_verbatim() -> None:
    assert load_markdown(MARKDOWN.encode()).text == MARKDOWN


def test_hash_without_a_space_is_not_a_heading() -> None:
    assert load_markdown(b"#hashtag\n").headings == ()


def test_document_without_headings_is_one_section() -> None:
    loaded = load_markdown(b"Just text.")

    assert detect_from_headings(loaded) == [Section((), 0, 10)]


# HTML

HTML = b"""<html><body>
<p>Preface.</p>
<h1>Guide</h1><p>Overview.</p>
<h2>Setup<br>steps</h2><p>Do this.</p>
<h5>Note</h5><p>A styled label, not a section.</p>
<h2>Teardown</h2><p>Undo that.</p>
</body></html>"""


def test_html_headings_up_to_h4_become_sections() -> None:
    loaded = load_html(HTML)

    sections = detect_from_headings(loaded)

    assert paths(sections) == [(), ("Guide",), ("Guide", "Setup steps"), ("Guide", "Teardown")]
    assert covers_without_gaps(sections, 0, len(loaded.text))


def test_html_heading_offsets_point_at_the_heading_text() -> None:
    loaded = load_html(HTML)

    for heading in loaded.headings:
        assert loaded.text[heading.start :].startswith(heading.title.split()[0])


# Books

BOOK = """The Project Gutenberg eBook of Anvils

*** START OF THE PROJECT GUTENBERG EBOOK ANVILS ***

ANVILS, A NOVEL

 Contents
 Chapter 1
 Chapter 2

Chapter 1

The anvil fell.

CHAPTER II.

It fell again.

*** END OF THE PROJECT GUTENBERG EBOOK ANVILS ***

Licence text that must not be indexed.
"""


def test_book_chapters_are_sections_and_the_licence_is_left_out() -> None:
    loaded = LoadedDocument(BOOK)

    sections = detect_book_chapters(loaded)

    assert paths(sections) == [(FRONT_MATTER,), ("Chapter 1",), ("CHAPTER II",)]
    indexed = "".join(loaded.text[s.start : s.end] for s in sections)
    assert "Licence text" not in indexed
    assert "*** START" not in indexed
    assert "The anvil fell." in indexed


def test_indented_contents_entries_are_not_chapters() -> None:
    sections = detect_book_chapters(LoadedDocument(BOOK))

    assert "It fell again." in BOOK[sections[2].start : sections[2].end]
    assert " Chapter 1\n Chapter 2" in BOOK[sections[0].start : sections[0].end]


def test_a_chapter_heading_closing_an_illustration_counts() -> None:
    text = "[Illustration: TITLE\n\nChapter I.]\n\nIt is a truth.\n\nCHAPTER II.\n\nMore.\n"

    assert paths(detect_book_chapters(LoadedDocument(text))) == [
        (FRONT_MATTER,),
        ("Chapter I",),
        ("CHAPTER II",),
    ]


def test_a_story_collection_has_stories_and_parts() -> None:
    text = (
        "I. A SCANDAL IN BOHEMIA\n\n\nI.\n\nTo Sherlock.\n\nII.\n\nAt three.\n\n"
        "II. THE LEAGUE\n\nA man.\n"
    )

    assert paths(detect_book_chapters(LoadedDocument(text))) == [
        ("I. A SCANDAL IN BOHEMIA",),
        ("I. A SCANDAL IN BOHEMIA", "Part I"),
        ("I. A SCANDAL IN BOHEMIA", "Part II"),
        ("II. THE LEAGUE",),
    ]


def test_a_book_without_headings_is_its_body() -> None:
    text = (
        "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence"
    )

    [section] = detect_book_chapters(LoadedDocument(text))

    assert section.path == ()
    assert text[section.start : section.end].strip() == "Body."


# Registry


def test_document_type_decides_before_format() -> None:
    assert detector_for("10-K", "aapl.htm") is detect_sec_items
    assert detector_for("10-Q", "msft.txt") is detect_sec_items
    assert detector_for("book", "pg84.txt") is detect_book_chapters


def test_format_decides_for_other_types() -> None:
    assert detector_for("runbook", "deploy.md") is detect_from_headings
    assert detector_for("page", "guide.HTML") is detect_from_headings
    assert detector_for("notes", "notes.txt") is whole_document


def test_item_headings_are_not_split_out_of_other_documents() -> None:
    loaded = LoadedDocument("Item 1. Business\n\nNot a filing.\n\nItem 2. Properties\n\nStill not.")

    assert detector_for("notes", "notes.txt")(loaded) == [Section((), 0, len(loaded.text))]
