import pytest
from pdf_builder import make_pdf

from groundwork.ingestion.chunking import chunk_document, page_at
from groundwork.ingestion.loaders import NoTextError, PageSpan, load_pdf, loader_for
from groundwork.ingestion.structure import whole_document


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def test_pages_are_read_in_order_with_their_spans() -> None:
    loaded = load_pdf(make_pdf([["ACME sells anvils."], ["Coyotes buy them."]]))

    assert "ACME sells anvils." in loaded.text
    assert loaded.text.index("ACME") < loaded.text.index("Coyotes")
    first, second = loaded.page_spans
    assert (first.page, second.page) == (1, 2)
    assert loaded.text[first.char_start : first.char_end].startswith("ACME")
    assert loaded.text[second.char_start : second.char_end].startswith("Coyotes")


def test_a_blank_page_keeps_the_numbers_of_the_others() -> None:
    loaded = load_pdf(make_pdf([["One."], None, ["Three."]]))

    assert [span.page for span in loaded.page_spans] == [1, 3]


def test_a_pdf_without_a_text_layer_is_refused() -> None:
    with pytest.raises(NoTextError, match="no text layer"):
        load_pdf(make_pdf([None, None]))


def test_a_broken_pdf_is_refused() -> None:
    with pytest.raises(NoTextError, match="could not be read"):
        load_pdf(b"%PDF-1.4\nthis is not a pdf")


def test_pdf_is_a_supported_upload_type() -> None:
    assert loader_for("Report.PDF") is load_pdf


def test_chunks_carry_the_page_they_start_on() -> None:
    pages = [[f"Page {n} sentence {i} about anvils." for i in range(12)] for n in (1, 2, 3)]
    loaded = load_pdf(make_pdf(pages))

    chunks = chunk_document(
        "doc",
        loaded.text,
        whole_document(loaded),
        WordCounter(),
        chunk_tokens=40,
        chunk_overlap=0,
        page_spans=loaded.page_spans,
    )

    assert {chunk.page for chunk in chunks} == {1, 2, 3}
    for chunk in chunks:
        assert chunk.text.startswith(f"Page {chunk.page} ")


def test_page_at_handles_positions_before_between_and_after_pages() -> None:
    spans = (PageSpan(1, 0, 10), PageSpan(2, 12, 20))

    assert page_at(spans, 0) == 1
    assert page_at(spans, 11) == 1
    assert page_at(spans, 12) == 2
    assert page_at(spans, 500) == 2
    assert page_at((), 5) is None
