"""PDF, through its text layer only. Scanned pages are images, and OCR is out of scope."""

import io
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from groundwork.ingestion.loaders.base import LoadedDocument, NoTextError, PageSpan

# Pages are joined like paragraphs, so a sentence that runs across a page break can still be
# chunked whole, and a chunk's first character tells which page it starts on.
_PAGE_BREAK = "\n\n"


def load_pdf(raw: bytes) -> LoadedDocument:
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as error:
        # pypdf logs and raises on malformed files; the message is for users, the cause for logs.
        logging.getLogger(__name__).info("Unreadable PDF: %s", error)
        raise NoTextError("This PDF could not be read.") from error

    texts: list[str] = []
    spans: list[PageSpan] = []
    position = 0
    for number, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if not text:
            continue
        if texts:
            position += len(_PAGE_BREAK)
        spans.append(PageSpan(page=number, char_start=position, char_end=position + len(text)))
        texts.append(text)
        position += len(text)
    if not texts:
        raise NoTextError(
            "This PDF has no text layer: its pages are probably scanned images, and OCR is not "
            "supported."
        )
    return LoadedDocument(text=_PAGE_BREAK.join(texts), page_spans=tuple(spans))
