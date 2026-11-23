import hashlib
from dataclasses import replace
from datetime import date
from typing import Any

import pytest
from fetch_filings import Filing, find_latest_filing, manifest_entry, strip_edgar_script


def submissions(forms: list[tuple[str, str]]) -> dict[str, Any]:
    """Build a synthetic EDGAR response from (form, filing date) pairs."""
    return {
        "name": "ACME Corp",
        "tickers": ["ACME"],
        "filings": {
            "recent": {
                "form": [form for form, _ in forms],
                "filingDate": [date for _, date in forms],
                "reportDate": [f"period-{date}" for _, date in forms],
                "accessionNumber": [f"0000000001-{i}" for i in range(len(forms))],
                "primaryDocument": [f"acme-{i}.htm" for i in range(len(forms))],
            }
        },
    }


def test_picks_newest_10k_regardless_of_order() -> None:
    data = submissions([("10-K", "2023-01-01"), ("10-K", "2025-01-01"), ("10-K", "2024-01-01")])

    assert find_latest_filing(data, "ACME", "1").filing_date == "2025-01-01"


def test_ignores_other_forms_and_amendments() -> None:
    data = submissions(
        [
            ("10-K/A", "2025-06-01"),
            ("10-Q", "2025-05-01"),
            ("8-K", "2025-04-01"),
            ("10-K", "2025-01-01"),
        ]
    )

    filing = find_latest_filing(data, "ACME", "1")

    assert filing.form == "10-K"
    assert filing.filing_date == "2025-01-01"


def test_plain_text_search_reaches_older_blocks_and_skips_html() -> None:
    data = submissions([("10-K", "2025-01-01")])
    older = {
        "form": ["10-K405", "10-K", "10-K"],
        "filingDate": ["2001-04-27", "2000-01-01", "2002-01-01"],
        "reportDate": ["2001-01-28", "1999-12-31", "2001-12-31"],
        "accessionNumber": ["1", "2", "3"],
        "primaryDocument": ["new.txt", "old.txt", "html-era.htm"],
    }

    filing = find_latest_filing(
        data,
        "ACME",
        "1",
        forms=("10-K", "10-K405"),
        suffix=".txt",
        older_blocks=[older],
    )

    assert filing.primary_document == "new.txt"
    assert filing.form == "10-K405"


def test_filings_without_a_primary_document_are_skipped() -> None:
    data = submissions([("10-K", "2025-01-01")])
    data["filings"]["recent"]["primaryDocument"] = [""]

    with pytest.raises(ValueError, match="No 10-K"):
        find_latest_filing(data, "ACME", "1")


def test_fails_when_there_is_no_10k() -> None:
    with pytest.raises(ValueError, match="No 10-K"):
        find_latest_filing(submissions([("8-K", "2025-01-01")]), "ACME", "1")


def test_fails_when_cik_belongs_to_another_ticker() -> None:
    with pytest.raises(ValueError, match="not OTHER"):
        find_latest_filing(submissions([("10-K", "2025-01-01")]), "OTHER", "1")


def make_filing() -> Filing:
    return Filing(
        company="ACME Corp",
        ticker="ACME",
        cik="0000000123",
        form="10-K",
        period_end="2025-09-27",
        filing_date="2025-10-31",
        accession_number="0000000123-25-000079",
        primary_document="acme-20250927.htm",
    )


def test_source_url_strips_cik_padding_and_accession_dashes() -> None:
    assert make_filing().source_url == (
        "https://www.sec.gov/Archives/edgar/data/123/000000012325000079/acme-20250927.htm"
    )


def test_manifest_entry_has_required_metadata() -> None:
    entry = manifest_entry(make_filing(), b"content")

    assert entry.title == "ACME Corp · Form 10-K"
    assert entry.metadata == {"company": "ACME Corp", "ticker": "ACME", "form": "10-K"}
    assert entry.doc_type == "10-K"
    assert entry.doc_date == date(2025, 9, 27)
    assert entry.provenance["filing_date"] == "2025-10-31"
    assert entry.filename == "ACME-10-K-2025-09-27.htm"
    assert entry.source_url is not None
    assert entry.source_url.endswith("/acme-20250927.htm")


def test_a_10k405_is_filed_under_10k_and_keeps_its_form() -> None:
    entry = manifest_entry(replace(make_filing(), form="10-K405"), b"content")

    assert (entry.doc_type, entry.metadata["form"]) == ("10-K", "10-K405")
    assert entry.sha256 == hashlib.sha256(b"content").hexdigest()


# The tail of a 10-K as EDGAR serves it: the filing ends at its last page footer and EDGAR adds a
# script tag, whose path rotates over time, just before </body>.
FILING_TAIL = (
    b'<div style="text-align:center"><span>Apple Inc. | 2026 Form 10-K | 63</span></div>'
    b"</div></div>"
)
EDGAR_SCRIPT = (
    b'<script type="text/javascript"  '
    b'src="/7lZ1Ku/BO/Vw/Fq7R/M779h-6pJM/pOpOzL4DcbLNGt9D/MENZaywUEwE/JzUgeAU/PGhQ"></script>'
)


def test_strips_the_script_edgar_injects_before_the_body_end() -> None:
    served = FILING_TAIL + EDGAR_SCRIPT + b"</body></html>"

    assert strip_edgar_script(served) == FILING_TAIL + b"</body></html>"


def test_strips_only_the_tag_and_keeps_the_whitespace_around_it() -> None:
    served = FILING_TAIL + b"\n " + EDGAR_SCRIPT + b"</body>\n</html>"

    assert strip_edgar_script(served) == FILING_TAIL + b"\n </body>\n</html>"


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(FILING_TAIL + b"</body></html>", id="nothing injected"),
        pytest.param(
            b'<script type="text/javascript" src="/js/viewer.js"></script>'
            + FILING_TAIL
            + b"</body></html>",
            id="the filing's own script",
        ),
        pytest.param(
            EDGAR_SCRIPT + FILING_TAIL + b"</body></html>",
            id="the same tag away from the body end",
        ),
        pytest.param(
            FILING_TAIL
            + EDGAR_SCRIPT.replace(b"></script>", b">init()</script>")
            + b"</body></html>",
            id="a script with code in it",
        ),
    ],
)
def test_leaves_everything_else_untouched(document: bytes) -> None:
    assert strip_edgar_script(document) == document
