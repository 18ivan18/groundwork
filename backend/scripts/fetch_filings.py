"""Download annual reports for each company from SEC EDGAR into data/raw/sec-10k/.

Each company gets its latest 10-K (HTML) and its newest plain-text annual report. EDGAR stopped
receiving plain-text filings around 2001, so the text files are old, but they are real filings
that need no conversion, which makes them the corpus for the first TXT-only runs.

Run from backend/: uv run python scripts/fetch_filings.py
"""

import hashlib
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from groundwork.config import Settings
from groundwork.ingestion.corpus import (
    MANIFEST,
    SEC_COLLECTION,
    Manifest,
    ManifestDocument,
    write_manifest,
)

# Ticker -> SEC Central Index Key. CIKs are permanent, while ticker lookups need another request.
COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}
# EDGAR lists some names in capitals ("MICROSOFT CORP"). These are the names the filings use in
# their own text, so the library shows them consistently.
DISPLAY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}
# 10-K405 was a 10-K variant, discontinued in 2003. NVIDIA's plain-text filing uses it.
PLAIN_TEXT_FORMS = ("10-K", "10-K405")
# 10-K405 was a variant of the 10-K, so it is found under the same document type. The original
# form stays in the metadata.
DOC_TYPE_BY_FORM = {"10-K405": "10-K"}
# EDGAR allows 10 requests per second; stay well under that.
SECONDS_BETWEEN_REQUESTS = 0.2
# EDGAR serves its HTML documents with a script tag added just before </body>, whose src path
# rotates over time, so two fetches of the same filing differ. Only that exact tag is removed.
_EDGAR_SCRIPT = re.compile(
    rb'<script type="text/javascript"  src="/[A-Za-z0-9_/-]+"></script>(?=</body>)'
)


@dataclass(frozen=True)
class Filing:
    company: str
    ticker: str
    cik: str
    form: str
    period_end: str
    filing_date: str
    accession_number: str
    primary_document: str

    @property
    def source_url(self) -> str:
        # Archive paths use the CIK without zero padding and the accession number without dashes.
        accession = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}"
            f"/{accession}/{self.primary_document}"
        )

    @property
    def filename(self) -> str:
        suffix = Path(self.primary_document).suffix
        return f"{self.ticker}-{self.form}-{self.period_end}{suffix}"


def find_latest_filing(
    submissions: dict[str, Any],
    ticker: str,
    cik: str,
    *,
    forms: Sequence[str] = ("10-K",),
    suffix: str = ".htm",
    older_blocks: Iterable[dict[str, Any]] = (),
) -> Filing:
    """Pick the newest matching filing from an EDGAR submissions response.

    The exact form match excludes amendments (10-K/A), which are often partial documents.
    `older_blocks` are the extra files EDGAR splits long filing histories into.
    """
    if ticker not in submissions["tickers"]:
        raise ValueError(f"CIK {cik} belongs to {submissions['tickers']}, not {ticker}")

    blocks = [submissions["filings"]["recent"], *older_blocks]
    candidates = [
        (block, i)
        for block in blocks
        for i, form in enumerate(block["form"])
        if form in forms and block["primaryDocument"][i].endswith(suffix)
    ]
    if not candidates:
        raise ValueError(f"No {' or '.join(forms)} with a {suffix} document for {ticker}")

    # ISO dates sort correctly as strings.
    block, latest = max(candidates, key=lambda c: c[0]["filingDate"][c[1]])
    return Filing(
        company=submissions["name"],
        ticker=ticker,
        cik=cik,
        form=block["form"][latest],
        period_end=block["reportDate"][latest],
        filing_date=block["filingDate"][latest],
        accession_number=block["accessionNumber"][latest],
        primary_document=block["primaryDocument"][latest],
    )


def strip_edgar_script(content: bytes) -> bytes:
    """The document without the script tag EDGAR injects before </body>; else unchanged."""
    return _EDGAR_SCRIPT.sub(b"", content)


def manifest_entry(filing: Filing, content: bytes) -> ManifestDocument:
    doc_type = DOC_TYPE_BY_FORM.get(filing.form, filing.form)
    return ManifestDocument(
        key=Path(filing.filename).stem.lower(),
        filename=filing.filename,
        title=f"{filing.company} · Form {doc_type}",
        doc_type=doc_type,
        doc_date=date.fromisoformat(filing.period_end),
        metadata={"company": filing.company, "ticker": filing.ticker, "form": filing.form},
        source_url=filing.source_url,
        provenance={
            "cik": filing.cik,
            "filing_date": filing.filing_date,
            "accession_number": filing.accession_number,
            "primary_document": filing.primary_document,
        },
        sha256=hashlib.sha256(content).hexdigest(),
    )


def fetch(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    response.raise_for_status()
    time.sleep(SECONDS_BETWEEN_REQUESTS)
    return response


MISSING_USER_AGENT = 'SEC_USER_AGENT is required by EDGAR. Set it in .env: "Your Name your@email"'


def sec_client(user_agent: str) -> httpx.Client:
    """A client for EDGAR, which requires a User-Agent naming the requester."""
    return httpx.Client(headers={"User-Agent": user_agent}, timeout=60)


def fetch_filings(client: httpx.Client, folder: Path) -> Manifest:
    """Download every company's filings into folder and write its manifest there."""
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for ticker, cik in COMPANIES.items():
        submissions = fetch(client, f"https://data.sec.gov/submissions/CIK{cik}.json").json()
        older_blocks = [
            fetch(client, f"https://data.sec.gov/submissions/{file['name']}").json()
            for file in submissions["filings"]["files"]
        ]
        filings = [
            find_latest_filing(submissions, ticker, cik),
            find_latest_filing(
                submissions,
                ticker,
                cik,
                forms=PLAIN_TEXT_FORMS,
                suffix=".txt",
                older_blocks=older_blocks,
            ),
        ]
        for found in filings:
            filing = replace(found, company=DISPLAY_NAMES[ticker])
            # Stripped before hashing and writing, so rebuilds are reproducible. Fetches from
            # corpora-v2 on are stripped: corpora-v1 holds the committed bytes, which still have
            # the tag, so a new corpora version needs an eval rerun.
            content = strip_edgar_script(fetch(client, filing.source_url).content)
            (folder / filing.filename).write_bytes(content)
            entries.append(manifest_entry(filing, content))
            print(f"{filing.filename}: {len(content):,} bytes, filed {filing.filing_date}")

    manifest = Manifest(
        collection=SEC_COLLECTION,
        description="Annual reports (Form 10-K) of Apple, Microsoft and NVIDIA: the latest in "
        "inline XBRL HTML and the newest in plain text, from around 2000.",
        source="SEC EDGAR",
        licence="Public filings published by the SEC on EDGAR.",
        documents=tuple(entries),
    )
    (folder / MANIFEST).write_text(write_manifest(manifest))
    return manifest


def main() -> None:
    settings = Settings()
    if not settings.sec_user_agent:
        sys.exit(MISSING_USER_AGENT)
    with sec_client(settings.sec_user_agent) as client:
        fetch_filings(client, settings.data_dir / "raw" / SEC_COLLECTION)


if __name__ == "__main__":
    main()
