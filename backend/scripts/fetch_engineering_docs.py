"""Download the Kubernetes concept pages into data/raw/engineering/, pinned to one commit.

Every page under content/en/docs/concepts/ of github.com/kubernetes/website at REVISION, except
the section index pages (`_index.md`), which are mostly lists of links. The pages are Hugo
Markdown; each is rewritten as plain Markdown (see `convert`) and listed in the manifest with its
source URL at the pinned commit. Running the script again gives the same bytes.

The Kubernetes documentation is licensed under CC BY 4.0. The manifest records the licence, the
attribution and the fact that the pages were converted.

Run from backend/: uv run python scripts/fetch_engineering_docs.py
"""

import hashlib
import re
import time
from datetime import date
from pathlib import PurePosixPath
from typing import Any

import httpx
import yaml

from groundwork.config import Settings
from groundwork.ingestion.corpus import MANIFEST, Manifest, ManifestDocument, write_manifest

COLLECTION = "engineering"
REPOSITORY = "kubernetes/website"
REVISION = "6dc0b2fbf8b78a76f8fff211cfd5bf7b558ca751"
# The commit's date: the pages are dated by the snapshot they come from.
REVISION_DATE = date(2026, 11, 18)
DOCS_ROOT = "content/en/docs/concepts"
SECONDS_BETWEEN_REQUESTS = 0.1

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')
# {{< name attributes >}} and {{% name attributes %}}, opening or closing.
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w-]+)([^}]*?)\s*[>%]\}\}")
_CALLOUTS = {"note": "Note", "caution": "Caution", "warning": "Warning"}
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
# Hugo's custom heading IDs: "## Functions {#functions-of-the-ccm}".
_HEADING_ID = re.compile(r"^(#{1,6} .*?)\s*\{#[\w.-]+\}[ \t]*$", re.MULTILINE)


def _shortcode(match: re.Match[str]) -> str:
    closing, name, rest = match[1], match[2], match[3]
    attributes: dict[str, str] = dict(_ATTRIBUTE.findall(rest))
    if closing:
        return ""
    if name == "glossary_tooltip":
        return attributes.get("text") or attributes.get("term_id", "").replace("-", " ")
    if name in _CALLOUTS:
        return f"**{_CALLOUTS[name]}:** "
    if name == "feature-state":
        # Newer pages name the feature gate and let Hugo look up its state; the gate's name is
        # the useful part here.
        if gate := attributes.get("feature_gate_name"):
            return f"Feature gate: `{gate}`"
        return (
            f"FEATURE STATE: Kubernetes {attributes.get('for_k8s_version', '')} "
            f"[{attributes.get('state', '')}]"
        )
    if name == "code_sample":
        return f"(Example: `{attributes.get('file', '')}`)"
    # Tabs, tables of contents, version variables and the like carry no text of their own.
    return ""


def convert(raw: str) -> tuple[dict[str, Any], str]:
    """Hugo Markdown to plain Markdown: the front matter is returned apart, its description
    opens the text, comments go, and shortcodes become the text they stand for."""
    text = raw.replace("\r\n", "\n")
    front: dict[str, Any] = {}
    if match := _FRONT_MATTER.match(text):
        front = yaml.safe_load(match[1]) or {}
        text = text[match.end() :]
    text = _COMMENT.sub("", text)
    text = _SHORTCODE.sub(_shortcode, text)
    text = _HEADING_ID.sub(r"\1", text)
    description = " ".join(str(front.get("description", "")).split())
    if description:
        text = f"{description}\n\n{text}"
    text = _EXTRA_BLANK_LINES.sub("\n\n", text).strip() + "\n"
    return front, text


def output_name(page: str) -> str:
    """workloads/controllers/deployment.md -> workloads--controllers--deployment.md"""
    return page.replace("/", "--")


def document_key(page: str) -> str:
    return PurePosixPath(page).with_suffix("").as_posix().replace("/", "-").replace("_", "-")


def list_pages(client: httpx.Client) -> list[str]:
    """Paths under DOCS_ROOT of every page at REVISION, sorted."""
    parent, name = DOCS_ROOT.rsplit("/", 1)
    listing = client.get(
        f"https://api.github.com/repos/{REPOSITORY}/contents/{parent}", params={"ref": REVISION}
    )
    listing.raise_for_status()
    tree_sha = next(entry["sha"] for entry in listing.json() if entry["name"] == name)
    tree = client.get(
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/{tree_sha}", params={"recursive": "1"}
    )
    tree.raise_for_status()
    body = tree.json()
    if body.get("truncated"):
        raise RuntimeError("GitHub truncated the tree listing; the page list would be incomplete.")
    return sorted(
        entry["path"]
        for entry in body["tree"]
        if entry["type"] == "blob"
        and entry["path"].endswith(".md")
        and not entry["path"].endswith("_index.md")
    )


def main() -> None:
    settings = Settings()
    folder = settings.data_dir / "raw" / COLLECTION
    folder.mkdir(parents=True, exist_ok=True)
    documents = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for page in list_pages(client):
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{DOCS_ROOT}/{page}"
            response = client.get(url)
            response.raise_for_status()
            time.sleep(SECONDS_BETWEEN_REQUESTS)
            front, text = convert(response.text)
            content = text.encode()
            name = output_name(page)
            (folder / name).write_bytes(content)
            documents.append(
                ManifestDocument(
                    key=document_key(page),
                    filename=name,
                    title=str(front.get("title") or PurePosixPath(page).stem),
                    doc_type=str(front.get("content_type") or "concept"),
                    doc_date=REVISION_DATE,
                    metadata={"section": page.split("/")[0]},
                    source_url=f"https://github.com/{REPOSITORY}/blob/{REVISION}/{DOCS_ROOT}/{page}",
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
            print(f"{name}: {len(content):,} bytes")
    manifest = Manifest(
        collection=COLLECTION,
        description="Concept pages of the Kubernetes documentation: technical prose in Markdown "
        "with nested headings, code and tables.",
        source=f"https://github.com/{REPOSITORY}/tree/{REVISION}/{DOCS_ROOT}",
        licence="CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). © The Kubernetes "
        "Authors. Changed: Hugo front matter and shortcodes were rewritten as plain Markdown, "
        "and HTML comments removed.",
        revision=REVISION,
        documents=tuple(documents),
    )
    (folder / MANIFEST).write_text(write_manifest(manifest))
    print(f"{len(documents)} pages")


if __name__ == "__main__":
    main()
