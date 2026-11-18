from groundwork.ingestion.loaders import CELL_SEPARATOR, LoadedDocument, load_html
from groundwork.ingestion.structure.sec import FRONT_MATTER, detect_sec_items

# Synthetic, but shaped like the real inline XBRL 10-Ks: hidden facts in ix:header, a table of
# contents laid out as a table, headings split across spans or joined by &nbsp;, and financial
# tables with "$" and ")" in cells of their own.
FILING = """<?xml version='1.0' encoding='ASCII'?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<head><title>acme-20250131</title><style>.x { color: red; }</style></head>
<body>
<div style="display:none"><ix:header><ix:hidden>
  <ix:nonNumeric name="dei:AmendmentFlag" contextRef="c-1">false</ix:nonNumeric>
</ix:hidden><ix:resources><xbrli:context id="c-1">us-gaap:SecretContext</xbrli:context>
</ix:resources></ix:header></div>
<div><span>ACME Corp</span></div>
<table>
  <tr><td><a href="#i1">Item 1.</a></td><td>Business</td><td>3</td></tr>
  <tr><td><a href="#i7">Item 7.</a></td><td>Management's Discussion</td><td>9</td></tr>
</table>
<div><span>3</span></div>
<hr style="page-break-after:always"/>
<div id="i1"><span>ITEM 1. B</span><span>USINESS</span></div>
<div><span>ACME Corp sells anvils to </span><span>blacksmiths.</span>
  <span style="display:none">Hidden note.</span></div>
<div><span>&#8226;</span></div>
<div><span>Anvils for coyotes.</span></div>
<div><span>&#8226;</span><span style="padding-left:9pt">Rockets too.</span></div>
<div id="i7"><span>Item&#160;7.&#160;&#160;Management's Discussion</span></div>
<div>Line one<br/>line two</div>
<table>
  <tr><td></td><td colspan="3">2025</td><td></td><td colspan="3">2024</td></tr>
  <tr><td>Net sales</td><td>$</td>
      <td><ix:nonFraction name="us-gaap:Revenues">1,234</ix:nonFraction></td><td></td>
      <td>$</td><td>(5,678</td><td>)</td></tr>
  <tr><td>Growth</td><td></td><td>12</td><td>%</td><td style="display:none">99</td></tr>
</table>
<script>var tracking = "do not index";</script>
<!-- a comment that must not appear -->
</body></html>
"""


def text() -> str:
    return load_html(FILING.encode("ascii")).text


def test_hidden_xbrl_scripts_styles_and_comments_are_removed() -> None:
    loaded = text()

    for leftover in ("AmendmentFlag", "false", "SecretContext", "us-gaap", "acme-20250131"):
        assert leftover not in loaded
    assert "color: red" not in loaded
    assert "tracking" not in loaded
    assert "comment" not in loaded
    assert "Hidden note" not in loaded
    assert "99" not in loaded


def test_inline_elements_join_without_added_spaces() -> None:
    loaded = text()

    assert "ITEM 1. BUSINESS" in loaded
    assert "ACME Corp sells anvils to blacksmiths." in loaded


def test_block_elements_become_paragraphs() -> None:
    assert "ACME Corp\n\n" in text()


def test_line_breaks_are_kept_inside_a_paragraph() -> None:
    assert "Line one\nline two" in text()


def test_non_breaking_spaces_become_ordinary_spaces() -> None:
    assert "Item 7. Management's Discussion" in text()


def test_table_rows_become_lines_with_their_columns() -> None:
    loaded = text()

    # The header row starts with the empty label column, so each year sits over its values.
    assert "| 2025 | 2024\nNet sales | $1,234 | $(5,678)\nGrowth | 12%" in loaded


def test_page_numbers_are_dropped_and_bullets_join_their_text() -> None:
    paragraphs = text().split("\n\n")

    assert "3" not in paragraphs
    assert "• Anvils for coyotes." in paragraphs
    assert "• Rockets too." in paragraphs


def test_sections_skip_the_table_of_contents() -> None:
    loaded = text()

    titles = [section.path[0] for section in detect_sec_items(LoadedDocument(loaded))]

    assert f"Item 1.{CELL_SEPARATOR}Business" in loaded
    assert titles == [FRONT_MATTER, "Item 1 — Business", "Item 7 — Management's Discussion"]


def test_plain_html_without_xbrl_loads() -> None:
    page = b"<html><body><h1>ACME</h1><p>Anvils &amp; more.</p></body></html>"

    assert load_html(page).text == "ACME\n\nAnvils & more."


def test_hidden_attribute_hides_content() -> None:
    assert load_html(b"<body><p hidden>secret</p><p>shown</p></body>").text == "shown"


def test_nested_table_rows_are_not_repeated() -> None:
    page = b"<table><tr><td>outer</td><td><table><tr><td>inner</td></tr></table></td></tr></table>"

    assert load_html(page).text == "outer | inner"


def table_text(rows: str) -> str:
    return load_html(f"<html><body><table>{rows}</table></body></html>".encode()).text


def test_a_blank_in_a_row_keeps_later_values_in_their_columns() -> None:
    # Apple's share repurchase table: the Total row has no average price and no plan count.
    rendered = table_text(
        "<tr><td>Periods</td><td></td><td>Shares</td><td></td><td>Price</td><td></td>"
        "<td>Under plans</td><td></td><td>May yet be purchased</td></tr>"
        "<tr><td>Open market</td><td></td><td>27,594</td><td>$</td><td>317.45</td><td></td>"
        "<td>27,594</td><td></td><td></td></tr>"
        "<tr><td>Total</td><td></td><td>81,767</td><td></td><td></td><td></td><td></td>"
        "<td>$</td><td>114,004</td></tr>"
    )

    assert rendered.split("\n") == [
        "Periods | Shares | Price | Under plans | May yet be purchased",
        "Open market | 27,594 | $317.45 | 27,594",
        "Total | 81,767 | | | $114,004",
    ]


def test_spanning_headers_sit_over_their_values() -> None:
    rendered = table_text(
        '<tr><td></td><td colspan="3">2026</td><td></td><td colspan="3">2025</td></tr>'
        "<tr><td>Revenue</td><td>$</td><td>100</td><td></td><td></td><td>$</td><td>90</td>"
        "<td></td></tr>"
        "<tr><td>Margin</td><td></td><td>40</td><td>%</td><td></td><td></td><td>(5</td>"
        "<td>)</td></tr>"
    )

    assert rendered.split("\n") == ["| 2026 | 2025", "Revenue | $100 | $90", "Margin | 40% | (5)"]


def test_a_title_spanning_the_table_is_kept() -> None:
    rendered = table_text(
        '<tr><td colspan="3">Share repurchases</td></tr><tr><td>Q1</td><td></td><td>5</td></tr>'
    )

    assert rendered.split("\n") == ["Share repurchases", "Q1 | 5"]


def test_a_rowspan_pushes_the_next_row_right() -> None:
    rendered = table_text(
        '<tr><td rowspan="2">Products</td><td>iPhone</td><td>200</td></tr>'
        "<tr><td>Mac</td><td>30</td></tr>"
    )

    assert rendered.split("\n") == ["Products | iPhone | 200", "| Mac | 30"]
