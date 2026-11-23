# Answer evaluation

Run `2026-11-18-answer-2`: model `openai/gpt-oss-120b` (openai_compatible), embedder `BAAI/bge-small-en-v1.5`, top_k 6, min_relevance 0.58, chunk_tokens 400, index of 4800 chunks, commit `9c48e70`. Raw answers: `eval/answer_runs/2026-11-18-answer-2.jsonl`.

**Correct** means the answer states every expected fact of the question. **Evidence cited** means a cited passage from the filing the question names contains its evidence. **Cited only the other filing** counts answers whose sources from that company all come from its other generation of filings. The figure check (task 3) is not available yet.

## Answerable questions

| Questions | Answered | Correct | Evidence cited | Cited only the other filing | Refused: gate / model / no citations | Errors | Not asked |
|---|---|---|---|---|---|---|---|
| sec-10k · HTM (24) | 23/24 | 21/24 | 20/24 | 0/24 | 0 / 1 / 0 | 0 | 0 |
| sec-10k · TXT (21) | 19/21 | 18/21 | 19/21 | 0/21 | 0 / 2 / 0 | 0 | 0 |

## Unanswerable questions

| Questions | Refused | By the gate | By the model | No valid citations | Answered | Errors | Not asked |
|---|---|---|---|---|---|---|---|
| Off-topic (3) | 3/3 | 3 | 0 | 0 | 0 | 0 | 0 |
| In-domain unanswerable (5) | 5/5 | 0 | 5 | 0 | 0 | 0 | 0 |

## Cost

- LLM calls: 50 (the gate answered 3 without one).
- Latency per question with an LLM call: median 1847 ms, 95th percentile 3626 ms.
- Tokens: 139,196 in, 12,403 out.

## Per question

| Question | Set | Outcome | Correct | Evidence cited | Cited filings | Missing facts | ms | Tokens |
|---|---|---|---|---|---|---|---|---|
| a21 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 1518 | 2,998 |
| a22 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 1160 | 3,027 |
| a23 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 3399 | 3,077 |
| a24 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 1322 | 2,632 |
| a25 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 1953 | 3,117 |
| a26 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 2525 | 3,101 |
| a27 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26, AAPL 2000-09-30 | — | 1685 | 2,849 |
| a28 | sec-10k · HTM (24) | answered | yes | yes | AAPL 2026-09-26 | — | 1927 | 3,226 |
| m21 | sec-10k · HTM (24) | refused (model) | **no** | **no** | — | `223,000` | 1668 | 3,095 |
| m22 | sec-10k · HTM (24) | answered | yes | yes | MSFT 2026-06-30 | — | 2775 | 3,106 |
| m23 | sec-10k · HTM (24) | answered | yes | yes | MSFT 2026-06-30 | — | 3254 | 3,226 |
| m24 | sec-10k · HTM (24) | answered | yes | yes | MSFT 2026-06-30 | — | 1195 | 2,650 |
| m25 | sec-10k · HTM (24) | answered | **no** | **no** | MSFT 2026-06-30 | `133,749\|133\.7\d* ?billion` | 3271 | 3,473 |
| m26 | sec-10k · HTM (24) | answered | yes | yes | MSFT 2026-06-30 | — | 3944 | 3,060 |
| m27 | sec-10k · HTM (24) | answered | yes | yes | MSFT 2026-06-30 | — | 4292 | 2,822 |
| m28 | sec-10k · HTM (24) | answered | yes | **no** | MSFT 2026-06-30 | — | 2916 | 3,056 |
| n21 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25 | — | 2173 | 2,771 |
| n22 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25 | — | 2502 | 3,083 |
| n23 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25 | — | 2403 | 3,113 |
| n24 | sec-10k · HTM (24) | answered | **no** | **no** | NVDA 2001-01-28, NVDA 2026-01-25 | `22 ?(?:%\|percent)`, `14 ?(?:%\|percent)` | 3626 | 3,316 |
| n25 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25, NVDA 2001-01-28 | — | 3581 | 3,140 |
| n26 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25 | — | 2578 | 3,180 |
| n27 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25 | — | 2509 | 2,978 |
| n28 | sec-10k · HTM (24) | answered | yes | yes | NVDA 2026-01-25 | — | 2342 | 2,968 |
| a01 | sec-10k · TXT (21) | refused (model) | **no** | **no** | — | `8,568` | 1155 | 2,814 |
| a02 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30 | — | 1738 | 2,742 |
| a03 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30 | — | 1944 | 2,949 |
| a04 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30, AAPL 2026-09-26 | — | 2000 | 3,223 |
| a05 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30 | — | 2165 | 3,276 |
| a06 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30 | — | 1793 | 2,948 |
| a07 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30 | — | 1840 | 2,559 |
| a08 | sec-10k · TXT (21) | answered | yes | yes | AAPL 2000-09-30 | — | 1263 | 3,027 |
| m01 | sec-10k · TXT (21) | answered | yes | yes | MSFT 2026-06-30, MSFT 2001-06-30 | — | 1793 | 2,897 |
| m02 | sec-10k · TXT (21) | answered | yes | yes | MSFT 2001-06-30 | — | 1669 | 3,530 |
| m03 | sec-10k · TXT (21) | answered | yes | yes | MSFT 2001-06-30 | — | 2570 | 3,453 |
| m04 | sec-10k · TXT (21) | answered | **no** | yes | MSFT 2026-06-30, MSFT 2001-06-30 | `has not paid cash dividends\|not paid any cash dividends\|no cash dividends\|never paid` | 1462 | 3,083 |
| m05 | sec-10k · TXT (21) | answered | yes | yes | MSFT 2001-06-30 | — | 1839 | 3,614 |
| m06 | sec-10k · TXT (21) | answered | yes | yes | MSFT 2001-06-30 | — | 1449 | 3,114 |
| m07 | sec-10k · TXT (21) | answered | yes | yes | MSFT 2001-06-30 | — | 1675 | 2,597 |
| n01 | sec-10k · TXT (21) | answered | yes | yes | NVDA 2001-01-28 | — | 1933 | 3,141 |
| n02 | sec-10k · TXT (21) | answered | yes | yes | NVDA 2001-01-28 | — | 1177 | 2,957 |
| n03 | sec-10k · TXT (21) | answered | yes | yes | NVDA 2001-01-28 | — | 1674 | 3,135 |
| n04 | sec-10k · TXT (21) | refused (model) | **no** | **no** | — | `3dfx` | 1293 | 3,035 |
| n05 | sec-10k · TXT (21) | answered | yes | yes | NVDA 2001-01-28 | — | 1635 | 3,125 |
| n06 | sec-10k · TXT (21) | answered | yes | yes | NVDA 2001-01-28 | — | 2905 | 3,288 |
| u01 | Off-topic (3) | refused (gate) | — | — | — | — | 1334 | — |
| u02 | Off-topic (3) | refused (gate) | — | — | — | — | 1376 | — |
| u03 | Off-topic (3) | refused (gate) | — | — | — | — | 1142 | — |
| u04 | In-domain unanswerable (5) | refused (model) | — | — | — | — | 1854 | 2,644 |
| u08 | In-domain unanswerable (5) | refused (model) | — | — | — | — | 1614 | 2,949 |
| u09 | In-domain unanswerable (5) | refused (model) | — | — | — | — | 1774 | 2,989 |
| u10 | In-domain unanswerable (5) | refused (model) | — | — | — | — | 1733 | 2,666 |
| u11 | In-domain unanswerable (5) | refused (model) | — | — | — | — | 1711 | 2,780 |

## Answers

**a21.** How many full-time equivalent employees did Apple have at the end of fiscal 2026?

> Apple reported that, as of September 26 2026—the end of its fiscal 2026 year—it had approximately 167,000 full‑time‑equivalent employees[1].

**a22.** What were Apple's total net sales in fiscal 2026?

> Apple reported total net sales of $478,834 million for fiscal 2026 [1][2][3]. This figure represents the company’s overall revenue for the year.

**a23.** How much revenue did Apple earn from the iPhone in 2026?

> Apple’s iPhone net sales were $254,098 million for the fiscal year 2026[1][2].

**a24.** What was Apple's effective tax rate in 2026?

> Apple’s effective tax rate for 2026 was 17.0%[1][2].

**a25.** How many of its own shares did Apple buy back in 2026, and for how much?

> Apple repurchased roughly 297 million of its own shares in 2026, spending about $85.8 billion on the buy‑back [1]. The detailed share‑count table in the same filing shows a repurchase of 296,699 thousand shares (≈ 296.7 million) for that amount [1].

**a26.** Where are Apple's manufacturing partners located?

> Apple’s manufacturing partners are concentrated in Asia, with the bulk of outsourcing partners located in China mainland, India, Japan, South Korea, Taiwan and Vietnam, and the company also sources components from partners and facilities in the United States[1][2]. In addition, Apple relies on single‑source partners in the U.S., Europe and other parts of Asia for certain components and final assembly[1]. These locations reflect the primary regions from which Apple obtains the manufacturing and logistical services needed for its hardware products[3][2].

**a27.** How many shareholders of record did Apple have?

> Apple reported 21,458 shareholders of record as of October 23, 2026 [1]. Earlier, the company had 24,998 shareholders of record as of December 1, 2000 [2].

**a28.** How much did Apple's net sales in Greater China grow in 2026?

> Apple’s Greater China segment saw net sales rise from $64,377 million in 2025 to $82,378 million in 2026, a year‑over‑year increase of 28% [1]. This growth was driven primarily by higher iPhone and Mac sales in the region [1].

**m21.** How many people did Microsoft employ at the end of fiscal 2026?

> *Refused (model).*

**m22.** How fast did Microsoft's Azure revenue grow in fiscal 2026?

> Microsoft’s Azure (and other cloud services) revenue grew 41% in fiscal 2026, compared with the prior year [1].

**m23.** What was Microsoft Cloud revenue in fiscal 2026?

> In fiscal 2026 Microsoft reported Cloud revenue of $214.4 billion, covering Microsoft 365 Commercial cloud, Azure and other cloud services, the commercial portion of LinkedIn, and Dynamics 365[1]. This amount is also noted as a 27 % increase to $214.4 billion in the Management’s Discussion and Analysis for FY 2026[2].

**m24.** What was Microsoft's effective tax rate for fiscal 2026?

> Microsoft’s effective tax rate for fiscal 2026 was 19% [1].

**m25.** What was Microsoft's net income in fiscal 2026?

> Microsoft’s income before income taxes in fiscal 2026 was $103.6 billion in the United States plus $62.3 billion abroad, or about $165.9 billion total [1].  The company’s effective tax rate for the year was 19% [1], which translates to roughly $31.5 billion of income‑tax expense.  Subtracting that tax from pre‑tax earnings gives net income of approximately $134 billion for fiscal 2026.

**m26.** How much did Microsoft pay for Activision Blizzard?

> Microsoft completed its acquisition of Activision Blizzard on October 13, 2023 for a total purchase price of $75.4 billion, paid primarily in cash[1][2].

**m27.** How many registered holders of Microsoft common stock were there in 2026?

> There were 74,199 registered holders of Microsoft’s common stock as of July 23, 2026[1].

**m28.** What cash dividends per share did Microsoft declare in fiscal 2026?

> Microsoft’s Board of Directors declared a cash dividend of **$0.91 per share** for each of the four quarterly dividends paid in fiscal 2026 — the September 2025, December 2025, March 2026 and June 2026 declarations all listed a $0.91 per‑share amount[1][2].

**n21.** How many employees did NVIDIA have at the end of fiscal 2026?

> NVIDIA reported that it had approximately 42,000 employees at the end of fiscal year 2026 [1].

**n22.** What was NVIDIA's revenue for fiscal year 2026?

> NVIDIA reported total revenue of $215.9 billion for the fiscal year ended January 25, 2026 (shown as $215,938 million in the financial table)[1][2].

**n23.** How much revenue did NVIDIA's Data Center market generate in fiscal 2026?

> NVIDIA’s Data Center end‑market revenue for fiscal year 2026 was $193,737 million (approximately $193.7 billion)[1].

**n24.** How dependent is NVIDIA on a few large customers?

> NVIDIA’s revenue has long been concentrated in a handful of buyers. In fiscal 2001, its two biggest customers supplied about 36 % of total revenue—25 % from Edom and 11 % from Celestica [1]. The 2001 filing also warned that the company “has only a limited number of customers and our sales are highly concentrated,” and that a small group of add‑in board manufacturers, CEMs and PC OEMs would continue to account for a substantial portion of revenue [2]. The 2026 risk‑factors disclosure repeats this theme, noting that “a limited number of indirect customers… individually represent 10 % or more of our revenue” and that losing any large customer would hurt its financial condition [3]. Together, these passages show NVIDIA remains heavily dependent on a few large customers.

**n25.** Which foundries produce NVIDIA's semiconductor wafers?

> NVIDIA’s semiconductor wafers are fabricated by third‑party foundries. In its 2026 Form 10‑K the company states that it uses “foundries, such as Taiwan Semiconductor Manufacturing Company Limited, or TSMC, and Samsung Electronics Co., Ltd., or Samsung” for wafer production [1]. Earlier filings (2001) identified TSMC as its primary wafer manufacturer [2][3].

**n26.** How many shares did NVIDIA repurchase in fiscal 2026?

> NVIDIA repurchased 282 million shares of its common stock during fiscal 2026 [1][2][3][4].

**n27.** How much did NVIDIA spend on research and development in fiscal 2026?

> NVIDIA’s research and development expense for fiscal 2026 was $18,497 million (about $18.5 billion)[1].

**n28.** What did the acquisition of Mellanox add to NVIDIA's business?

> NVIDIA’s 2020 purchase of Mellanox broadened the company’s portfolio by adding networking products and capabilities, which let its platforms scale to data‑center size. The deal also paved the way for a new class of processors – the data processing unit (DPU) – to complement its GPUs and CPUs [1].

**a01.** How many people worked at Apple at the end of fiscal 2000?

> *Refused (model).*

**a02.** What were Apple's total net sales in fiscal 2000?

> Apple reported total net sales of $7.983 billion for fiscal 2000 (the year ended September 30, 2000)[1][2]. This amount is shown in the consolidated statements of operations as $7,983 million and is confirmed by the geographic sales breakdown, which also totals $7,983 million for the year[3].

**a03.** How much did Apple spend on research and development in 2000?

> Apple’s research and development spending in 2000 was $380 million [1][2]. Both the business overview and the management discussion note that R&D expenditures rose to that amount for the year.

**a04.** What happens to Apple if a key component only available from one vendor is delayed?

> If a key component that Apple obtains from a single or limited source is delayed, the company may be unable to ship the related products in the quantities and timing it desires, which can hurt its business and financial performance. The impact depends on how long it takes Apple to secure enough of the component from the original vendor or to find an alternative supplier [1]. Similar language appears in later filings, noting that delays can prevent Apple from obtaining sufficient quantities on commercially reasonable terms and may adversely affect its results of operations [2].

**a05.** What is the technical foundation of Apple's next operating system?

> Apple’s forthcoming client operating system, Mac OS X, will be built on an open‑source UNIX‑based foundation[1].

**a06.** What special bonus did Steve Jobs receive for serving as interim CEO?

> Steve Jobs was granted a special executive bonus for his interim‑CEO service in the form of an aircraft—a Gulfstream V—valued at about $90 million to the company[1][2].

**a07.** What was Apple's effective tax rate for 2000?

> Apple’s effective tax rate for the year 2000 was 28% [1][2].

**a08.** Which internet service provider did Apple partner with to offer access to Macintosh users?

> Apple partnered with EarthLink Network Inc. (EarthLink), an Internet service provider, to deliver ISP service to Macintosh users in the United States[1][2].

**m01.** How many employees did Microsoft have?

> Microsoft employed about 223,000 full‑time people as of June 30, 2026 [1]. Earlier, the company reported roughly 47,600 full‑time employees on June 30, 2001 [2].

**m02.** How much did Microsoft spend on research and development in 2001?

> Microsoft spent about $4.38 billion on research and development in fiscal 2001 [1]. The same amount is reflected in the detailed financial table, which lists R&D expense as $4,379 million for 2001 [2].

**m03.** What is the status of the US government's antitrust case against Microsoft?

> The antitrust case U.S. v. Microsoft, brought by the DOJ and state attorneys general, was partially overturned when the D.C. Circuit on June 28 2001 affirmed some findings, reversed others and vacated the June 7 2000 breakup judgment, remanding the matter for a new trial on one Section 1 claim and for a new judgment consistent with its ruling [1].  On September 6 2001 the plaintiffs said they would no longer seek to break up Microsoft, would pursue conduct‑remedy relief, and would not retry the remaining Section 1 claim [1].  Microsoft responded by petitioning the Supreme Court for a writ of certiorari on August 7 2001 and may seek further review after a final judgment is entered [2].  The case therefore remains active, with the original breakup order vacated, the matter remanded, and further appellate and possibly Supreme Court review pending.

**m04.** Does Microsoft return cash to shareholders through dividends?

> Yes. In its 2026 Form 10‑K Microsoft reported that its Board declared cash dividends of $27 billion for fiscal 2026 (and $24.7 billion for fiscal 2025) and said it will continue returning capital to shareholders through dividends[1]. By contrast, the 2001 filing noted that Microsoft had not paid cash dividends on its common stock at that time[2].

**m05.** How much revenue came from Desktop Applications at Microsoft?

> Microsoft’s Desktop Applications segment generated roughly $9.5 billion of revenue in fiscal 2001.  The Management’s Discussion and Analysis section lists Desktop Applications revenue as $9.54 billion for the year ended June 30 2001 [1].  The segment‑information table reports Desktop Applications revenue as $9,580 million (i.e., $9.58 billion) for the same fiscal year [2].

**m06.** When is Windows XP expected to reach customers?

> Windows XP was slated for widespread availability in October 2001, according to Microsoft’s 2001 Form 10‑K filing [1]. This date reflects the company’s expectation for when customers would be able to obtain the new operating system.

**m07.** How many registered shareholders did Microsoft have?

> Microsoft reported that, as of July 31 2001, it had **112,001 registered holders of record** of its common stock[1].

**n01.** Who are NVIDIA's largest customers?

> NVIDIA’s two biggest customers in fiscal 2001 were **Edom**, which accounted for about 25 % of the company’s revenue, and **Celestica**, which contributed roughly 11 % of revenue, together representing about 36 % of total sales that year[1][2].

**n02.** Which foundry manufactures NVIDIA's chips?

> NVIDIA’s graphics processors are fabricated by TSMC, which serves as the foundry for its chips[1].

**n03.** How much money did Microsoft prepay NVIDIA for the game console chips?

> Microsoft prepaid NVIDIA $200 million for the Xbox graphics chips in April 2000, as disclosed in NVIDIA’s 2001 Form 10‑K filings[1][2].

**n04.** What did NVIDIA buy from its former graphics rival?

> *Refused (model).*

**n05.** How many engineers did NVIDIA employ in research and development?

> NVIDIA had 458 full‑time employees engaged in research and development as of January 28 2001, up from 214 employees on January 30 2000[1][2].

**n06.** What interest rate do NVIDIA's convertible notes pay?

> NVIDIA’s convertible subordinated notes carry a fixed interest rate of 4 ¾ percent per year (4.75% annually) [1][2][3]. This rate is applied to the notes from issuance and is payable semi‑annually.

**u01.** What was the weather like in Paris last summer?

> *Refused (gate).*

**u02.** What is a good recipe for chocolate cake?

> *Refused (gate).*

**u03.** Who won the football World Cup in 2018?

> *Refused (gate).*

**u04.** How many retail stores does Apple operate?

> *Refused (model).*

**u08.** What was Tesla's revenue last year?

> *Refused (model).*

**u09.** How many Xbox Series X consoles did Microsoft sell in fiscal 2026?

> *Refused (model).*

**u10.** How many daily active users does Microsoft Teams have?

> *Refused (model).*

**u11.** How many apps are available on Apple's App Store?

> *Refused (model).*
