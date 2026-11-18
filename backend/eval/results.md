# Retrieval evaluation

Embedder: `BAAI/bge-small-en-v1.5`. Overlap: 60 tokens. Candidates per retriever: 20. RRF k: 60. Re-ranker: `jinaai/jina-reranker-v1-tiny-en` on the top 20 after fusion. Documents indexed: 205 in books, engineering, northwind, sec-10k. Questions: 111 answerable, 15 unanswerable. Unfiltered searches over every indexed collection; a question counts as found only in the documents it names.

## Success, recall and MRR

**S@k** (Success@k): share of questions with any of their evidence in the top k (called R@k before M1). **Recall@k**: mean share of each question's evidence found in the top k; it differs from S@k only for questions that need several facts. MRR counts ranks up to 10.

| Chunk tokens | Chunks | Mode | Questions | S@1 | S@3 | S@6 | S@8 | S@10 | S@20 | Recall@6 | Recall@20 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 9379 | reranked | books (13) | 0.31 | 0.62 | 0.62 | 0.77 | 0.77 | 0.77 | 0.50 | 0.73 | 0.458 |
| 200 | 9379 | reranked | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.667 |
| 200 | 9379 | reranked | books · single (11) | 0.27 | 0.55 | 0.55 | 0.73 | 0.73 | 0.73 | 0.50 | 0.68 | 0.420 |
| 200 | 9379 | reranked | engineering (27) | 0.59 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.98 | 0.749 |
| 200 | 9379 | reranked | engineering · multi_hop (3) | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.83 | 0.733 |
| 200 | 9379 | reranked | engineering · single (24) | 0.58 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.751 |
| 200 | 9379 | reranked | northwind (26) | 0.88 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.933 |
| 200 | 9379 | reranked | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 200 | 9379 | reranked | northwind · MD (23) | 0.91 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.946 |
| 200 | 9379 | reranked | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | reranked | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 200 | 9379 | reranked | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.000 |
| 200 | 9379 | reranked | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.583 |
| 200 | 9379 | reranked | northwind · single (17) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | reranked | sec-10k (45) | 0.56 | 0.76 | 0.84 | 0.89 | 0.89 | 0.93 | 0.84 | 0.93 | 0.671 |
| 200 | 9379 | reranked | sec-10k · HTM (24) | 0.58 | 0.75 | 0.83 | 0.92 | 0.92 | 0.96 | 0.83 | 0.96 | 0.683 |
| 200 | 9379 | reranked | sec-10k · TXT (21) | 0.52 | 0.76 | 0.86 | 0.86 | 0.86 | 0.90 | 0.86 | 0.90 | 0.656 |
| 200 | 9379 | hybrid | books (13) | 0.00 | 0.31 | 0.38 | 0.46 | 0.69 | 0.77 | 0.27 | 0.73 | 0.195 |
| 200 | 9379 | hybrid | books · multi_hop (2) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.500 |
| 200 | 9379 | hybrid | books · single (11) | 0.00 | 0.18 | 0.27 | 0.36 | 0.64 | 0.73 | 0.23 | 0.68 | 0.140 |
| 200 | 9379 | hybrid | engineering (27) | 0.67 | 0.85 | 0.96 | 0.96 | 0.96 | 1.00 | 0.93 | 0.98 | 0.772 |
| 200 | 9379 | hybrid | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 1.00 | 0.50 | 0.83 | 0.667 |
| 200 | 9379 | hybrid | engineering · single (24) | 0.67 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.785 |
| 200 | 9379 | hybrid | northwind (26) | 0.50 | 0.81 | 0.88 | 0.92 | 0.96 | 1.00 | 0.87 | 1.00 | 0.673 |
| 200 | 9379 | hybrid | northwind · HTML (3) | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.750 |
| 200 | 9379 | hybrid | northwind · MD (23) | 0.48 | 0.83 | 0.87 | 0.91 | 0.96 | 1.00 | 0.85 | 1.00 | 0.663 |
| 200 | 9379 | hybrid | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | hybrid | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 200 | 9379 | hybrid | northwind · multi_hop (3) | 0.33 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.548 |
| 200 | 9379 | hybrid | northwind · restricted (3) | 0.33 | 0.33 | 0.67 | 0.67 | 1.00 | 1.00 | 0.67 | 1.00 | 0.450 |
| 200 | 9379 | hybrid | northwind · single (17) | 0.53 | 0.88 | 0.94 | 0.94 | 0.94 | 1.00 | 0.94 | 1.00 | 0.706 |
| 200 | 9379 | hybrid | sec-10k (45) | 0.44 | 0.69 | 0.82 | 0.89 | 0.91 | 0.93 | 0.82 | 0.93 | 0.600 |
| 200 | 9379 | hybrid | sec-10k · HTM (24) | 0.54 | 0.71 | 0.88 | 0.92 | 0.92 | 0.96 | 0.88 | 0.96 | 0.663 |
| 200 | 9379 | hybrid | sec-10k · TXT (21) | 0.33 | 0.67 | 0.76 | 0.86 | 0.90 | 0.90 | 0.76 | 0.90 | 0.528 |
| 200 | 9379 | dense | books (13) | 0.08 | 0.23 | 0.31 | 0.54 | 0.54 | 0.62 | 0.23 | 0.58 | 0.205 |
| 200 | 9379 | dense | books · multi_hop (2) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.50 | 0.500 |
| 200 | 9379 | dense | books · single (11) | 0.00 | 0.18 | 0.27 | 0.55 | 0.55 | 0.64 | 0.23 | 0.59 | 0.151 |
| 200 | 9379 | dense | engineering (27) | 0.56 | 0.81 | 0.96 | 1.00 | 1.00 | 1.00 | 0.94 | 0.98 | 0.721 |
| 200 | 9379 | dense | engineering · multi_hop (3) | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 0.83 | 0.667 |
| 200 | 9379 | dense | engineering · single (24) | 0.58 | 0.79 | 0.96 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 0.728 |
| 200 | 9379 | dense | northwind (26) | 0.38 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.487 |
| 200 | 9379 | dense | northwind · HTML (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.667 |
| 200 | 9379 | dense | northwind · MD (23) | 0.35 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.464 |
| 200 | 9379 | dense | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | dense | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 200 | 9379 | dense | northwind · multi_hop (3) | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.333 |
| 200 | 9379 | dense | northwind · restricted (3) | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.333 |
| 200 | 9379 | dense | northwind · single (17) | 0.35 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.480 |
| 200 | 9379 | dense | sec-10k (45) | 0.36 | 0.64 | 0.87 | 0.89 | 0.89 | 0.93 | 0.87 | 0.93 | 0.533 |
| 200 | 9379 | dense | sec-10k · HTM (24) | 0.38 | 0.62 | 0.88 | 0.92 | 0.92 | 0.96 | 0.88 | 0.96 | 0.547 |
| 200 | 9379 | dense | sec-10k · TXT (21) | 0.33 | 0.67 | 0.86 | 0.86 | 0.86 | 0.90 | 0.86 | 0.90 | 0.517 |
| 200 | 9379 | sparse | books (13) | 0.23 | 0.31 | 0.46 | 0.54 | 0.62 | 0.69 | 0.38 | 0.65 | 0.305 |
| 200 | 9379 | sparse | books · multi_hop (2) | 0.50 | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | 0.562 |
| 200 | 9379 | sparse | books · single (11) | 0.18 | 0.27 | 0.45 | 0.45 | 0.55 | 0.64 | 0.41 | 0.59 | 0.259 |
| 200 | 9379 | sparse | engineering (27) | 0.63 | 0.78 | 0.89 | 0.89 | 0.96 | 0.96 | 0.87 | 0.94 | 0.715 |
| 200 | 9379 | sparse | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.50 | 0.50 | 0.667 |
| 200 | 9379 | sparse | engineering · single (24) | 0.62 | 0.79 | 0.92 | 0.92 | 1.00 | 1.00 | 0.92 | 1.00 | 0.721 |
| 200 | 9379 | sparse | northwind (26) | 0.69 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.810 |
| 200 | 9379 | sparse | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 200 | 9379 | sparse | northwind · MD (23) | 0.70 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.807 |
| 200 | 9379 | sparse | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | sparse | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.333 |
| 200 | 9379 | sparse | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | sparse | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.567 |
| 200 | 9379 | sparse | northwind · single (17) | 0.71 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.825 |
| 200 | 9379 | sparse | sec-10k (45) | 0.36 | 0.62 | 0.69 | 0.76 | 0.80 | 0.87 | 0.69 | 0.87 | 0.503 |
| 200 | 9379 | sparse | sec-10k · HTM (24) | 0.38 | 0.62 | 0.71 | 0.79 | 0.83 | 0.92 | 0.71 | 0.92 | 0.523 |
| 200 | 9379 | sparse | sec-10k · TXT (21) | 0.33 | 0.62 | 0.67 | 0.71 | 0.76 | 0.81 | 0.67 | 0.81 | 0.479 |
| 300 | 6101 | reranked | books (13) | 0.31 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.50 | 0.58 | 0.449 |
| 300 | 6101 | reranked | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.75 | 0.750 |
| 300 | 6101 | reranked | books · single (11) | 0.27 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.50 | 0.55 | 0.394 |
| 300 | 6101 | reranked | engineering (27) | 0.70 | 0.89 | 0.96 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.819 |
| 300 | 6101 | reranked | engineering · multi_hop (3) | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.750 |
| 300 | 6101 | reranked | engineering · single (24) | 0.71 | 0.92 | 0.96 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 0.828 |
| 300 | 6101 | reranked | northwind (26) | 0.88 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.933 |
| 300 | 6101 | reranked | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 300 | 6101 | reranked | northwind · MD (23) | 0.91 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.946 |
| 300 | 6101 | reranked | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | reranked | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | reranked | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.000 |
| 300 | 6101 | reranked | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.583 |
| 300 | 6101 | reranked | northwind · single (17) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | reranked | sec-10k (45) | 0.58 | 0.82 | 0.84 | 0.91 | 0.91 | 0.93 | 0.84 | 0.93 | 0.703 |
| 300 | 6101 | reranked | sec-10k · HTM (24) | 0.50 | 0.83 | 0.83 | 0.92 | 0.92 | 0.96 | 0.83 | 0.96 | 0.664 |
| 300 | 6101 | reranked | sec-10k · TXT (21) | 0.67 | 0.81 | 0.86 | 0.90 | 0.90 | 0.90 | 0.86 | 0.90 | 0.748 |
| 300 | 6101 | hybrid | books (13) | 0.23 | 0.54 | 0.62 | 0.62 | 0.62 | 0.62 | 0.54 | 0.58 | 0.397 |
| 300 | 6101 | hybrid | books · multi_hop (2) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.75 | 0.500 |
| 300 | 6101 | hybrid | books · single (11) | 0.27 | 0.45 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.379 |
| 300 | 6101 | hybrid | engineering (27) | 0.63 | 0.93 | 0.96 | 0.96 | 0.96 | 1.00 | 0.93 | 1.00 | 0.773 |
| 300 | 6101 | hybrid | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 1.00 | 0.50 | 1.00 | 0.667 |
| 300 | 6101 | hybrid | engineering · single (24) | 0.62 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.786 |
| 300 | 6101 | hybrid | northwind (26) | 0.54 | 0.85 | 0.96 | 0.96 | 0.96 | 1.00 | 0.96 | 1.00 | 0.697 |
| 300 | 6101 | hybrid | northwind · HTML (3) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.389 |
| 300 | 6101 | hybrid | northwind · MD (23) | 0.61 | 0.83 | 0.96 | 0.96 | 0.96 | 1.00 | 0.96 | 1.00 | 0.737 |
| 300 | 6101 | hybrid | northwind · contradiction (2) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | hybrid | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | hybrid | northwind · multi_hop (3) | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.733 |
| 300 | 6101 | hybrid | northwind · restricted (3) | 0.00 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.361 |
| 300 | 6101 | hybrid | northwind · single (17) | 0.71 | 0.88 | 0.94 | 0.94 | 0.94 | 1.00 | 0.94 | 1.00 | 0.784 |
| 300 | 6101 | hybrid | sec-10k (45) | 0.49 | 0.80 | 0.89 | 0.91 | 0.91 | 0.93 | 0.89 | 0.93 | 0.655 |
| 300 | 6101 | hybrid | sec-10k · HTM (24) | 0.54 | 0.92 | 0.92 | 0.92 | 0.92 | 0.96 | 0.92 | 0.96 | 0.715 |
| 300 | 6101 | hybrid | sec-10k · TXT (21) | 0.43 | 0.67 | 0.86 | 0.90 | 0.90 | 0.90 | 0.86 | 0.90 | 0.585 |
| 300 | 6101 | dense | books (13) | 0.08 | 0.15 | 0.38 | 0.46 | 0.46 | 0.62 | 0.31 | 0.62 | 0.174 |
| 300 | 6101 | dense | books · multi_hop (2) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.50 | 0.500 |
| 300 | 6101 | dense | books · single (11) | 0.00 | 0.09 | 0.36 | 0.45 | 0.45 | 0.64 | 0.32 | 0.64 | 0.115 |
| 300 | 6101 | dense | engineering (27) | 0.56 | 0.85 | 0.96 | 0.96 | 0.96 | 0.96 | 0.93 | 0.96 | 0.713 |
| 300 | 6101 | dense | engineering · multi_hop (3) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.444 |
| 300 | 6101 | dense | engineering · single (24) | 0.62 | 0.83 | 0.96 | 0.96 | 0.96 | 0.96 | 0.94 | 0.96 | 0.747 |
| 300 | 6101 | dense | northwind (26) | 0.38 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 | 0.67 | 0.69 | 0.513 |
| 300 | 6101 | dense | northwind · HTML (3) | 0.00 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.111 |
| 300 | 6101 | dense | northwind · MD (23) | 0.43 | 0.74 | 0.74 | 0.74 | 0.74 | 0.74 | 0.72 | 0.74 | 0.565 |
| 300 | 6101 | dense | northwind · contradiction (2) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| 300 | 6101 | dense | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | dense | northwind · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.50 | 0.67 | 0.667 |
| 300 | 6101 | dense | northwind · restricted (3) | 0.00 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.222 |
| 300 | 6101 | dense | northwind · single (17) | 0.47 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 | 0.598 |
| 300 | 6101 | dense | sec-10k (45) | 0.47 | 0.73 | 0.80 | 0.89 | 0.89 | 0.93 | 0.80 | 0.93 | 0.612 |
| 300 | 6101 | dense | sec-10k · HTM (24) | 0.50 | 0.83 | 0.88 | 0.96 | 0.96 | 0.96 | 0.88 | 0.96 | 0.667 |
| 300 | 6101 | dense | sec-10k · TXT (21) | 0.43 | 0.62 | 0.71 | 0.81 | 0.81 | 0.90 | 0.71 | 0.90 | 0.549 |
| 300 | 6101 | sparse | books (13) | 0.23 | 0.54 | 0.62 | 0.62 | 0.62 | 0.77 | 0.50 | 0.77 | 0.374 |
| 300 | 6101 | sparse | books · multi_hop (2) | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.600 |
| 300 | 6101 | sparse | books · single (11) | 0.18 | 0.55 | 0.55 | 0.55 | 0.55 | 0.73 | 0.50 | 0.73 | 0.333 |
| 300 | 6101 | sparse | engineering (27) | 0.63 | 0.81 | 0.93 | 0.96 | 0.96 | 0.96 | 0.91 | 0.94 | 0.736 |
| 300 | 6101 | sparse | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.667 |
| 300 | 6101 | sparse | engineering · single (24) | 0.62 | 0.83 | 0.96 | 1.00 | 1.00 | 1.00 | 0.94 | 0.98 | 0.745 |
| 300 | 6101 | sparse | northwind (26) | 0.69 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.817 |
| 300 | 6101 | sparse | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 300 | 6101 | sparse | northwind · MD (23) | 0.70 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.814 |
| 300 | 6101 | sparse | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | sparse | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | sparse | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | sparse | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.567 |
| 300 | 6101 | sparse | northwind · single (17) | 0.71 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.825 |
| 300 | 6101 | sparse | sec-10k (45) | 0.31 | 0.58 | 0.78 | 0.84 | 0.84 | 0.89 | 0.78 | 0.89 | 0.487 |
| 300 | 6101 | sparse | sec-10k · HTM (24) | 0.25 | 0.62 | 0.88 | 0.88 | 0.88 | 0.92 | 0.88 | 0.92 | 0.480 |
| 300 | 6101 | sparse | sec-10k · TXT (21) | 0.38 | 0.52 | 0.67 | 0.81 | 0.81 | 0.86 | 0.67 | 0.86 | 0.494 |
| 400 | 4800 | reranked | books (13) | 0.38 | 0.54 | 0.54 | 0.69 | 0.69 | 0.69 | 0.46 | 0.65 | 0.484 |
| 400 | 4800 | reranked | books · multi_hop (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.75 | 1.000 |
| 400 | 4800 | reranked | books · single (11) | 0.27 | 0.45 | 0.45 | 0.64 | 0.64 | 0.64 | 0.41 | 0.64 | 0.390 |
| 400 | 4800 | reranked | engineering (27) | 0.70 | 0.89 | 0.96 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.820 |
| 400 | 4800 | reranked | engineering · multi_hop (3) | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.750 |
| 400 | 4800 | reranked | engineering · single (24) | 0.71 | 0.92 | 0.96 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 0.829 |
| 400 | 4800 | reranked | northwind (26) | 0.88 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.933 |
| 400 | 4800 | reranked | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 400 | 4800 | reranked | northwind · MD (23) | 0.91 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.946 |
| 400 | 4800 | reranked | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | reranked | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 400 | 4800 | reranked | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.000 |
| 400 | 4800 | reranked | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.583 |
| 400 | 4800 | reranked | northwind · single (17) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | reranked | sec-10k (45) | 0.62 | 0.80 | 0.87 | 0.89 | 0.91 | 0.91 | 0.87 | 0.91 | 0.724 |
| 400 | 4800 | reranked | sec-10k · HTM (24) | 0.58 | 0.83 | 0.83 | 0.88 | 0.92 | 0.92 | 0.83 | 0.92 | 0.712 |
| 400 | 4800 | reranked | sec-10k · TXT (21) | 0.67 | 0.76 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.738 |
| 400 | 4800 | hybrid | books (13) | 0.38 | 0.54 | 0.62 | 0.62 | 0.69 | 0.69 | 0.54 | 0.65 | 0.485 |
| 400 | 4800 | hybrid | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 0.75 | 0.750 |
| 400 | 4800 | hybrid | books · single (11) | 0.36 | 0.45 | 0.55 | 0.55 | 0.64 | 0.64 | 0.50 | 0.64 | 0.436 |
| 400 | 4800 | hybrid | engineering (27) | 0.63 | 0.89 | 0.96 | 0.96 | 0.96 | 1.00 | 0.93 | 1.00 | 0.775 |
| 400 | 4800 | hybrid | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 1.00 | 0.50 | 1.00 | 0.667 |
| 400 | 4800 | hybrid | engineering · single (24) | 0.62 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.788 |
| 400 | 4800 | hybrid | northwind (26) | 0.73 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.843 |
| 400 | 4800 | hybrid | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 400 | 4800 | hybrid | northwind · MD (23) | 0.74 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.844 |
| 400 | 4800 | hybrid | northwind · contradiction (2) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.750 |
| 400 | 4800 | hybrid | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 400 | 4800 | hybrid | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 1.000 |
| 400 | 4800 | hybrid | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.583 |
| 400 | 4800 | hybrid | northwind · single (17) | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.892 |
| 400 | 4800 | hybrid | sec-10k (45) | 0.47 | 0.69 | 0.80 | 0.84 | 0.87 | 0.91 | 0.80 | 0.91 | 0.598 |
| 400 | 4800 | hybrid | sec-10k · HTM (24) | 0.54 | 0.71 | 0.83 | 0.88 | 0.88 | 0.92 | 0.83 | 0.92 | 0.650 |
| 400 | 4800 | hybrid | sec-10k · TXT (21) | 0.38 | 0.67 | 0.76 | 0.81 | 0.86 | 0.90 | 0.76 | 0.90 | 0.539 |
| 400 | 4800 | dense | books (13) | 0.08 | 0.31 | 0.46 | 0.46 | 0.46 | 0.69 | 0.42 | 0.65 | 0.218 |
| 400 | 4800 | dense | books · multi_hop (2) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.500 |
| 400 | 4800 | dense | books · single (11) | 0.00 | 0.27 | 0.45 | 0.45 | 0.45 | 0.73 | 0.41 | 0.68 | 0.167 |
| 400 | 4800 | dense | engineering (27) | 0.59 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 0.767 |
| 400 | 4800 | dense | engineering · multi_hop (3) | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.611 |
| 400 | 4800 | dense | engineering · single (24) | 0.62 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.786 |
| 400 | 4800 | dense | northwind (26) | 0.65 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.92 | 0.94 | 0.782 |
| 400 | 4800 | dense | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.778 |
| 400 | 4800 | dense | northwind · MD (23) | 0.65 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 | 0.91 | 0.93 | 0.783 |
| 400 | 4800 | dense | northwind · contradiction (2) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.500 |
| 400 | 4800 | dense | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 400 | 4800 | dense | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.83 | 1.000 |
| 400 | 4800 | dense | northwind · restricted (3) | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.556 |
| 400 | 4800 | dense | northwind · single (17) | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 400 | 4800 | dense | sec-10k (45) | 0.42 | 0.64 | 0.73 | 0.78 | 0.82 | 0.84 | 0.73 | 0.84 | 0.561 |
| 400 | 4800 | dense | sec-10k · HTM (24) | 0.50 | 0.71 | 0.75 | 0.83 | 0.88 | 0.88 | 0.75 | 0.88 | 0.628 |
| 400 | 4800 | dense | sec-10k · TXT (21) | 0.33 | 0.57 | 0.71 | 0.71 | 0.76 | 0.81 | 0.71 | 0.81 | 0.484 |
| 400 | 4800 | sparse | books (13) | 0.31 | 0.54 | 0.54 | 0.62 | 0.62 | 0.77 | 0.50 | 0.77 | 0.434 |
| 400 | 4800 | sparse | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0.750 |
| 400 | 4800 | sparse | books · single (11) | 0.27 | 0.45 | 0.45 | 0.55 | 0.55 | 0.73 | 0.45 | 0.73 | 0.377 |
| 400 | 4800 | sparse | engineering (27) | 0.56 | 0.85 | 0.89 | 0.93 | 0.96 | 0.96 | 0.87 | 0.96 | 0.703 |
| 400 | 4800 | sparse | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.667 |
| 400 | 4800 | sparse | engineering · single (24) | 0.54 | 0.88 | 0.92 | 0.96 | 1.00 | 1.00 | 0.90 | 1.00 | 0.707 |
| 400 | 4800 | sparse | northwind (26) | 0.69 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.817 |
| 400 | 4800 | sparse | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 400 | 4800 | sparse | northwind · MD (23) | 0.70 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.814 |
| 400 | 4800 | sparse | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | sparse | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 400 | 4800 | sparse | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | sparse | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.567 |
| 400 | 4800 | sparse | northwind · single (17) | 0.71 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.825 |
| 400 | 4800 | sparse | sec-10k (45) | 0.36 | 0.67 | 0.71 | 0.76 | 0.82 | 0.89 | 0.71 | 0.89 | 0.512 |
| 400 | 4800 | sparse | sec-10k · HTM (24) | 0.38 | 0.75 | 0.75 | 0.75 | 0.79 | 0.88 | 0.75 | 0.88 | 0.539 |
| 400 | 4800 | sparse | sec-10k · TXT (21) | 0.33 | 0.57 | 0.67 | 0.76 | 0.86 | 0.90 | 0.67 | 0.90 | 0.480 |

## Candidates before fusion

Each retriever passes its top 20 passages to RRF. *Either* is the share found by at least one of them: the most hybrid search can return. *Lost by fusion* lists questions found by a retriever but not in the fused top 6.

| Chunk tokens | Questions | Dense S@20 | BM25 S@20 | Either S@20 | Hybrid S@6 | Only dense | Only BM25 | Lost by fusion |
|---|---|---|---|---|---|---|---|---|
| 200 | books (13) | 0.62 | 0.69 | 0.77 | 0.38 | bk-05 | bk-04, bk-12 | bk-04, bk-05, bk-09, bk-10, bk-11 |
| 200 | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | none | bk-12 | none |
| 200 | books · single (11) | 0.64 | 0.64 | 0.73 | 0.27 | bk-05 | bk-04 | bk-04, bk-05, bk-09, bk-10, bk-11 |
| 200 | engineering (27) | 1.00 | 0.96 | 1.00 | 0.96 | k8s-27 | none | k8s-27 |
| 200 | engineering · multi_hop (3) | 1.00 | 0.67 | 1.00 | 0.67 | k8s-27 | none | k8s-27 |
| 200 | engineering · single (24) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 200 | northwind (26) | 0.62 | 1.00 | 1.00 | 0.88 | none | nw-01, nw-10, nw-11, nw-14, nw-17, nw-18, nw-22, nw-23, nw-24, nw-25 | nw-17, nw-22, nw-24 |
| 200 | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | none | nw-23 | none |
| 200 | northwind · MD (23) | 0.61 | 1.00 | 1.00 | 0.87 | none | nw-01, nw-10, nw-11, nw-14, nw-17, nw-18, nw-22, nw-24, nw-25 | nw-17, nw-22, nw-24 |
| 200 | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 200 | northwind · injection (1) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 200 | northwind · multi_hop (3) | 0.33 | 1.00 | 1.00 | 0.67 | none | nw-17, nw-18 | nw-17 |
| 200 | northwind · restricted (3) | 0.33 | 1.00 | 1.00 | 0.67 | none | nw-22, nw-23 | nw-22 |
| 200 | northwind · single (17) | 0.65 | 1.00 | 1.00 | 0.94 | none | nw-01, nw-10, nw-11, nw-14, nw-24, nw-25 | nw-24 |
| 200 | sec-10k (45) | 0.93 | 0.87 | 0.93 | 0.82 | m04, m07, n24 | none | a02, m04, m07, m25, n24 |
| 200 | sec-10k · HTM (24) | 0.96 | 0.92 | 0.96 | 0.88 | n24 | none | m25, n24 |
| 200 | sec-10k · TXT (21) | 0.90 | 0.81 | 0.90 | 0.76 | m04, m07 | none | a02, m04, m07 |
| 300 | books (13) | 0.62 | 0.77 | 0.85 | 0.62 | bk-05 | bk-08, bk-09, bk-12 | bk-04, bk-05, bk-09 |
| 300 | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | none | bk-12 | none |
| 300 | books · single (11) | 0.64 | 0.73 | 0.82 | 0.55 | bk-05 | bk-08, bk-09 | bk-04, bk-05, bk-09 |
| 300 | engineering (27) | 0.96 | 0.96 | 1.00 | 0.96 | k8s-27 | k8s-18 | k8s-27 |
| 300 | engineering · multi_hop (3) | 1.00 | 0.67 | 1.00 | 0.67 | k8s-27 | none | k8s-27 |
| 300 | engineering · single (24) | 0.96 | 1.00 | 1.00 | 1.00 | none | k8s-18 | none |
| 300 | northwind (26) | 0.69 | 1.00 | 1.00 | 0.96 | none | nw-01, nw-05, nw-15, nw-16, nw-17, nw-20, nw-21, nw-24 | nw-24 |
| 300 | northwind · HTML (3) | 0.33 | 1.00 | 1.00 | 1.00 | none | nw-20, nw-21 | none |
| 300 | northwind · MD (23) | 0.74 | 1.00 | 1.00 | 0.96 | none | nw-01, nw-05, nw-15, nw-16, nw-17, nw-24 | nw-24 |
| 300 | northwind · contradiction (2) | 0.00 | 1.00 | 1.00 | 1.00 | none | nw-15, nw-16 | none |
| 300 | northwind · injection (1) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 300 | northwind · multi_hop (3) | 0.67 | 1.00 | 1.00 | 1.00 | none | nw-17 | none |
| 300 | northwind · restricted (3) | 0.67 | 1.00 | 1.00 | 1.00 | none | nw-20 | none |
| 300 | northwind · single (17) | 0.76 | 1.00 | 1.00 | 0.94 | none | nw-01, nw-05, nw-21, nw-24 | nw-24 |
| 300 | sec-10k (45) | 0.93 | 0.89 | 0.96 | 0.89 | a01, m07, m25 | m02 | a01, m02, m25 |
| 300 | sec-10k · HTM (24) | 0.96 | 0.92 | 0.96 | 0.92 | m25 | none | m25 |
| 300 | sec-10k · TXT (21) | 0.90 | 0.86 | 0.95 | 0.86 | a01, m07 | m02 | a01, m02 |
| 400 | books (13) | 0.69 | 0.77 | 0.85 | 0.62 | bk-05 | bk-08, bk-12 | bk-05, bk-08, bk-09 |
| 400 | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | none | bk-12 | none |
| 400 | books · single (11) | 0.73 | 0.73 | 0.82 | 0.55 | bk-05 | bk-08 | bk-05, bk-08, bk-09 |
| 400 | engineering (27) | 1.00 | 0.96 | 1.00 | 0.96 | k8s-27 | none | k8s-27 |
| 400 | engineering · multi_hop (3) | 1.00 | 0.67 | 1.00 | 0.67 | k8s-27 | none | k8s-27 |
| 400 | engineering · single (24) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 400 | northwind (26) | 0.96 | 1.00 | 1.00 | 1.00 | none | nw-16 | none |
| 400 | northwind · HTML (3) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 400 | northwind · MD (23) | 0.96 | 1.00 | 1.00 | 1.00 | none | nw-16 | none |
| 400 | northwind · contradiction (2) | 0.50 | 1.00 | 1.00 | 1.00 | none | nw-16 | none |
| 400 | northwind · injection (1) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 400 | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 400 | northwind · restricted (3) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 400 | northwind · single (17) | 1.00 | 1.00 | 1.00 | 1.00 | none | none | none |
| 400 | sec-10k (45) | 0.84 | 0.89 | 0.93 | 0.80 | a27, n24 | a21, m01, m02, m28 | a27, m01, m02, m28, n02, n24 |
| 400 | sec-10k · HTM (24) | 0.88 | 0.88 | 0.96 | 0.83 | a27, n24 | a21, m28 | a27, m28, n24 |
| 400 | sec-10k · TXT (21) | 0.81 | 0.90 | 0.90 | 0.76 | none | m01, m02 | m01, m02, n02 |

## Questions not found within top 6

A rank of None means the passage was not in the top 20.

| Chunk tokens | Mode | Question | Rank |
|---|---|---|---|
| 200 | reranked | a01 | None |
| 200 | reranked | a02 | 16 |
| 200 | reranked | n04 | None |
| 200 | reranked | m25 | 18 |
| 200 | reranked | m28 | None |
| 200 | reranked | n23 | 7 |
| 200 | reranked | n24 | 7 |
| 200 | reranked | bk-03 | None |
| 200 | reranked | bk-04 | 7 |
| 200 | reranked | bk-06 | None |
| 200 | reranked | bk-08 | None |
| 200 | reranked | bk-09 | 7 |
| 200 | hybrid | a01 | None |
| 200 | hybrid | a02 | 9 |
| 200 | hybrid | m04 | 7 |
| 200 | hybrid | m07 | 8 |
| 200 | hybrid | n04 | None |
| 200 | hybrid | m25 | 14 |
| 200 | hybrid | m28 | None |
| 200 | hybrid | n24 | 8 |
| 200 | hybrid | k8s-27 | 12 |
| 200 | hybrid | nw-17 | 7 |
| 200 | hybrid | nw-22 | 10 |
| 200 | hybrid | nw-24 | 11 |
| 200 | hybrid | bk-03 | None |
| 200 | hybrid | bk-04 | 10 |
| 200 | hybrid | bk-05 | 14 |
| 200 | hybrid | bk-06 | None |
| 200 | hybrid | bk-08 | None |
| 200 | hybrid | bk-09 | 9 |
| 200 | hybrid | bk-10 | 7 |
| 200 | hybrid | bk-11 | 10 |
| 200 | dense | a01 | None |
| 200 | dense | n01 | 18 |
| 200 | dense | n04 | None |
| 200 | dense | a23 | 12 |
| 200 | dense | m28 | None |
| 200 | dense | n23 | 7 |
| 200 | dense | k8s-01 | 7 |
| 200 | dense | nw-01 | None |
| 200 | dense | nw-10 | None |
| 200 | dense | nw-11 | None |
| 200 | dense | nw-14 | None |
| 200 | dense | nw-17 | None |
| 200 | dense | nw-18 | None |
| 200 | dense | nw-22 | None |
| 200 | dense | nw-23 | None |
| 200 | dense | nw-24 | None |
| 200 | dense | nw-25 | None |
| 200 | dense | bk-02 | 7 |
| 200 | dense | bk-03 | None |
| 200 | dense | bk-04 | None |
| 200 | dense | bk-05 | 7 |
| 200 | dense | bk-06 | None |
| 200 | dense | bk-08 | None |
| 200 | dense | bk-09 | 15 |
| 200 | dense | bk-10 | 8 |
| 200 | dense | bk-12 | None |
| 200 | sparse | a01 | None |
| 200 | sparse | a02 | 19 |
| 200 | sparse | m01 | 9 |
| 200 | sparse | m04 | None |
| 200 | sparse | m07 | None |
| 200 | sparse | n02 | 8 |
| 200 | sparse | n04 | None |
| 200 | sparse | a22 | 7 |
| 200 | sparse | a23 | 8 |
| 200 | sparse | a27 | 11 |
| 200 | sparse | m25 | 19 |
| 200 | sparse | m28 | None |
| 200 | sparse | n23 | 9 |
| 200 | sparse | n24 | None |
| 200 | sparse | k8s-17 | 10 |
| 200 | sparse | k8s-23 | 9 |
| 200 | sparse | k8s-27 | None |
| 200 | sparse | bk-03 | None |
| 200 | sparse | bk-05 | None |
| 200 | sparse | bk-06 | None |
| 200 | sparse | bk-08 | None |
| 200 | sparse | bk-09 | 9 |
| 200 | sparse | bk-10 | 13 |
| 200 | sparse | bk-13 | 8 |
| 300 | reranked | a01 | None |
| 300 | reranked | a02 | 8 |
| 300 | reranked | n04 | None |
| 300 | reranked | m21 | 8 |
| 300 | reranked | m25 | 16 |
| 300 | reranked | m28 | None |
| 300 | reranked | n24 | 7 |
| 300 | reranked | k8s-01 | 8 |
| 300 | reranked | bk-03 | None |
| 300 | reranked | bk-04 | None |
| 300 | reranked | bk-05 | None |
| 300 | reranked | bk-06 | None |
| 300 | reranked | bk-09 | None |
| 300 | hybrid | a01 | None |
| 300 | hybrid | m02 | 8 |
| 300 | hybrid | n04 | None |
| 300 | hybrid | m25 | 16 |
| 300 | hybrid | m28 | None |
| 300 | hybrid | k8s-27 | 13 |
| 300 | hybrid | nw-24 | 12 |
| 300 | hybrid | bk-03 | None |
| 300 | hybrid | bk-04 | None |
| 300 | hybrid | bk-05 | None |
| 300 | hybrid | bk-06 | None |
| 300 | hybrid | bk-09 | None |
| 300 | dense | a01 | 17 |
| 300 | dense | m02 | None |
| 300 | dense | m04 | 14 |
| 300 | dense | m06 | 7 |
| 300 | dense | n01 | 7 |
| 300 | dense | n04 | None |
| 300 | dense | m25 | 8 |
| 300 | dense | m28 | None |
| 300 | dense | n24 | 7 |
| 300 | dense | k8s-18 | None |
| 300 | dense | nw-01 | None |
| 300 | dense | nw-05 | None |
| 300 | dense | nw-15 | None |
| 300 | dense | nw-16 | None |
| 300 | dense | nw-17 | None |
| 300 | dense | nw-20 | None |
| 300 | dense | nw-21 | None |
| 300 | dense | nw-24 | None |
| 300 | dense | bk-02 | 7 |
| 300 | dense | bk-03 | None |
| 300 | dense | bk-04 | 18 |
| 300 | dense | bk-05 | 11 |
| 300 | dense | bk-06 | None |
| 300 | dense | bk-08 | None |
| 300 | dense | bk-09 | None |
| 300 | dense | bk-12 | None |
| 300 | sparse | a01 | None |
| 300 | sparse | a07 | 7 |
| 300 | sparse | m04 | 14 |
| 300 | sparse | m07 | None |
| 300 | sparse | n02 | 8 |
| 300 | sparse | n04 | None |
| 300 | sparse | n06 | 8 |
| 300 | sparse | m25 | None |
| 300 | sparse | m28 | None |
| 300 | sparse | n24 | 17 |
| 300 | sparse | k8s-17 | 8 |
| 300 | sparse | k8s-27 | None |
| 300 | sparse | bk-03 | None |
| 300 | sparse | bk-04 | 15 |
| 300 | sparse | bk-05 | None |
| 300 | sparse | bk-06 | None |
| 300 | sparse | bk-09 | 11 |
| 400 | reranked | a01 | None |
| 400 | reranked | n04 | None |
| 400 | reranked | m21 | 10 |
| 400 | reranked | m25 | None |
| 400 | reranked | m28 | None |
| 400 | reranked | n24 | 7 |
| 400 | reranked | k8s-01 | 7 |
| 400 | reranked | bk-03 | None |
| 400 | reranked | bk-04 | 7 |
| 400 | reranked | bk-05 | None |
| 400 | reranked | bk-06 | None |
| 400 | reranked | bk-08 | None |
| 400 | reranked | bk-09 | 7 |
| 400 | hybrid | a01 | None |
| 400 | hybrid | m01 | 14 |
| 400 | hybrid | m02 | 9 |
| 400 | hybrid | n02 | 7 |
| 400 | hybrid | n04 | None |
| 400 | hybrid | a27 | 7 |
| 400 | hybrid | m25 | None |
| 400 | hybrid | m28 | None |
| 400 | hybrid | n24 | 15 |
| 400 | hybrid | k8s-27 | 14 |
| 400 | hybrid | bk-03 | None |
| 400 | hybrid | bk-05 | None |
| 400 | hybrid | bk-06 | None |
| 400 | hybrid | bk-08 | None |
| 400 | hybrid | bk-09 | 10 |
| 400 | dense | a01 | None |
| 400 | dense | m01 | None |
| 400 | dense | m02 | None |
| 400 | dense | m06 | 10 |
| 400 | dense | n01 | 15 |
| 400 | dense | n04 | None |
| 400 | dense | a21 | None |
| 400 | dense | a25 | 8 |
| 400 | dense | m25 | None |
| 400 | dense | m28 | None |
| 400 | dense | n23 | 7 |
| 400 | dense | n24 | 9 |
| 400 | dense | nw-16 | None |
| 400 | dense | bk-03 | None |
| 400 | dense | bk-04 | 15 |
| 400 | dense | bk-05 | 13 |
| 400 | dense | bk-06 | None |
| 400 | dense | bk-08 | None |
| 400 | dense | bk-09 | 13 |
| 400 | dense | bk-12 | None |
| 400 | sparse | a01 | None |
| 400 | sparse | a07 | 7 |
| 400 | sparse | m01 | 8 |
| 400 | sparse | m04 | 10 |
| 400 | sparse | m07 | 11 |
| 400 | sparse | n04 | None |
| 400 | sparse | n06 | 10 |
| 400 | sparse | a25 | 17 |
| 400 | sparse | a27 | None |
| 400 | sparse | m21 | 9 |
| 400 | sparse | m25 | None |
| 400 | sparse | m28 | 13 |
| 400 | sparse | n24 | None |
| 400 | sparse | k8s-17 | 10 |
| 400 | sparse | k8s-20 | 8 |
| 400 | sparse | k8s-27 | None |
| 400 | sparse | bk-03 | None |
| 400 | sparse | bk-04 | 16 |
| 400 | sparse | bk-05 | None |
| 400 | sparse | bk-06 | None |
| 400 | sparse | bk-08 | 19 |
| 400 | sparse | bk-09 | 7 |

## BM25 and Postgres ts_rank

The same sparse search with Postgres full-text search in place of bm25s, for comparison only (D17): `ts_rank` has no inverse document frequency. Compare with the *sparse* rows above.

| Chunk tokens | Chunks | Mode | Questions | S@1 | S@3 | S@6 | S@8 | S@10 | S@20 | Recall@6 | Recall@20 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 9379 | sparse | books (13) | 0.23 | 0.31 | 0.46 | 0.54 | 0.62 | 0.69 | 0.38 | 0.65 | 0.305 |
| 200 | 9379 | sparse | books · multi_hop (2) | 0.50 | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | 0.562 |
| 200 | 9379 | sparse | books · single (11) | 0.18 | 0.27 | 0.45 | 0.45 | 0.55 | 0.64 | 0.41 | 0.59 | 0.259 |
| 200 | 9379 | sparse | engineering (27) | 0.63 | 0.78 | 0.89 | 0.89 | 0.96 | 0.96 | 0.87 | 0.94 | 0.715 |
| 200 | 9379 | sparse | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.50 | 0.50 | 0.667 |
| 200 | 9379 | sparse | engineering · single (24) | 0.62 | 0.79 | 0.92 | 0.92 | 1.00 | 1.00 | 0.92 | 1.00 | 0.721 |
| 200 | 9379 | sparse | northwind (26) | 0.69 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.810 |
| 200 | 9379 | sparse | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 200 | 9379 | sparse | northwind · MD (23) | 0.70 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.807 |
| 200 | 9379 | sparse | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | sparse | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.333 |
| 200 | 9379 | sparse | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | sparse | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.567 |
| 200 | 9379 | sparse | northwind · single (17) | 0.71 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.825 |
| 200 | 9379 | sparse | sec-10k (45) | 0.36 | 0.62 | 0.69 | 0.76 | 0.80 | 0.87 | 0.69 | 0.87 | 0.503 |
| 200 | 9379 | sparse | sec-10k · HTM (24) | 0.38 | 0.62 | 0.71 | 0.79 | 0.83 | 0.92 | 0.71 | 0.92 | 0.523 |
| 200 | 9379 | sparse | sec-10k · TXT (21) | 0.33 | 0.62 | 0.67 | 0.71 | 0.76 | 0.81 | 0.67 | 0.81 | 0.479 |
| 300 | 6101 | sparse | books (13) | 0.23 | 0.54 | 0.62 | 0.62 | 0.62 | 0.77 | 0.50 | 0.77 | 0.374 |
| 300 | 6101 | sparse | books · multi_hop (2) | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.600 |
| 300 | 6101 | sparse | books · single (11) | 0.18 | 0.55 | 0.55 | 0.55 | 0.55 | 0.73 | 0.50 | 0.73 | 0.333 |
| 300 | 6101 | sparse | engineering (27) | 0.63 | 0.81 | 0.93 | 0.96 | 0.96 | 0.96 | 0.91 | 0.94 | 0.736 |
| 300 | 6101 | sparse | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.667 |
| 300 | 6101 | sparse | engineering · single (24) | 0.62 | 0.83 | 0.96 | 1.00 | 1.00 | 1.00 | 0.94 | 0.98 | 0.745 |
| 300 | 6101 | sparse | northwind (26) | 0.69 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.817 |
| 300 | 6101 | sparse | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 300 | 6101 | sparse | northwind · MD (23) | 0.70 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.814 |
| 300 | 6101 | sparse | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | sparse | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | sparse | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | sparse | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.567 |
| 300 | 6101 | sparse | northwind · single (17) | 0.71 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.825 |
| 300 | 6101 | sparse | sec-10k (45) | 0.31 | 0.58 | 0.78 | 0.84 | 0.84 | 0.89 | 0.78 | 0.89 | 0.487 |
| 300 | 6101 | sparse | sec-10k · HTM (24) | 0.25 | 0.62 | 0.88 | 0.88 | 0.88 | 0.92 | 0.88 | 0.92 | 0.480 |
| 300 | 6101 | sparse | sec-10k · TXT (21) | 0.38 | 0.52 | 0.67 | 0.81 | 0.81 | 0.86 | 0.67 | 0.86 | 0.494 |
| 400 | 4800 | sparse | books (13) | 0.31 | 0.54 | 0.54 | 0.62 | 0.62 | 0.77 | 0.50 | 0.77 | 0.434 |
| 400 | 4800 | sparse | books · multi_hop (2) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0.750 |
| 400 | 4800 | sparse | books · single (11) | 0.27 | 0.45 | 0.45 | 0.55 | 0.55 | 0.73 | 0.45 | 0.73 | 0.377 |
| 400 | 4800 | sparse | engineering (27) | 0.56 | 0.85 | 0.89 | 0.93 | 0.96 | 0.96 | 0.87 | 0.96 | 0.703 |
| 400 | 4800 | sparse | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.667 |
| 400 | 4800 | sparse | engineering · single (24) | 0.54 | 0.88 | 0.92 | 0.96 | 1.00 | 1.00 | 0.90 | 1.00 | 0.707 |
| 400 | 4800 | sparse | northwind (26) | 0.69 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.817 |
| 400 | 4800 | sparse | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 400 | 4800 | sparse | northwind · MD (23) | 0.70 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.814 |
| 400 | 4800 | sparse | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | sparse | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 400 | 4800 | sparse | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | sparse | northwind · restricted (3) | 0.33 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.567 |
| 400 | 4800 | sparse | northwind · single (17) | 0.71 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.825 |
| 400 | 4800 | sparse | sec-10k (45) | 0.36 | 0.67 | 0.71 | 0.76 | 0.82 | 0.89 | 0.71 | 0.89 | 0.512 |
| 400 | 4800 | sparse | sec-10k · HTM (24) | 0.38 | 0.75 | 0.75 | 0.75 | 0.79 | 0.88 | 0.75 | 0.88 | 0.539 |
| 400 | 4800 | sparse | sec-10k · TXT (21) | 0.33 | 0.57 | 0.67 | 0.76 | 0.86 | 0.90 | 0.67 | 0.90 | 0.480 |
| 200 | 9379 | ts_rank | books (13) | 0.15 | 0.23 | 0.46 | 0.54 | 0.62 | 0.69 | 0.46 | 0.65 | 0.251 |
| 200 | 9379 | ts_rank | books · multi_hop (2) | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 | 0.556 |
| 200 | 9379 | ts_rank | books · single (11) | 0.09 | 0.18 | 0.45 | 0.55 | 0.55 | 0.64 | 0.45 | 0.59 | 0.196 |
| 200 | 9379 | ts_rank | engineering (27) | 0.59 | 0.78 | 0.89 | 0.93 | 0.93 | 0.96 | 0.87 | 0.94 | 0.694 |
| 200 | 9379 | ts_rank | engineering · multi_hop (3) | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.50 | 0.50 | 0.500 |
| 200 | 9379 | ts_rank | engineering · single (24) | 0.62 | 0.79 | 0.92 | 0.96 | 0.96 | 1.00 | 0.92 | 1.00 | 0.718 |
| 200 | 9379 | ts_rank | northwind (26) | 0.62 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.783 |
| 200 | 9379 | ts_rank | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 200 | 9379 | ts_rank | northwind · MD (23) | 0.61 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.777 |
| 200 | 9379 | ts_rank | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | ts_rank | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 200 | 9379 | ts_rank | northwind · multi_hop (3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 200 | 9379 | ts_rank | northwind · restricted (3) | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.667 |
| 200 | 9379 | ts_rank | northwind · single (17) | 0.59 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.757 |
| 200 | 9379 | ts_rank | sec-10k (45) | 0.47 | 0.64 | 0.73 | 0.78 | 0.78 | 0.91 | 0.73 | 0.91 | 0.580 |
| 200 | 9379 | ts_rank | sec-10k · HTM (24) | 0.62 | 0.67 | 0.75 | 0.83 | 0.83 | 0.96 | 0.75 | 0.96 | 0.679 |
| 200 | 9379 | ts_rank | sec-10k · TXT (21) | 0.29 | 0.62 | 0.71 | 0.71 | 0.71 | 0.86 | 0.71 | 0.86 | 0.468 |
| 300 | 6101 | ts_rank | books (13) | 0.15 | 0.46 | 0.69 | 0.77 | 0.77 | 0.77 | 0.62 | 0.77 | 0.356 |
| 300 | 6101 | ts_rank | books · multi_hop (2) | 0.50 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0.625 |
| 300 | 6101 | ts_rank | books · single (11) | 0.09 | 0.45 | 0.64 | 0.73 | 0.73 | 0.73 | 0.59 | 0.73 | 0.307 |
| 300 | 6101 | ts_rank | engineering (27) | 0.52 | 0.81 | 0.93 | 0.96 | 0.96 | 0.96 | 0.91 | 0.96 | 0.669 |
| 300 | 6101 | ts_rank | engineering · multi_hop (3) | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.500 |
| 300 | 6101 | ts_rank | engineering · single (24) | 0.54 | 0.83 | 0.96 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 0.691 |
| 300 | 6101 | ts_rank | northwind (26) | 0.58 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.761 |
| 300 | 6101 | ts_rank | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 300 | 6101 | ts_rank | northwind · MD (23) | 0.57 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.751 |
| 300 | 6101 | ts_rank | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 300 | 6101 | ts_rank | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 300 | 6101 | ts_rank | northwind · multi_hop (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 1.00 | 0.833 |
| 300 | 6101 | ts_rank | northwind · restricted (3) | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.667 |
| 300 | 6101 | ts_rank | northwind · single (17) | 0.59 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.752 |
| 300 | 6101 | ts_rank | sec-10k (45) | 0.38 | 0.64 | 0.84 | 0.89 | 0.89 | 0.98 | 0.84 | 0.98 | 0.544 |
| 300 | 6101 | ts_rank | sec-10k · HTM (24) | 0.42 | 0.67 | 0.88 | 0.92 | 0.92 | 1.00 | 0.88 | 1.00 | 0.580 |
| 300 | 6101 | ts_rank | sec-10k · TXT (21) | 0.33 | 0.62 | 0.81 | 0.86 | 0.86 | 0.95 | 0.81 | 0.95 | 0.504 |
| 400 | 4800 | ts_rank | books (13) | 0.31 | 0.54 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.467 |
| 400 | 4800 | ts_rank | books · multi_hop (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | ts_rank | books · single (11) | 0.18 | 0.45 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 | 0.370 |
| 400 | 4800 | ts_rank | engineering (27) | 0.56 | 0.78 | 0.89 | 0.93 | 0.96 | 0.96 | 0.87 | 0.96 | 0.680 |
| 400 | 4800 | ts_rank | engineering · multi_hop (3) | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.667 |
| 400 | 4800 | ts_rank | engineering · single (24) | 0.54 | 0.79 | 0.92 | 0.96 | 1.00 | 1.00 | 0.90 | 1.00 | 0.682 |
| 400 | 4800 | ts_rank | northwind (26) | 0.54 | 0.92 | 0.96 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 0.729 |
| 400 | 4800 | ts_rank | northwind · HTML (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.833 |
| 400 | 4800 | ts_rank | northwind · MD (23) | 0.52 | 0.91 | 0.96 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.716 |
| 400 | 4800 | ts_rank | northwind · contradiction (2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |
| 400 | 4800 | ts_rank | northwind · injection (1) | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.500 |
| 400 | 4800 | ts_rank | northwind · multi_hop (3) | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.833 |
| 400 | 4800 | ts_rank | northwind · restricted (3) | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.667 |
| 400 | 4800 | ts_rank | northwind · single (17) | 0.53 | 0.88 | 0.94 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 0.703 |
| 400 | 4800 | ts_rank | sec-10k (45) | 0.27 | 0.69 | 0.80 | 0.82 | 0.89 | 0.96 | 0.80 | 0.96 | 0.502 |
| 400 | 4800 | ts_rank | sec-10k · HTM (24) | 0.42 | 0.75 | 0.79 | 0.83 | 0.92 | 0.96 | 0.79 | 0.96 | 0.602 |
| 400 | 4800 | ts_rank | sec-10k · TXT (21) | 0.10 | 0.62 | 0.81 | 0.81 | 0.86 | 0.95 | 0.81 | 0.95 | 0.389 |

## Search latency

Wall time of one search on this machine, embedding the question included. One run, so only differences of tens of milliseconds mean anything.

| Chunk tokens | Mode | Median ms | Max ms |
|---|---|---|---|
| 200 | reranked | 339 | 680 |
| 200 | hybrid | 15 | 100 |
| 200 | dense | 10 | 26 |
| 200 | sparse | 2 | 28 |
| 300 | reranked | 463 | 1225 |
| 300 | hybrid | 13 | 36 |
| 300 | dense | 10 | 97 |
| 300 | sparse | 2 | 8 |
| 400 | reranked | 617 | 1356 |
| 400 | hybrid | 10 | 26 |
| 400 | dense | 7 | 12 |
| 400 | sparse | 2 | 105 |

## Relevance gate (MIN_RELEVANCE)

### 200 tokens

Best dense similarity: answerable min 0.418, median 0.779; off-topic max 0.576; in-domain unanswerable max 0.719.

| Threshold | Answerable kept | Off-topic blocked | In-domain unanswerable blocked |
|---|---|---|---|
| 0.50 | 110/111 | 2/6 | 0/9 |
| 0.55 | 107/111 | 5/6 | 0/9 |
| 0.60 | 103/111 | 6/6 | 0/9 |
| 0.65 | 99/111 | 6/6 | 0/9 |
| 0.70 | 91/111 | 6/6 | 5/9 |
| 0.75 | 76/111 | 6/6 | 9/9 |

### 300 tokens

Best dense similarity: answerable min 0.435, median 0.772; off-topic max 0.567; in-domain unanswerable max 0.719.

| Threshold | Answerable kept | Off-topic blocked | In-domain unanswerable blocked |
|---|---|---|---|
| 0.50 | 110/111 | 2/6 | 0/9 |
| 0.55 | 108/111 | 5/6 | 0/9 |
| 0.60 | 102/111 | 6/6 | 0/9 |
| 0.65 | 97/111 | 6/6 | 1/9 |
| 0.70 | 86/111 | 6/6 | 7/9 |
| 0.75 | 70/111 | 6/6 | 9/9 |

### 400 tokens

Best dense similarity: answerable min 0.554, median 0.772; off-topic max 0.574; in-domain unanswerable max 0.716.

| Threshold | Answerable kept | Off-topic blocked | In-domain unanswerable blocked |
|---|---|---|---|
| 0.50 | 111/111 | 2/6 | 0/9 |
| 0.55 | 111/111 | 5/6 | 0/9 |
| 0.60 | 107/111 | 6/6 | 0/9 |
| 0.65 | 103/111 | 6/6 | 0/9 |
| 0.70 | 95/111 | 6/6 | 6/9 |
| 0.75 | 73/111 | 6/6 | 9/9 |
