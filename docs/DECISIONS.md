# Decisions

Every decision that shapes groundwork, with the reasoning behind it. New decisions continue from D28 and
are added here when they are made; `docs/HANDOFF.md` section 3 holds the same D15–D26 table.

## D1–D14: from NovaBridge

From `docs/archive/HANDOFF-novabridge.md` section 3, unchanged. Later decisions replace some of them:
D16 replaces D6 (Chroma) and the Chroma half of D5; D17 keeps bm25s from D5, derived from Postgres;
D21 replaces the in-memory jobs of D10; D22 makes the sync interfaces of D2 async; D27 amends D9's
"can be committed": the corpora live in a versioned GitHub Release, not in git.

| # | Decision | Why |
|---|---|---|
| D1 | Plain Python core, **no LlamaIndex/LangChain** | Every step (chunking, fusion, prompt, citation check) is visible, testable and explainable. LlamaIndex would duplicate our own LLM adapter layer and hide the citation logic in its templates. Trade-off: more code; the brief only *suggests* frameworks. |
| D2 | **LLM behind an adapter interface**, provider chosen by env var | `LLM_PROVIDER=openai_compatible` (Groq, Ollama, OpenAI by changing `base_url`) or `anthropic`. Shows clean ports-and-adapters design, keeps the system runnable for free (Groq/Ollama), and lets tests use a `FakeLLM`. |
| D3 | **Embedder is a separate interface** from the LLM | Anthropic has no embeddings API, so the embedding provider can't follow the LLM provider. |
| D4 | Embeddings: **`BAAI/bge-small-en-v1.5` via `fastembed`** to start; selected by `EMBEDDING_PROVIDER` | Small ONNX model, fast on CPU, runs locally and free, so the system works end to end first. Limit: 512 input tokens, so `CHUNK_TOKENS` must stay below it. `sentence_transformers` remains a provider for **Qwen3-Embedding-0.6B** (Apache 2.0) and `microsoft/harrier-oss-v1-0.6b` (MIT). The final choice comes from **our own recall@k** in a later comparison, not the leaderboard. |
| D5 | **Hybrid retrieval**: BM25 (`bm25s`) + dense (Chroma), merged with **Reciprocal Rank Fusion** written by hand | Financial text is full of exact terms (tickers, "Item 1A", product names, numbers) where BM25 wins, and paraphrased questions where dense wins. RRF needs no score calibration. `score(d) = Σ 1/(k + rank_i(d))`, k = 60 (Cormack et al., 2009). About 15 lines; unit-tested. |
| D6 | **Vector store: ChromaDB** (persistent, embedded) | Zero ops, stores text + metadata, good enough for thousands of chunks. Chroma is the **single source of truth** for chunks; the BM25 index is derived from it and can be rebuilt. |
| D7 | **Citations checked on the server** | Passages are numbered `[1]..[k]` in the prompt; the model must cite `[n]`. The server parses citations, drops numbers that don't exist, renumbers the rest in order of first appearance, and returns only cited passages as `sources`. Zero valid citations → treated as insufficient evidence. |
| D8 | **"Not enough evidence" path** has two layers | (a) Retrieval gate: if the best dense cosine similarity < `MIN_RELEVANCE` (tuned on the eval set), the LLM is not called. (b) The model is told to reply exactly `INSUFFICIENT_EVIDENCE` if the passages don't answer the question. Either case returns `insufficient_evidence: true`. |
| D9 | **Corpus: SEC EDGAR 10-K filings** for Apple, Microsoft and NVIDIA (latest annual report each) | Real, relevant, and US government filings are public domain, so they can be committed. EDGAR requires a descriptive `User-Agent` with contact info: read it from `SEC_USER_AGENT` in `.env`, never hard-code it. **First runs use their newest plain-text annual reports** (FY2000–2001, the last before EDGAR moved to HTML): real `.txt` filings needing no conversion. Since M8b part 1 the current HTML 10-Ks are indexed **beside** them, not instead: both generations stay, and answers name the period of each figure (see `backend/eval/decisions.md`). |
| D10 | **Document upload** from the UI | `POST /api/documents` (multipart). Indexing runs as a background job; the UI polls job status. Formats arrive gradually: **TXT and Markdown first** (M2, M8), then HTML, then PDF (text layer only, **no OCR**) in M8b. Max 20 MB. |
| D11 | **Frontend: Vite + React + TypeScript, shadcn/ui, TanStack Query** | shadcn/ui components are copied into the repo, so they stay readable and restylable to match `docs/design/` through the design tokens. TanStack Query handles request state and job polling, replacing hand-written loading/error/polling code. Trade-off: Tailwind and Radix become dependencies. |
| D12 | **Python 3.12, uv, ruff, mypy, pytest** | Modern standard tooling; `uv.lock` makes installs reproducible. |
| D13 | **Docker Compose**: `api` (FastAPI/uvicorn) + `web` (nginx serving the Vite build and proxying `/api`) | One command to run. Optional `ollama` service behind a compose profile for a fully local LLM. |
| D14 | **Cross-encoder re-ranker** (`jinaai/jina-reranker-v1-tiny-en` via `fastembed`) on the top `RERANK_CANDIDATES` (20) after RRF; `RERANKER_MODEL=` turns it off | RRF ranks a passage found by both retrievers above any found by one, which dropped most misses. Chosen from six fastembed cross-encoders on recall, CPU latency and license (`backend/eval/decisions.md`, "Re-ranker"). The relevance gate still uses dense similarity, over all candidates before re-ranking. |

## D15–D26: groundwork

From `docs/HANDOFF.md` section 3.

| # | Decision | Why |
|---|---|---|
| D15 | **Continue in this repo; rename the product and package to `groundwork`** (`novabridge_qa` → `groundwork`). The version EPAM saw is tagged **`v1-submission`** (`12e8ea7`, the last `main` before M0). The `~/repos/groundwork` scaffold is reference only: M0 copies the files still referenced to `docs/archive/groundwork-scaffold/` and deletes the scaffold; its `DECISIONS.md` is ported to `docs/DECISIONS.md`, and its migration is read for ideas but not copied (section 5). The GitHub repo and the local folder are renamed to `groundwork`. | This repo has the history, the measured evals and 361 tests; the scaffold has one commit. The tag keeps the version EPAM saw. |
| D16 | **PostgreSQL 16 + pgvector is the single source of truth** for documents, chunks, vectors, jobs, conversations, tickets and checkpoints. **Chroma is removed** once M2 shows eval parity. | ACL, transactions, a durable queue and several workers all need a real database. Keeping Chroma would mean writing every ACL filter twice. |
| D17 | **bm25s stays, derived from Postgres**, one index per chunk index (D20), persisted on a shared volume and reloaded by each API worker when the index's `generation` changes. A Postgres full-text (`ts_rank`) retriever is added only as a measured comparison. | `ts_rank` has no IDF, so rare exact tokens ("Item 7A", figures, error codes) lose the weight that makes hybrid search work. BM25 found 0.90 of the TXT questions in its top 20 against dense's 0.81. |
| D18 | **Plain SQL through psycopg 3** (async pool) in the app; **Alembic with hand-written SQL** (`op.execute`) for migrations. No ORM. | Every index, cast and filter is visible and explainable. Same choice as the scaffold's DECISIONS #4 (`docs/archive/groundwork-scaffold/DECISIONS.md`). |
| D19 | **Vectors in an untyped `vector` column, with one partial HNSW expression index per chunk index**: `USING hnsw ((embedding::vector(384)) vector_cosine_ops) WHERE index_id = 7`. Queries repeat the same cast and predicate, with **`index_id` and the dimension written into the SQL as literals** (both are integers from the `indexes` table, never user input), not passed as parameters. | Different embedders have different dimensions, and HNSW needs a fixed one. This keeps one `chunks` table for every experiment. Literals matter: psycopg 3 prepares a statement server-side after 5 runs, Postgres may then choose a generic plan, and a generic plan can't prove `index_id = $1` matches the partial index's predicate, so the query silently falls back to a sequential scan. |
| D20 | **Chunk indexes are first-class:** an `indexes` table records chunker, contextualizer, embedder and their parameters; chunks belong to one index; one index is marked active (an alias). Experiments build new indexes beside the active one. | Chunking and embedder experiments become side-by-side comparisons, and switching models becomes the blue/green index swap from interview answer Q17. |
| D21 | **Job queue in Postgres** (`SELECT … FOR UPDATE SKIP LOCKED`), consumed by a `worker` service. No Redis, no Celery. | One less service; transactional with the data it changes; a classic pattern worth knowing. Revisit if throughput demands it. |
| D22 | **Async from the API to the store and the LLM, starting in M2.** In M2 endpoints become `async def`, `QAService` and the `ChunkStore` become async over the psycopg async pool, and CPU work (embedding, re-ranking, BM25) runs through `asyncio.to_thread`. The sync LLM client is called through `asyncio.to_thread` until M4 replaces it with async adapters. | Streaming, agent tool calls in parallel and SSE all want async. Doing it with the store in M2 means the async pool is never called from sync code. CPU work stays off the event loop (ONNX Runtime releases the GIL). |
| D23 | **Authentication is a header (`X-User`)** against users and groups seeded from `conf/users.yaml`, behind an `Authenticator` interface. Real auth (OIDC) is out of scope and the README says so. | Lets the ACL be built and tested properly without building an identity provider. |
| D24 | **Streaming (resolves improvement 6):** stream raw tokens as a preview, then send a final `answer` event with the validated, renumbered answer that replaces the preview. | Server-side citation checking (D7) needs the whole answer. Sentence buffering is the alternative; Ivan can switch later. |
| D25 | **Primary LLM: Anthropic** (prompt caching makes contextual chunking cheap); **fallback: OpenAI-compatible** (Groq); **local:** Ollama profile. Small and large model per task, set in `conf/models.yaml`. | Groq's free tier can't carry contextual chunking or agent runs. |
| D26 | **Faithfulness is our own claim-level judge** (a different model from the answer model), calibrated against hand labels. RAGAS runs as an optional comparison script only. | Keeps LangChain out of the main dependencies (rule 4), and a judge we wrote is one we can explain and calibrate. |

## D27 on: decided after the handoff

| # | Decision | Why |
|---|---|---|
| D27 | **Corpora live outside git, in a versioned GitHub Release** (amends D9's "can be committed"). All four collections are packed together by `backend/scripts/build_corpora.py` into one reproducible tarball, `corpora-vN.tar.gz`, the only asset of the Release `corpora-vN` on this repository. Each collection's manifest travels inside the tarball, beside the files it lists. The repository keeps only the lock, `backend/conf/corpora.lock.json` (version, URL, sha256). The ingest and eval scripts download, verify and install the locked version into `backend/data/raw/`, which git ignores, and record in `data/raw/.corpora-version.json` the lock and every installed file's sha256; they make no request when the lock and the files still match. A missing lock is an error. **They never replace or delete local changes:** `data/raw/` is replaced only when it is missing, empty or exactly what its record says (so its Release restores it); changed, missing or added files stop the run with a list, whatever the lock. A `data/raw/` without a record is adopted if it matches the locked version byte for byte and refused otherwise. `GROUNDWORK_CORPORA=local` skips all checks for a maintainer iterating on Northwind, and says loudly that results from that run name no corpora version. A published corpora version never changes; publishing (`gh release create`) is a manual step, never CI. **`corpora-v1` holds exactly the bytes committed on the M1 branch** that the first four-collection baseline was measured on (209 files, byte-identical). From `corpora-v2` on, `fetch_filings.py` strips the `<script>` tag EDGAR injects into its HTML filings, so v2 differs from v1 in those files and needs an eval rerun. **Accepted risk:** Northwind is fictional and has no upstream; until it moves to a separate corpora repository (#128), the Release asset is its only copy outside Ivan's machine, and its edits have no history. | Committing about 17 MB of corpora made the M1 pull request unreviewable, and every refresh would have added bytes to history for good. The lock's sha256 makes each commit name the exact corpus bytes it expects, so every eval number names its data. Manifests inside the tarball can't drift from their files. The Docker image holds code only. PR #124 is squash-merged, so the corpus files its earlier commits add (about 3.8 MB) don't enter `main`'s history; the filings and books committed before M1 (about 14.8 MB) stay in it. |

## S1–S14: the standalone scaffold

From `docs/archive/groundwork-scaffold/DECISIONS.md` (numbered 1–14 there), with what became of each.

| # | Decision | Status |
|---|---|---|
| S1 | Project is named **groundwork**. | Kept: D15. |
| S2 | Single root `pyproject.toml`; the package is the `api/` directory. | Superseded: this repo keeps `backend/src/` (D15, section 5 package layout). |
| S3 | Config as plain Pydantic models from YAML; no `pydantic-settings`. | Superseded: `pydantic-settings` stays for secrets and environment values; reviewed behaviour moves to YAML under `conf/` (handoff section 7). |
| S4 | Migrations are hand-written SQL in Alembic (`op.execute`), no autogenerate. | Kept: D18. |
| S5 | `chunks.access_groups` is denormalised from `documents`. | Kept: section 5 schema, so the ACL runs in the same query as the search. |
| S6 | `chunks.tsv` generated over `content` only. | Superseded: section 5 generates it over `context` and `text`. |
| S7 | HNSW parameters written out (`m=16, ef_construction=64`). | Kept: M2. |
| S8 | Sync SQLAlchemy engine. | Superseded: D18 (psycopg 3, no ORM) and D22 (async from M2). |
| S9 | Host ports API 8010, DB 5434, overridable. | Open: decided when the `db` service is added in M2. |
| S10 | API container runs `alembic upgrade head` before `uvicorn`. | Superseded: M2 runs migrations as a one-off `migrate` job, as the scaffold itself noted for several replicas. |
| S11 | DB tests marked `db`: skipped locally without Postgres, required in CI. | Kept: M2. |
| S12 | The LLM interface is a `typing.Protocol` that grows per milestone. | Kept: D2 and M4. |
| S13 | `web/` empty until its milestone. | Not applicable: this repo already has a frontend. |
| S14 | Python pinned to 3.12. | Kept: D12. |
