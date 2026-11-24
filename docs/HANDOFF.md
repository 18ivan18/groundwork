# HANDOFF: turning NovaBridge into groundwork

> For the Claude Code agent working in this repo. Read this file fully before writing code, then
> `CLAUDE.md`, `docs/archive/HANDOFF-novabridge.md` (the original, for the decisions still in force) and
> `backend/eval/decisions.md`.
>
> The owner is **Ivan**. The EPAM interview is over. This repo now becomes **groundwork**: a
> generic, in-depth knowledge assistant with RAG, agents, access control, evaluation and
> observability, built to learn every part properly. Nothing here is a time-boxed take-home any
> more: full scope, done well, measured. Ivan must still be able to explain every line.

---

## 1. Where we start and where we're going

### What exists (NovaBridge, `main` at `08989f4`)

A working, measured RAG system over six SEC 10-K filings:

- **Ingestion:** loader registry (TXT, Markdown, inline-XBRL HTML), `Item N.` section detection,
  400-token windows with 60 overlap on sentence boundaries, title + section header prepended for
  indexing (`domain/models.py: indexed_text`), sha256 document IDs, background upload jobs in memory.
- **Storage:** Chroma (embedded, single process) as the source of truth; bm25s rebuilt from it.
- **Retrieval:** dense top 20 + BM25 top 20 → hand-written RRF (k = 60) → `jina-reranker-v1-tiny-en`
  on the top 20 → top 6. Relevance gate on the best dense similarity (0.58).
- **Generation:** sync `LLMClient` protocol (Groq/OpenAI-compatible, Anthropic, `FakeLLM`), numbered
  passages in escaped delimiters, server-side citation validation and renumbering, three refusal
  reasons (`gate`, `model`, `no_citations`), token counts.
- **Evaluation:** 45 answerable + 8 unanswerable questions, evidence-pattern Success@k and MRR per
  stage, candidate-recall diagnosis, resumable answer eval against the real model.
- **Frontend:** Vite + React + TS, shadcn/ui, TanStack Query, zod; ask, answer with citation chips and
  source cards, document library, upload queue with job polling.
- **Quality:** strict mypy, ruff, 323 backend + 38 frontend tests, CI on every push.

### What groundwork adds

Everything the `groundwork` brief planned (`docs/archive/groundwork-scaffold/BRIEF.md`) that NovaBridge lacks, plus
the pending NovaBridge improvements, plus a chunking lab:

| Area | Added |
|---|---|
| Domain | Generic documents and collections instead of tickers and 10-K types; structure detection chosen by format |
| Corpora | Engineering docs (public), Northwind internal docs (fictional, with access groups and planted cases), public-domain books, and the existing 10-Ks |
| Storage | PostgreSQL 16 + pgvector as the source of truth; **bm25s kept as a derived index** |
| Access control | Users and groups; ACL enforced inside every retriever; uploads with owners |
| Jobs | Persisted job queue in Postgres, a worker service, several API workers |
| LLM layer | Async, streaming, tool calling, structured output, retries with backoff, provider fallback, model routing, cost accounting, prompt caching |
| Prompts | Versioned files with a hash check in CI and a prompt test suite |
| RAG | Conversations, query rewriting, SSE streaming, date-aware handling of contradicting documents, answer cache, supporting-sentence highlighting, PDF loader, table alignment fix |
| Chunking lab | `Chunker` and `Contextualizer` strategies, semantic chunking, contextual chunking (LLM-written context), an index registry so strategies sit side by side, a UI to compare them |
| Evaluation | Generic question schema, multi-hop and ACL questions, true recall for multi-evidence questions, claim-level faithfulness judge, held-out set, CI regression gate |
| Agents | Hand-written agent loop, then LangGraph with a Postgres checkpointer and human approval, a supervisor variant, memory, an MCP server |
| Operations | OpenTelemetry tracing, Prometheus metrics, cost and latency summary, rate limiting, PII redaction |
| Security | OWASP Top 10 for LLM Applications mapping, red-team tests, documented residual risk |
| UI | A routed app shell with Chat, Library, Jobs, Tickets, Lab, Observability and Settings screens, each designed in `docs/design/` before it is built |
| Delivery | Docker images built in CI; optional Terraform for AWS |

### Numbers to protect

These come from `backend/eval/results.md` and `answer_results.md`. Every milestone up to M2 must
reproduce them within noise (one question ≈ 0.04–0.05; HNSW builds differ slightly).

| What | HTML (24 q) | TXT (21 q) |
|---|---|---|
| Candidate recall at 20 (either retriever) | 0.96 | 0.90 |
| Success@6, re-ranked / hybrid / dense / BM25 | 0.92 / 0.83 / 0.75 / 0.75 | 0.81 / 0.76 / 0.71 / 0.52 |
| Answer eval (gpt-oss-120b, re-ranked) | 23 answered, 23 correct | 19 answered, 17 correct |
| Unanswerable | 8/8 refused (3 gate, 5 model) | |

---

## 2. Rules for the agent

The rules in `CLAUDE.md` and section 2 of `docs/archive/HANDOFF-novabridge.md` still apply, with these changes:

1. **One milestone at a time.** Each milestone is a branch (`gw/m3-acl`, …) and a pull request. Finish
   it, run both check commands, show Ivan, and wait for approval before the next one. Commits stay
   small and conventional.
2. **Measure, don't assume.** Any change to ingestion, retrieval or prompts reports before-and-after
   numbers from the eval scripts in its pull request. Every number in the docs comes from a script in
   the repo.
3. **Run the real flow, not only the tests.** Fakes have hidden real bugs here before (the `【1】`
   citations, the table misalignment, the dialog focus).
4. **Frameworks:** the core RAG pipeline stays plain Python: no LangChain or LlamaIndex. **Exception:**
   LangGraph (and the `langchain-core` it depends on) is allowed **only** under `backend/src/groundwork/agent/graph/`
   for the M9 comparison with the hand-written loop. RAGAS, which pulls LangChain, may only live in an
   optional dependency group used by one eval script. Update `CLAUDE.md` to say this in M0.
5. **Ask before adding any dependency** not in section 4 of this file.
6. **The ACL is enforced inside every retriever, before results are ranked and cut**, never after
   generation and never by the prompt. Every new way to reach document text (stream, cache, agent
   tool, MCP, job status, document list) gets an ACL test.
7. **No side effect without human approval.** In this system that means creating a ticket.
8. **No existence leaks.** A user who cannot see a document must not learn it exists: not through
   search, document lists, upload deduplication, job status, cache hits or error messages.
9. **Prompts are versioned files** under `backend/prompts/`. Changing one without bumping its version
   fails CI.
10. **LLM budget:** set a spend limit on every provider account. Default to the small model wherever
    the eval says quality holds. Pace real-model runs; Groq's free tier is about 8,000 tokens a minute.
11. **No invented facts** in fixtures, corpora or docs. The fictional Northwind corpus is the one place
    with invented content, and it names no real company or person.
12. Model IDs and prices are configuration, never facts in code. `.env.example` and
    `conf/models.yaml` say to check the provider's current list.

---

## 3. Decisions

Decisions D1–D14 in `docs/archive/HANDOFF-novabridge.md` stay unless replaced here. New decisions continue from D15 and
are copied into `docs/DECISIONS.md` (created in M0) as they're made.

| # | Decision | Why |
|---|---|---|
| D15 | **Continue in this repo; rename the product and package to `groundwork`** (`novabridge_qa` → `groundwork`). The version EPAM saw is tagged **`v1-submission`** (`12e8ea7`, the last `main` before M0). The `~/repos/groundwork` scaffold is reference only: M0 copies the files still referenced to `docs/archive/groundwork-scaffold/` and deletes the scaffold; its `DECISIONS.md` is ported to `docs/DECISIONS.md`, and its migration is read for ideas but not copied (section 5). The GitHub repo and the local folder are renamed to `groundwork`. | This repo has the history, the measured evals and 361 tests; the scaffold has one commit. The tag keeps the version EPAM saw. |
| D16 | **PostgreSQL 16 + pgvector is the single source of truth** for documents, chunks, vectors, jobs, conversations, tickets and checkpoints. **Chroma is removed** once M2 shows eval parity. | ACL, transactions, a durable queue and several workers all need a real database. Keeping Chroma would mean writing every ACL filter twice. |
| D17 | **bm25s stays, derived from Postgres**, one index per chunk index (D20), persisted on a shared volume and reloaded by each API worker when the index's `generation` or `generation_token` changes. A saved index is loaded only for the publication it was built from (D28). A Postgres full-text (`ts_rank`) retriever is added only as a measured comparison. | `ts_rank` has no IDF, so rare exact tokens ("Item 7A", figures, error codes) lose the weight that makes hybrid search work. BM25 found 0.90 of the TXT questions in its top 20 against dense's 0.81. |
| D18 | **Plain SQL through psycopg 3** (async pool) in the app; **Alembic with hand-written SQL** (`op.execute`) for migrations. No ORM. | Every index, cast and filter is visible and explainable. Same choice as the scaffold (S4 in `docs/DECISIONS.md`). |
| D19 | **Vectors in an untyped `vector` column, with one partial HNSW expression index per chunk index**: `USING hnsw ((embedding::vector(384)) vector_cosine_ops) WHERE index_id = 7`. Queries repeat the same cast and predicate, with **`index_id` and the dimension written into the SQL as literals** (both are integers from the `indexes` table, never user input), not passed as parameters. | Different embedders have different dimensions, and HNSW needs a fixed one. This keeps one `chunks` table for every experiment. Literals matter: psycopg 3 prepares a statement server-side after 5 runs, Postgres may then choose a generic plan, and a generic plan can't prove `index_id = $1` matches the partial index's predicate, so the query silently falls back to a sequential scan. |
| D20 | **Chunk indexes are first-class:** an `indexes` table records chunker, contextualizer, embedder and their parameters; chunks belong to one index; one index is marked active (an alias). Experiments build new indexes beside the active one. | Chunking and embedder experiments become side-by-side comparisons, and switching models becomes the blue/green index swap from interview answer Q17. |
| D21 | **Job queue in Postgres** (`SELECT … FOR UPDATE SKIP LOCKED`), consumed by a `worker` service. No Redis, no Celery. | One less service; transactional with the data it changes; a classic pattern worth knowing. Revisit if throughput demands it. |
| D22 | **Async from the API to the store and the LLM, starting in M2.** In M2 endpoints become `async def`, `QAService` and the `ChunkStore` become async over the psycopg async pool, and CPU work (embedding, re-ranking, BM25) runs through `asyncio.to_thread`. The sync LLM client is called through `asyncio.to_thread` until M4 replaces it with async adapters. | Streaming, agent tool calls in parallel and SSE all want async. Doing it with the store in M2 means the async pool is never called from sync code. CPU work stays off the event loop (ONNX Runtime releases the GIL). |
| D23 | **Authentication is a header (`X-User`)** against users and groups seeded from `conf/users.yaml`, behind an `Authenticator` interface. Real auth (OIDC) is out of scope and the README says so. | Lets the ACL be built and tested properly without building an identity provider. |
| D24 | **Streaming (resolves improvement 6):** stream raw tokens as a preview, then send a final `answer` event with the validated, renumbered answer that replaces the preview. | Server-side citation checking (D7) needs the whole answer. Sentence buffering is the alternative; Ivan can switch later. |
| D25 | **Primary LLM: Anthropic** (prompt caching makes contextual chunking cheap); **fallback: OpenAI-compatible** (Groq); **local:** Ollama profile. Small and large model per task, set in `conf/models.yaml`. | Groq's free tier can't carry contextual chunking or agent runs. |
| D26 | **Faithfulness is our own claim-level judge** (a different model from the answer model), calibrated against hand labels. RAGAS runs as an optional comparison script only. | Keeps LangChain out of the main dependencies (rule 4), and a judge we wrote is one we can explain and calibrate. |

---

## 4. Dependencies

Already present stay. New, approved by this handoff (still say which milestone adds each):

**Backend runtime:** `psycopg[binary,pool]`, `pgvector` (Python adapter), `alembic` (+ its `sqlalchemy`
dependency, migrations only), `pypdf`, `pyyaml`, `mcp` (official SDK), `opentelemetry-sdk`,
`opentelemetry-exporter-otlp`, `prometheus-client`.

**Backend, agent comparison only (M9):** `langgraph`, `langgraph-checkpoint-postgres`.

**Optional group `ragas` (M7):** `ragas`. Never in the Docker image.

**Dev (M2):** `pytest-asyncio`. A CI Postgres service was chosen over `testcontainers[postgres]` (D28).

**Frontend:** a router, added in M3b: **TanStack Router** by default (it pairs with TanStack Query
and types route params and search params, which the library filters live in), or React Router if Ivan
prefers. Confirm with Ivan before installing. Nothing else is needed. SSE is read with `fetch` and a stream
reader (`EventSource` can't POST). Ask before adding a Markdown renderer; answers are rendered as
text, never HTML (M11).

**Services:** `pgvector/pgvector:pg16` (pgvector ≥ 0.8 for iterative index scans), optional
`langfuse` behind a compose profile (M10).

---

## 5. Target architecture

```
                         ┌──────────────── worker service (jobs from Postgres) ─────────────────┐
 upload / corpus ─▶ jobs ─▶ loader ─▶ structure detector ─▶ Chunker ─▶ Contextualizer ─▶ Embedder ─▶ chunks (Postgres)
                                                                                                └─▶ publish: the generation moves
                                                                                                    └─▶ bm25s rebuilt per index

 user ─▶ Authenticator (X-User → groups)
      ─▶ conversation ─▶ query rewrite (small model, follow-ups only)
      ─▶ answer cache? (key includes groups + index generation + prompt version)
      ─▶ Router (small model): RAG or agent
           RAG:   HybridRetriever(index, groups, filters)
                    ├─ dense: pgvector, ACL + filters in SQL WHERE, iterative HNSW scan
                    ├─ sparse: bm25s, ACL + filters as a mask before top-n
                    └─ RRF ─▶ cross-encoder re-rank ─▶ top k
                  ─▶ relevance gate ─▶ prompt (versioned) ─▶ LLM (stream) ─▶ citation validator ─▶ SSE
           agent: loop with tools (search_docs, get_passage, list_documents, calculate, create_ticket*)
                  ─▶ evidence tracking ─▶ citation validator ─▶ SSE (steps + answer)
                  * create_ticket pauses for human approval
 every stage ─▶ OpenTelemetry span (tokens, cost, latency, prompt version, index id)
```

### Package layout after the rename

```
backend/src/groundwork/
  domain/        models.py (Document, Chunk, Collection, User, SearchFilters, ScoredPassage)
  auth/          users.py (Authenticator, HeaderAuthenticator)
  db/            pool.py, migrations/ (Alembic), queries/ (one module per table group)
  ingestion/     loaders/ (txt, md, html, pdf), structure/ (detectors by format),
                 chunkers/ (window, semantic), contextualizers/ (none, header, llm),
                 pipeline.py, jobs.py (queue), worker.py, corpus.py
  retrieval/     store.py (ChunkStore protocol), pgvector_store.py, sparse.py, fusion.py,
                 reranker.py, embedder.py, hybrid.py, indexes.py (registry, active alias)
  llm/           base.py (messages, content blocks, tools, events), anthropic_client.py,
                 openai_compatible.py, fake.py, resilient.py (retry, fallback, budget),
                 routing.py, pricing.py, structured.py
  prompts/       loader.py (front matter, version, hash)        (prompt files: backend/prompts/*.md)
  qa/            service.py, rewrite.py, citations.py, highlight.py, cache.py, conversations.py
  agent/         loop.py, tools.py, evidence.py, router.py, memory.py, graph/ (LangGraph, M9)
  tickets/       service.py
  mcp_server.py
  observability/ tracing.py, metrics.py, redaction.py
  api/           routes_*.py, sse.py, deps.py, ratelimit.py
```

### Data model (a new first migration)

The scaffold's migration (`docs/archive/groundwork-scaffold/0001_initial.py`) is **reference only, not a
port**: it has only `documents`, `chunks` and `tickets`, and one global HNSW index that D19 replaces.
Write the migration from the schema below.

```sql
users        (id text PK, display_name text, groups text[] NOT NULL)     -- seeded from conf/users.yaml
collections  (id text PK, name text, description text, default_access_groups text[])
documents    (id bigserial PK, manifest_key text,       -- the corpus document's key; NULL for uploads
              content_hash text NOT NULL, collection_id text REFERENCES collections,
              title text, filename text, mime text, doc_type text, doc_date date,
              metadata jsonb DEFAULT '{}', access_groups text[] NOT NULL, owner_id text REFERENCES users,
              licence text, planted text,                 -- e.g. 'injection', 'contradiction-old'
              created_at timestamptz DEFAULT now())
              -- two partial unique indexes: a corpus document is identified by (collection_id,
              -- manifest_key) WHERE manifest_key IS NOT NULL; an upload by (content_hash,
              -- collection_id, access_groups, owner_id) NULLS NOT DISTINCT WHERE manifest_key IS NULL.
              -- See rule 8. NULLS NOT DISTINCT (Postgres 15+): an upload has no owner before M3, and
              -- without it uploading a file twice would insert duplicates. access_groups is stored
              -- sorted and de-duplicated (a CHECK or the insert code), so {hr,everyone} = {everyone,hr}.
indexes      (id serial PK, name text UNIQUE, chunker text, chunker_params jsonb,
              contextualizer text, contextualizer_params jsonb, embedder_model text, dimension int,
              status text CHECK (status IN ('building','ready','retired')),
              generation bigint NOT NULL DEFAULT 0,
              generation_token uuid NOT NULL DEFAULT gen_random_uuid(),  -- new with each generation
              unpublished bool NOT NULL DEFAULT false,    -- has writes BM25 has not seen
              is_active bool NOT NULL DEFAULT false,
              created_at timestamptz DEFAULT now())
              -- partial unique index: only one row with is_active
chunks       (id bigserial PK, index_id int REFERENCES indexes ON DELETE CASCADE,
              document_id bigint REFERENCES documents ON DELETE CASCADE, ordinal int,
              text text NOT NULL,              -- verbatim, what users see and cite
              context text,                    -- contextualizer output, indexed but never cited
              section_path text[],             -- heading path, e.g. {'Item 7', 'Liquidity'}
              page int, char_start int, char_end int, token_count int,
              access_groups text[] NOT NULL,   -- denormalised from documents: the ACL runs in the same query
              collection_id text, doc_date date,
              embedding vector,                -- untyped: see D19
              tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(context,'') || ' ' || text)) STORED,
              UNIQUE (index_id, document_id, ordinal))
              -- GIN on access_groups, GIN on tsv, btree on (index_id, collection_id, doc_date) and on
              -- document_id,
              -- one partial HNSW expression index per index row (created by code when the index is built)
chunk_contexts (cache_key text PK, context text, model text, prompt_version text,
                input_tokens int, output_tokens int, created_at timestamptz)   -- key: hash(doc, chunk, prompt, model)
jobs         (id uuid PK, kind text, payload jsonb, status text, progress int, attempts int,
              max_attempts int, run_after timestamptz, locked_by text, locked_at timestamptz,
              error text, created_by text, created_at, updated_at)
conversations(id uuid PK, user_id text, title text, created_at)
messages     (id bigserial PK, conversation_id uuid, role text, content text,
              sources jsonb,                   -- a snapshot of each cited passage, so a deleted
                                               -- document leaves "source deleted", not a broken answer
              refusal text, usage jsonb, trace_id text, created_at)
feedback     (id bigserial PK, message_id bigint REFERENCES messages ON DELETE CASCADE, user_id text,
              rating smallint, reason text, promoted_to_eval bool DEFAULT false, created_at)
eval_labels  (id bigserial PK, kind text,           -- 'claim' (judge calibration) | 'failure' (taxonomy)
              run_id text, question_id text, claim text, passage_id text, label text,
              note text, labelled_by text, created_at)
answer_cache (key text PK, response jsonb, created_at, hits int)
tickets      (id bigserial PK, title text, body text, priority text, status text,
              created_by text, approved_by text, thread_id text, created_at)
user_memories(id bigserial PK, user_id text, content text, source text, created_at)
rate_limits  (user_id text, window_start timestamptz, route text, count int, PRIMARY KEY (...))
-- LangGraph's checkpoint tables are created by its own setup() in M9.
```

---

## 6. Milestones

Each milestone lists **Why**, **Scope**, **Files**, **Tests**, **Measure** and **Done when**. Order
matters: each one gives the next a measured baseline.

### M0: Freeze, rename, ground rules

**Why:** keep the submitted version reachable, then make every later diff about one thing.

**Scope:**
- Tag the last `main` before M0 as `v1-submission` (D15) and push the tag. Done: `12e8ea7`.
- Delete the NovaBridge mockups (`docs/design/01-ask.html` … `04-no-evidence.html`) and `ai-logs/`; both
  stay reachable at `v1-submission`.
- Commit the untracked files: `books/` (a corpus in M1; check the Project Gutenberg licence header is
  kept), `docs/design/` (the M3b–M12 mockups and `groundwork.css`) and this file.
- **Absorb and delete the scaffold.** `~/repos/groundwork` has one commit and no remote, so deleting it
  is permanent. First copy what this handoff still references into `docs/archive/groundwork-scaffold/`: `README.md` (the brief, as `BRIEF.md`),
  `DECISIONS.md`, `CLAUDE.md`, `docker-compose.yml`, `conf/users.yaml`, `conf/app.yaml` and
  `api/migrations/versions/0001_initial.py`, with a short `README` saying they are a frozen reference.
  Ivan confirms the list, then the scaffold folder is deleted. Its code (`api/`, `web/`, tests) is not
  kept: this repo supersedes it.
- Rename the package `novabridge_qa` → `groundwork`, the Docker images, the compose project, the API
  title and the UI name. Purely mechanical: no behaviour change.
- **Rename the repository**, after the scaffold is gone and the rename above is merged: the GitHub repo
  `novabridge-qa` → `groundwork` (`gh repo rename`; GitHub redirects the old URLs) and update `origin`;
  the local folder `~/repos/epam` → `~/repos/groundwork`. Moving the folder resets Claude Code's
  per-project memory, which is keyed by path: move `~/.claude/projects/-Users-…-repos-epam/memory/` to
  the new project key.
- Move `docs/HANDOFF.md` → `docs/archive/HANDOFF-novabridge.md` and `docs/IMPROVEMENTS.md` →
  `docs/archive/`; this file becomes `docs/HANDOFF.md`. Create `docs/DECISIONS.md` with D1–D26.
- Update `CLAUDE.md`: new rules from section 2 (LangGraph and RAGAS exceptions, ACL rule, no
  existence leaks), and point the dependency rule at section 4 of the new `docs/HANDOFF.md` (this
  file), not the archived one.
- Add a CI job that builds both Docker images (open item from IMPROVEMENTS).

**Tests:** all existing tests pass unchanged apart from import paths.
**Measure:** `run_eval.py` output identical to `results.md` (same index, same code).
**Done when:** `v1-submission` pushed, untracked files committed, scaffold archived and deleted, package rename merged, repo and
folder renamed, CI green including image builds.

### M1: Generic domain, structure by format, new corpora, generic eval set

**Why:** stop being a 10-K tool before changing storage, so the store change in M2 is measured on
the final corpus.

**Scope:**
- **Domain:** replace `company`, `ticker`, `DocType` literals and `period_end` with `collection_id`,
  free-text `doc_type`, `doc_date`, `metadata: dict`, `access_groups`, `owner_id`.
  `SearchFilters` becomes `collections`, `doc_types`, `date_from`, `date_to`, `metadata` equality.
  10-K manifests map ticker and form into `metadata` and `doc_type`, and the period into `doc_date`.
- **Structure detection by format** (`ingestion/structure/`): a `StructureDetector` protocol returning
  sections with a **heading path**; Markdown (ATX headings), HTML (`h1`–`h4`), SEC filings (the
  existing `Item N.` rule, selected by `doc_type` 10-K/10-Q, not run on everything), books (chapter
  headings: `CHAPTER`/`Chapter` + numeral), plain text fallback (one section). Chosen per document
  from a registry; tests for each.
- `Chunk.section` becomes `section_path: tuple[str, ...]`; the indexing header joins title and path.
- **PDF loader** (improvement 8): `pypdf`, fills `page_spans`, chunks get page numbers, a PDF without a
  text layer gets a 422.
- **HTML table alignment** (improvement 11), as specified in `docs/archive/IMPROVEMENTS.md`. Its own
  commit with before-and-after eval numbers.
- **Corpora** (`backend/data/raw/<collection>/manifest.json`, each entry with source, licence,
  access groups, date and `planted`). Not committed (D27): all four are published together as a
  corpora version, the tarball `corpora-vN.tar.gz` on the GitHub Release `corpora-vN`, built by
  `scripts/build_corpora.py` with the manifests inside and published by hand (`gh release create`).
  `backend/conf/corpora.lock.json` names the version, URL and sha256; the ingest and eval scripts
  download, verify and install it into the git-ignored `data/raw/` when it is missing or another
  version, and stop rather than replace local changes there (D27).
  The M1 baseline is measured on `corpora-v1`:
  - `sec-10k`: the existing six filings, access `everyone`, fetched by `scripts/fetch_filings.py`.
  - `engineering`: about 150 Markdown pages from the Kubernetes website docs (CC BY 4.0), pinned to
    a commit hash recorded in the manifest, fetched by `scripts/fetch_engineering_docs.py`.
  - `northwind`: about 30 short fictional documents for "Northwind Labs": runbooks and on-call guides
    (`everyone`), HR policies (`hr`), finance runbooks (`finance`). **Planted:** an old and a new
    expense policy with dates that contradict each other; one document with an indirect prompt
    injection (hidden text telling the assistant to create a ticket and reveal HR content); one HR
    document whose title alone would be sensitive (for the existence-leak tests). Ivan reviews the
    texts before they're packed into a corpora version; `build_corpora.py` takes them from the local
    `data/raw/northwind/`, as there is no upstream.
  - `books`: three Project Gutenberg books (Frankenstein, Pride and Prejudice, Sherlock Holmes),
    public domain, `everyone`, fetched by `scripts/fetch_books.py`. Long, weakly structured prose: the best
    test bed for semantic and contextual chunking.
- **Users:** `conf/users.yaml` with `alice` (everyone), `hana` (everyone, hr), `filip` (everyone,
  finance), `admin` (everyone, hr, finance, admin). Loaded now, enforced in M3.
- **Eval schema** (`eval/questions.jsonl`): generic fields `id`, `collection`, `question`,
  `documents` (manifest keys whose evidence counts), `evidence` (a list: one pattern per required
  fact, so multi-hop questions need all of them), `expected`, `user` (default `alice`), `type`
  (`single`, `multi_hop`, `unanswerable_off_topic`, `unanswerable_in_domain`, `restricted`,
  `contradiction`, `injection`). Convert the 53 existing questions mechanically and check the
  converted scores match. Add about 30 engineering, 25 Northwind (including restricted questions asked
  by the wrong user, the contradiction and the injection) and 15 books questions. **Write them before
  running anything on the new corpora**, and set aside 20 of them, untouched, as the held-out set
  (`eval/heldout.jsonl`) used once per major milestone report.
- **Metrics vocabulary:** rename the report's R@k to **Success@k** (any required evidence in the top
  k) and add **Recall@k** (share of required evidence found) for multi-hop questions, plus MRR.
  Report per collection and per question type.
- **Prompt:** remove "financial analysts" and the investment-advice rule; the period rule becomes a
  date rule ("when passages carry different dates, say which date each fact comes from"). A new
  version, measured.
- Frontend: replace ticker scopes (`lib/scopes.ts`) with collection filters; upload form asks for
  collection, document type and date instead of company and period.

**Tests:** structure detectors, PDF loader, table alignment, manifest validation, converted eval
questions all validate against their texts (`run_eval.validate`).
**Measure:** finance numbers unchanged within noise (except the table-alignment commit, reported
separately); first baseline for the new collections.
**Done when:** `run_eval.py` produces per-collection tables for four collections from one command,
and Ivan has approved the new questions and the Northwind texts.

### M2: Postgres as the source of truth, bm25s derived, CI regression gate

**Why:** the store every later feature needs (D16–D19). Done before ACL so its effect is measured
alone.

**Scope:**
- Compose: `db` (`pgvector/pgvector:pg16`, healthcheck, volume), `migrate` one-off job running
  `alembic upgrade head`, then `ingest`, `api`, `web`. Ports from the scaffold's `docs/archive/groundwork-scaffold/docker-compose.yml`.
- Migrations: the schema in section 5 (ACL columns included now, enforced in M3).
- Extract a **`ChunkStore` protocol** from `retrieval/dense.py` (`add`, `search`, `get`,
  `all_passages(index_id)`, `documents`, `has_document`), implement `PgVectorStore`, port the tests,
  then delete Chroma and its dependency once parity is shown. The protocol is **async** (D22).
- **Async from the API down** (D22): routes become `async def`, `QAService.ask` and the store async
  over the psycopg async pool; embedding, re-ranking and BM25 run through `asyncio.to_thread`; the
  still-sync LLM client is called through `asyncio.to_thread` until M4. `pytest-asyncio` added here.
- **Index registry** (`retrieval/indexes.py`): create an index row, build its partial HNSW expression
  index (`m = 16, ef_construction = 64`, written out), mark it active. The existing configuration
  becomes index `window-400-header-bge-small`.
- Writes are one transaction per document (replaces the reverse-batch trick from Chroma); upsert on
  `(index_id, document_id, ordinal)`; the document row is the "fully indexed" marker.
- **bm25s per index**, built from `chunks`, saved under
  `data/bm25/<index_id>/<generation>-<token>/`, loaded lazily. Each API worker checks the index's
  `generation` and `generation_token` (cached for a second) before a search and reloads when they
  changed. The token is random and new with each publication, and a saved index is loaded only when
  its index id, generation and token all match, so a recreated or restored database is never served
  one saved for another state. An index built while there are unpublished writes is not saved. Keep
  the in-memory side arrays (chunk id, collection, date, access groups as a bitmask) aligned with
  bm25s positions so filters apply before the top-n cut. Check whether the installed bm25s supports a mask on `retrieve` (`weight_mask`); otherwise keep today's
  score-all-then-filter loop and note its cost. The group bitmask needs a **group → bit mapping built
  per generation** from the groups present in the index, because M3 adds a `private:<user>` group per
  uploader and the set of groups is not fixed.
- **Publishing once per job, not per document:** every write that changes an index's chunks sets
  `unpublished` in the same transaction. Publishing moves the generation, gives it a new
  token and clears `unpublished`, and does nothing when there is nothing to publish. A corpus ingest
  publishes when it ends, also after a failure, so bm25s is rebuilt once rather than once per file; an
  upload publishes in the transaction that stores it; ingest and API start publish what a crashed
  process left.
- **Corpus sync** (`ingestion/sync.py`): an ingest makes the index hold exactly the corpus documents
  the manifests describe, beside the uploads, which it never changes. Documents that left the corpus
  are removed first, from every index. A changed one (bytes, tags or access groups) is removed from
  every index and stored again in this one, in one transaction, after its text is embedded; another
  index gets it at its own next sync. An upload in this index with the same bytes, collection, groups
  and owner stands in for a corpus document. Every index that lost chunks is published, and a
  sync that published anything vacuums `chunks` with `INDEX_CLEANUP ON`, so that no HNSW graph keeps
  deleted chunks. A document that fails doesn't stop the others: it keeps its old copy unless its access
  groups changed, when the old copy is removed from every index, and ingest names it and exits non-zero
  (D28). One sync runs at a time on a database; a sync with no corpus documents is refused.
- **Comparison retriever:** `ts_rank` on `tsv` as a fourth sparse option in the eval only (D17).
- `docs/RETRIEVAL.md`: `EXPLAIN ANALYZE` of a dense query showing the HNSW index used, the same query
  with a selective filter, and what changes with `hnsw.ef_search`. Include the **sixth run of the same
  query through the app's pool** to show the partial index is still used once psycopg prepares the
  statement (D19).
- **CI gate:** a Postgres service in CI (D28), the fastembed models and the corpora cached (the corpora
  by the lock file's hash, and every file's hash is still checked), `run_eval.py --ci` on the
  `sec-10k` and `northwind` collections with **exact search** (`SET LOCAL enable_indexscan = off` for
  the vector query) so CI is deterministic. It installs the locked corpora version and builds its
  index from scratch in the eval database, the app's name plus `_eval` (D28). Fails if Success@6 drops
  more than 3 points below `eval/baseline.json`, per collection; if a collection in the baseline has no
  results; if the baseline names another corpora version; or if the corpora are local. Updating the
  baseline is a deliberate, reviewed commit.

**Tests:** store contract tests run against Postgres (marked `db`; required in CI, skipped locally
without a database, as in the scaffold (S11 in `docs/DECISIONS.md`)); idempotent re-ingest, including an ownerless
document (the partial unique indexes); corpus sync through manifests and the real retriever; a
publication reloads BM25; a bulk ingest publishes once; the dense query plan uses the partial HNSW
index after six executions.
**Measure:** every retrieval number from M1 within noise on Postgres; exact versus HNSW top-20
differences per run (expect a few questions, as with Chroma); `ts_rank` versus bm25s table.
**Done when:** Chroma is gone, the eval matches, and the CI gate fails a deliberately broken branch.

### M3: Users, ACL, persisted jobs, several workers

**Why:** the core promise of the groundwork scenario: a user never gets an answer built from a
document they can't see.

**Scope:**
- `Authenticator` protocol and `HeaderAuthenticator` (`X-User`); unknown user → 401. Every route gets
  the user through a dependency; the frontend sends `X-User` from a minimal user picker (the shell's
  real switcher comes in M3b).
- **Dense ACL in SQL:**
  ```sql
  SET LOCAL hnsw.iterative_scan = strict_order;   -- pgvector ≥ 0.8
  SELECT c.id, 1 - ((c.embedding::vector(384)) <=> %(q)s) AS similarity
  FROM chunks c
  WHERE c.index_id = %(index_id)s
    AND c.access_groups && %(groups)s
    AND <filters>
  ORDER BY (c.embedding::vector(384)) <=> %(q)s
  LIMIT %(n)s;
  ```
  Without iterative scans, HNSW finds its `ef_search` nearest neighbours first and filters after, so a
  restrictive filter can return fewer than n rows. Write the test that shows it, then the fix.
- **Sparse ACL:** the group bitmask from M2, applied before the top-n cut.
- **Fetching chunks by id** (for BM25-only hits) repeats the ACL `WHERE`: defence in depth.
- **Documents and uploads:** uploads get `owner_id` and access groups chosen from the uploader's own
  groups (default: owner only, a `private:<user>` group). Deduplication is scoped (rule 8): the same
  bytes uploaded by alice when hana already has them as `hr` become a separate document; the job
  never reports another user's document.
- `GET /api/documents` and `/api/jobs/{id}` return only what the user may see; a job of another user
  is a 404, not a 403.
- **Job queue** (D21): `jobs` table, `SKIP LOCKED` claim, heartbeat (`locked_at`), retry with backoff
  up to `max_attempts`, then `dead`. `ingestion/worker.py` as its own compose service; the corpus
  ingest becomes jobs too, but only to add: removals and replacements go through corpus sync (D28),
  since a job per file can't see that a document left the corpus. A document stored without its manifest
  key counts as an upload, and a sync leaves it alone. The worker doesn't keep "unpublished" in memory: the
  store's record does that, and `publish` is the only way to move the generation. Job kinds:
  `ingest_document`, `build_index` (M6), `contextualize` (M6).
- API runs with several uvicorn workers (`--workers 2` in compose) to prove no in-process state is
  load-bearing.
- **Library API** (the backend for the M3b Library screen):
  - `GET /api/documents` with filters (collection, doc type, date range, owner, access group, status,
    format), title search, sorting and pagination; only documents the user may see.
  - `GET /api/documents/{id}`: metadata, the indexes containing it, its job history,
    `GET /api/documents/{id}/file` to download the original.
  - `PATCH /api/documents/{id}`: title, doc type, date, metadata and access groups (only groups the
    editor belongs to). Changing groups updates the copied `chunks.access_groups` in one transaction
    that also marks each affected index as unpublished, and publishes them; no re-embedding. Anything
    else that changes what BM25 holds (a title, for one) marks them the same way.
  - `DELETE /api/documents/{id}` and `POST /api/documents/delete` (bulk): owner or admin only. Deletes
    the document row, which cascades to its chunks in every index and to cached contexts; marks each affected
    index as unpublished in the same transaction, and publishes it (bm25s rebuilds, the answer cache
    misses); cancels a running ingestion job for it; deletes the stored file. Collections marked
    `protected` (the demo corpora) need `admin`. Corpus documents are the manifests': the next sync
    undoes an edit to one and brings back a deleted one, so M3 must choose. Recommended: the library
    doesn't edit or delete them. For a document the user can't see, both return 404 (rule 8).
- **Jobs API:** `GET /api/jobs` (own jobs; all for admin) with status filter, `POST
  /api/jobs/{id}/retry` for failed or dead jobs, `POST /api/jobs/{id}/cancel` for queued or running
  ones (a running job checks a cancel flag between embedding batches).

**Tests:** for every retrieval mode (`dense`, `sparse`, `hybrid`, `reranked`): alice never receives
an `hr` or `finance` chunk; hana never receives `finance`; a filter that leaves fewer matching chunks
than `n` still returns all of them; document list and job status hide others' documents; duplicate
upload across users; job retry and dead-letter; a job survives an API restart; delete by a
non-owner is a 404; after a delete, no retriever, cache or document list returns the document;
deleting during ingestion cancels the job and leaves no chunks; changing access groups takes effect on
the next search in every worker.
**Measure:** the `restricted` questions in the eval: refused for the wrong user, answered for the
right one; retrieval numbers for `alice` on everyone-visible questions unchanged.
**Done when:** the ACL test matrix passes, two API workers serve requests, and an upload completes
after the API container restarts mid-job.

### M3b: App shell, Library, Upload and Jobs screens

**Why:** the M3 APIs need a UI, and every later milestone (Chat in M5, Lab in M6–M7, Tickets in M9,
Admin in M10, Settings in M12) needs a routed shell to add its screen to.

**Design first:** the mockups in `docs/design/` (index `00-index.html`) are approved by Ivan before any
screen is built; a screen that changes while built updates its mockup in the same pull request.
Tokens come from `frontend/src/styles/tokens.css`.

**Scope:**
- **Router** (section 4): TanStack Router by default, confirmed with Ivan before installing. Routes for
  every screen in `00-index.html`; screens of later milestones are placeholders until built.
- **App shell:** the navigation from `00-index.html` (Chat, Library, Jobs, Tickets, Lab, Admin,
  Settings; Lab and Admin shown only to `admin`), the active index name, and the **user switcher**
  showing the user's groups. The switcher sets `X-User` for every request and clears the TanStack
  Query cache on change, so no data from one user is shown to the next.
- **Library** (`03-library.html`): table with filters, sort, title search and pagination held in the
  URL search params (typed through the router); bulk select; access groups and status per document.
- **Document drawer** (`04-library-drawer.html`): `/library/:documentId`; edit metadata and access
  groups (only the editor's own groups offered), the indexes containing it, last job, download, delete.
- **Delete** (`05-library-delete.html`): confirmation naming what is removed; bulk delete with mixed
  ownership says which documents the user can't delete instead of failing the batch.
- **Upload** (`06-upload.html`): several files, batch settings (collection, doc type, date), access
  limited to the uploader's own groups, default owner only; replaces today's upload queue.
- **Jobs** (`07-jobs.html`): own jobs (all for admin), progress, retry for failed and dead jobs with the
  reason, cancel for queued and running ones.
- **States** (`18-states.html`): empty, loading, no match, offline, not found and source deleted, as
  shared components later screens reuse. A document the user can't see renders "not found", the same as
  one that doesn't exist (rule 8).

**Tests:** route search params round-trip for library filters; switching user clears cached data
(alice's list never shows after switching to hana and back is refetched); drawer offers only the
editor's groups; bulk delete with mixed ownership; job retry and cancel buttons per status; zod schemas
for every new response.
**Measure:** none beyond the checks; the real-flow run in the pull request covers upload → job →
library → drawer edit → delete as two different users.
**Done when:** Ivan has approved the mockups, every M3 API is reachable from the UI, and the real-flow
run is written down in the pull request.

### M4: LLM layer v2, versioned prompts, tracing skeleton

**Why:** streaming, agents and cost control all depend on it.

**Scope:**
- **Messages with content blocks** (`llm/base.py`): `TextBlock`, `ToolUseBlock`, `ToolResultBlock`;
  `ToolSpec(name, description, input_schema)`; `Completion(content, stop_reason, usage)`;
  `Usage(input, output, cache_read, cache_write, cost_usd)`.
- `LLMClient` (async): `complete(system, messages, *, model, max_tokens, tools=None)`, `stream(...)`
  yielding `TextDelta | ToolCall | UsageEvent | Stop` events, `structured(system, messages,
  output_model: type[BaseModel])` returning a validated object: native structured output where the
  provider and model support it, otherwise a forced tool call; one retry on a validation error
  (with the error fed back), then raise.
- Adapters: Anthropic (system as parameter, `cache_control` on the system prompt and long shared
  prefixes), OpenAI-compatible (Groq, Ollama, OpenAI; full-width citation quirk kept in mind),
  `FakeLLM` scripted by a list of turns including tool calls and stream chunks.
- **`ResilientLLM`** wrapper: our own retry with exponential backoff and jitter on 429 and 5xx
  (honour `retry-after`), SDK retries set to 0 so retries aren't doubled; **fallback** to the next
  provider in the chain on exhausted retries or 5xx; a per-request token budget that raises
  `BudgetExceeded`; never falls back on 4xx other than 429.
- **Model routing:** `conf/models.yaml` maps tasks (`rewrite`, `route`, `answer`, `contextualize`,
  `judge`, `agent`) to provider and model size; prices per million tokens in the same file with a
  "check the provider's pricing page" comment.
- **Prompts as files:** `backend/prompts/<name>.md` with front matter (`name`, `version`,
  `description`); `prompts/loader.py`; `prompts/versions.lock` stores each prompt's text hash; a test
  fails if the text changed without a version bump. Every answer, trace and eval record carries the
  prompt version.
- **Prompt test suite** (`eval/prompts/answer_style.yaml`): 20 cases (cites every claim, refuses
  without evidence, names dates, ignores instructions in passages, stays in scope, follows the format).
  `eval/run_prompt_eval.py` scores a prompt version against the real model; record v1 → v3 in
  `docs/PROMPTS.md`, one change at a time.
- **Tracing skeleton:** request ID middleware, `observability/tracing.py` (`span(name, **attrs)`
  context manager over OpenTelemetry), console/JSONL exporter by default. Spans for each LLM call with
  tokens, cost and model. Exporters come in M10.
- The API, `QAService` and the store are already async since M2 (D22); the new async adapters replace
  the `asyncio.to_thread` calls around the old sync LLM client.

**Tests:** retry and fallback with a fake failing provider (429 then success, 500 ×3 then fallback,
401 never retried); budget exceeded; structured output retry; tool-call round trip with `FakeLLM`;
prompt hash test; cost arithmetic.
**Measure:** answer eval on Anthropic and on Groq: correctness, refusals, tokens and cost per answer,
cache-read tokens with prompt caching on.
**Done when:** the full answer eval runs on Anthropic with Groq as an automatic fallback, and
`docs/PROMPTS.md` has measured v1 → v3.

### M5: RAG pipeline v2 and the streaming chat UI

**Why:** the user-facing features groundwork planned for `/ask`.

**Scope:**
- **Conversations:** tables from section 5; `POST /api/conversations`, `GET` list and messages.
- **Query rewriting** (`qa/rewrite.py`, small model, structured output): only when there is history,
  turn the follow-up into a standalone question; the rewrite is shown in the UI and stored. Retrieval
  and the gate use the rewritten question. Measure on 10 follow-up eval items whether it helps;
  rewriting first-turn questions too is a separate measured experiment.
- **SSE endpoint** `POST /api/conversations/{id}/ask/stream` (keeps `POST /api/ask` for scripts):
  events `rewrite`, `retrieval` (passages, gate outcome), `token`, `answer` (validated, D24),
  `error` (safe message, never provider text), `done`. The gate answers immediately without the LLM.
  nginx: `proxy_buffering off` for this location only.
- **Contradicting documents:** passages carry `doc_date`; the prompt prefers the newest and says an
  older version exists; tested with the planted expense policies.
- **Answer cache** (`qa/cache.py`): key = hash of sorted user groups, normalised question, active
  index id and generation, prompt version and model. Only first-turn questions are cached. **No
  semantic cache**: "R&D in 2001" and "R&D in 2026" are nearly the same vector.
- **Query-embedding cache:** in-process LRU, keyed by model and text.
- **Supporting-sentence highlighting** (improvement 7): deterministic first (shared numbers and word
  overlap), embedding similarity as the fallback; character ranges in the API; `SourceCard` scrolls to
  and highlights them.
- **Frontend:** chat layout with conversation list, streamed tokens, preview replaced by the checked
  answer, citation chips scrolling to highlighted sources, the rewrite shown under the question, user
  switcher, collection filter.

**Tests:** SSE event order; an error mid-stream; the gate path streams no tokens; cache key includes
groups (alice can't get hana's cached answer, the M8 red-team test starts here); cache invalidates when
the index is published; highlight ranges.
**Measure:** time to first token, total latency, cache hit rate on a repeated eval run, highlight
accuracy on 30 real answers checked by hand.
**Done when:** a follow-up question streams through nginx in the browser with highlighted sources,
and the contradiction question answers from the new policy while mentioning the old one.

### M6: The chunking lab: strategies, semantic and contextual chunking, embedder comparison

**Why:** the experiments Ivan wants to play with, built so every result is comparable.

**Scope:**
- **`Chunker` protocol:** `chunk(text, sections, counter, params) -> list[Chunk]`, never across a
  section boundary.
  - `window`: today's packer (sentence/paragraph units, size and overlap).
  - `semantic`: split each section into sentences (reuse `_BOUNDARY` and the abbreviation guard),
    embed each sentence with its neighbours (`buffer` sentences either side), compute the cosine
    distance between consecutive sentences, and cut where it exceeds a threshold. Parameters:
    `breakpoint` (`percentile` with 90/95, or `std` with 1.0/1.5), `buffer` (0/1), `min_tokens`,
    `max_tokens` (a hard cap below the embedder limit: semantic chunks vary a lot in size, and an
    oversized one is split with the window packer). No overlap by default. Its embedding cost is one
    vector per sentence at ingest: record it.
- **`Contextualizer` protocol** (orthogonal to the chunker): what is prepended for indexing, stored in
  `chunks.context` so cited text stays verbatim.
  - `none`, `header` (today's title + heading path, the 0.76 → 0.86 gain), `llm`, `header+llm`.
  - `llm` (Anthropic's Contextual Retrieval): the small model gets the document (or, if it exceeds a
    configured size, the enclosing section plus the document's title and outline) and the chunk, and
    writes 1–2 sentences situating the chunk. A versioned prompt (`prompts/contextualize.md`), the
    document as a cached prefix so each extra chunk costs only its own tokens, results cached in
    `chunk_contexts` so re-indexing is free, run as `contextualize` jobs and paced to the rate limit.
    Anthropic reported large drops in retrieval failures with this method; we check on our corpora
    rather than assume.
  - **Token budget:** bge-small reads 512 tokens and the re-ranker saw up to 447 already. Header +
    context + chunk must fit, so contextual indexes cap the context (e.g. 80 tokens) and use a smaller
    chunk size. Compare strategies at the **same chunk size** (e.g. 320) plus the current 400/header
    as the baseline, or the comparison mixes two changes.
- **Late chunking** (stretch): needs a long-context embedder with token-level outputs; only after the
  embedder comparison, and only if a candidate supports it.
- **Embedder comparison** (improvement 5): bge-small (fastembed), Qwen3-Embedding-0.6B and harrier
  (sentence-transformers extra), each as its own index; record indexing time, query latency, image
  size impact and the retuned `MIN_RELEVANCE` per model.
- **Experiment runner** (`eval/run_chunking_lab.py`): builds an index per configuration from a YAML
  matrix (`eval/chunking_matrix.yaml`), runs the retrieval eval per collection and mode, writes
  `eval/chunking_results.md` and keeps or drops the indexes. Suggested first matrix:
  `window{200,320,400}` × `{header, header+llm}`, `semantic{p90,p95}×{buffer 0,1}` × `{header,
  header+llm}`, on all four collections. Then the answer eval on the best three.
- **Also sweep** RRF k (10, 30, 60, 100) with and without the re-ranker, and re-ranker candidates
  (10, 20, 40), since both are free once indexes exist.
- **Lab UI (admin only):** list indexes with their parameters and status; set the active index; a
  **chunk viewer** showing a document's chunk boundaries and contexts under two strategies side by
  side; a **compare** view that runs one question against two indexes and shows both rankings with
  dense, sparse, RRF and re-rank positions.

**Tests:** both chunkers respect sections and caps; semantic chunker on a synthetic text with a clear
topic shift; contextualizer cache hit; the budget assertion (header + context + chunk ≤ embedder limit);
building an index publishes it and activating it moves every worker to it; the eval never touches the
active index.
**Measure:** `eval/chunking_results.md` from one command, with ingest cost (time, tokens, dollars) per
configuration beside the retrieval scores; held-out set run once for the chosen configuration.
`docs/CHUNKING.md` explains the methods, the results, and what was chosen and why.
**Done when:** the matrix runs end to end, Ivan has looked at the chunk viewer on each collection, and
the active index is chosen from the numbers.

### M7: Evaluation v2

**Why:** to judge generation, not just retrieval, and to keep the numbers honest.

**Scope:**
- **Generic answer eval** (`run_answer_eval.py`): per collection and question type, run as the
  question's `user`; resumable and paced as today; records prompt version, index id, model, tokens,
  cost and refusal reason.
- **Claim-level faithfulness judge** (D26, `eval/judge.py`): split the answer into claims (structured
  output), check each claim against the passages it cites (`supported` / `partial` / `unsupported`
  with a reason), using a different model family from the answer model. Report faithfulness (supported
  claims / claims) and **citation precision** (citations that support their claim).
- **Calibrate the judge:** Ivan labels about 30 claims by hand (`eval/judge_labels.jsonl`); report
  agreement and the confusion matrix before using the judge's numbers anywhere.
- **Context precision and recall** from our evidence patterns (deterministic) next to the judge.
- **RAGAS comparison** (`eval/run_ragas.py`, optional `ragas` group): faithfulness, answer relevancy,
  context precision and recall on the same answers; a section on where it agrees with ours.
- **Failure taxonomy** in the report: corpus missing, bad chunking, retrieval miss, re-ranking miss,
  context noise, generation error, unfaithful claim, bad citation, failed to abstain. Every wrong
  answer gets one label, read by hand.
- **CI:** retrieval gate from M2 stays on every pull request. A `workflow_dispatch` and nightly job
  runs the answer eval and judge on a sample with repository secrets, posting the table as an artifact.
- `docs/EVALUATION.md`: one results table per configuration, the judge calibration, one weakness
  found, one change made, before and after.

**Tests:** scoring functions, claim splitting with `FakeLLM`, label agreement arithmetic.
**Done when:** `docs/EVALUATION.md` numbers match the scripts' output, and the judge's agreement with
Ivan's labels is reported.

### M8: The agent, by hand

**Why:** the agent loop written explicitly, before any framework.

**Scope:**
- `agent/loop.py`, aim for under ~200 lines: model call → tool calls (run concurrently with
  `asyncio.gather` when the model returns several) → tool results → repeat. Stops on an answer, the
  **step limit**, the **token budget**, or a **timeout**. Every step recorded as a structured trace and
  streamed as SSE events (`step`, `tool_call`, `tool_result`, `token`, `answer`).
- **Tools** (`agent/tools.py`), each with a Pydantic input model that generates its JSON schema and
  validates the arguments: `search_docs(query, collections?, date_from?, date_to?, k?)`,
  `get_passage(chunk_id)`, `get_document_outline(document_id)`, `list_documents(collection?)`,
  `calculate(expression)` (an `ast`-based evaluator of arithmetic only, never `eval`),
  `create_ticket(title, body, priority)` which **only returns a draft** in this milestone. Every tool
  runs **as the requesting user**. Tool errors (bad arguments, not found, empty results) go back to the
  model as results, never as exceptions.
- **Evidence tracking** (`agent/evidence.py`): every passage a tool returned gets a number; the final
  answer may cite only those, checked by the existing citation validator.
- **Router** (`agent/router.py`, small model, structured output): `rag` or `agent`, with a reason.
  Comparison, trend, multi-hop and "do something" requests go to the agent.
- **Memory:** short-term history trimmed to a token budget (system prompt and the last steps kept
  verbatim, older tool results summarised); working state (retrieved passage numbers, intermediate
  figures) kept by code, not left in the transcript.
- **Frontend:** an agent mode showing each step and tool call live, collapsible, with the final answer
  and its sources.

**Tests:** a scripted `FakeLLM` multi-step run; parallel tool calls; the step limit stops a looping
model; budget and timeout; malformed tool arguments return an error result; a tool can't return a
restricted chunk; the final answer can't cite an unretrieved passage; `calculate` rejects anything but
arithmetic.
**Measure:** a trajectory eval on the multi-hop questions (right tools, steps, final correctness),
agent versus plain RAG on the same questions, and the router's accuracy and its effect on simple
questions' latency.
**Done when:** the multi-hop eval shows where the agent wins, and simple questions aren't slower.

### M9: LangGraph, human approval, tickets, supervisor, MCP, long-term memory

**Why:** what a framework adds (checkpoints, interrupts, graphs) and what it costs, measured against
M8.

**Scope:**
- **LangGraph agent** (`agent/graph/`, rule 4): the same tools and evidence tracking as a graph with
  `AsyncPostgresSaver`. Before `create_ticket` runs, the graph **interrupts**; the UI shows an
  **approval card** with the draft; `POST /api/agent/{thread_id}/approve` (optionally with edits) or
  `/reject` resumes it. Only an approved ticket is written to `tickets`, with `approved_by`.
- **Supervisor variant:** a supervisor routing to a `researcher` (retrieval-heavy tools) and a `writer`
  (drafts the answer or ticket), passing structured results rather than whole transcripts.
- **Long-term memory:** `user_memories` written only on explicit user request (a `remember` tool needing
  approval) or by an approved summary; each memory has a source and can be listed and deleted in the UI;
  retrieved by similarity and treated as data, like passages.
- **MCP server** (`mcp_server.py`, official SDK): `search_documents`, `get_passage`, `list_documents`,
  and optionally `ask` (the full pipeline with citation checking). stdio for local clients with the user
  from an environment variable; streamable HTTP with a bearer token mapped to a user. Returns stable
  chunk ids and verbatim text. `docs/MCP.md`: how to connect Claude Code and Claude Desktop.
- `docs/AGENTS.md`: hand loop versus LangGraph (lines of code, what each makes easy, what each hides,
  how each is tested), single agent versus supervisor (when each is worth it, with trajectory-eval
  numbers), and the approval flow.

**Tests:** a run pauses at approval, **survives an API restart** (the checkpoint) and resumes; a rejected
ticket is never written; approval by another user is refused; the supervisor routes correctly on scripted
turns; MCP tools enforce the ACL for the mapped user; memory deletion.
**Done when:** the restart test passes in CI, a ticket goes draft → approved → written in the browser,
and Claude Code can search the library through MCP as alice and not see HR documents.

### M10: Observability, cost and limits

**Scope:**
- Spans for every stage: rewrite, route, embedding, dense, sparse, fusion, re-rank, gate, generation,
  citation check, each agent step and tool call; attributes for tokens, cost, latency, model, prompt
  version, index id, refusal reason. Exporters: JSONL (default), OTLP to a self-hosted **Langfuse**
  under a compose profile.
- `/metrics` (Prometheus): latency histograms per stage, tokens and cost counters per model, refusals
  by reason, cache hits, job queue depth and failures.
- `GET /api/metrics/summary` (admin): p50 and p95 latency, average cost per question, cache hit rate,
  refusal mix, over a time window; a small admin page shows it.
- **Rate limiting** per user and route, stricter on ask and agent routes, stored in Postgres so it holds
  across workers; 429 with `retry-after`.
- **PII redaction** in logs and traces (emails, phone numbers; Presidio optional); question text stays
  out of production logs as today.
- `docs/OPERATIONS.md`: how prompts, models and indexes are versioned and rolled out (index alias swap,
  prompt version bump with eval), what is logged and what isn't, budgets and alerts worth setting.

**Tests:** span tree for one request with `FakeLLM`; redaction; rate limit across two workers.
**Done when:** one question's full trace shows in Langfuse with cost, and the summary page matches a
script's computation from the JSONL traces.

### M11: Security

**Scope:**
- `docs/SECURITY.md`: map each OWASP Top 10 for LLM Applications (2025) risk to this app: LLM01
  prompt injection, LLM02 sensitive information disclosure, LLM03 supply chain, LLM04 data and model
  poisoning, LLM05 improper output handling, LLM06 excessive agency, LLM07 system prompt leakage, LLM08
  vector and embedding weaknesses, LLM09 misinformation, LLM10 unbounded consumption. For each: where
  it applies, the defence in place, the test that covers it, and the residual risk.
- **Red-team tests** (real model, marked and run in the nightly job; scripted versions with `FakeLLM`
  in CI):
  - the planted injection document doesn't cause a ticket to be drafted or approved, nor restricted
    content to be revealed, through RAG, the agent or MCP;
  - alice asking about the HR leave policy gets no HR content through `/ask`, the stream, the cache, the
    agent, MCP, the document list or upload deduplication;
  - a direct "ignore your instructions" message doesn't change tool permissions or the system prompt;
  - a passage claiming a false fact with no instructions around it: recorded as what gets through
    (the NovaBridge ACME test), with traceability as the defence.
- Model output rendered as text in the UI, never HTML; upload limits; dependency audit in CI
  (`pip-audit`, `npm audit` reported, not blocking at first).

**Done when:** every OWASP row names a test or says honestly why there isn't one, and the residual-risk
section lists what still gets through.

### M12: Design document, README, polish

**Scope:**
- `docs/DESIGN.md`: one page as presented in a system-design interview: requirements, architecture
  (Mermaid), data model, model choices, evaluation, security, cost, scaling to 1M documents (what
  breaks first, OpenSearch versus pgvector + bm25s, embedding service, sharding bm25s), rollout.
- `docs/AI_WORKFLOW.md`: how this was built with AI coding agents. The NovaBridge session logs are at tag
  `v1-submission` (`ai-logs/` was removed in M0).
- README rewritten for groundwork: what it does, a short GIF of chat, agent approval and the chunking
  lab, how to run it (`make demo` ingests the corpora and opens the app), results tables, links to docs.
- Remaining open items from `docs/archive/IMPROVEMENTS.md` (disclaimer alignment, Anthropic and Ollama
  real runs).

**Done when:** a clean clone runs with `docker compose up --build` and `make demo`, and every number in
the README matches the scripts.

### M13 (optional): Cloud

Terraform for AWS: ECS Fargate (api, worker, web), RDS Postgres with pgvector, S3 for raw files, EFS
or S3 for bm25s indexes, Secrets Manager, CloudWatch or OTLP; GitHub Actions with `terraform plan`
on pull requests and apply on merge; remote state in S3 with locking. Bedrock as an LLM provider is
one more adapter.

---

## 7. Configuration

`config.py` keeps `pydantic-settings` for secrets and environment-specific values. Behaviour that is
reviewed in pull requests moves to YAML under `conf/`:

- `conf/users.yaml`: users and groups (M1).
- `conf/models.yaml`: providers, fallback chain, models per task, prices (M4).
- `conf/retrieval.yaml`: candidates, RRF k, re-rank candidates, top k, gate threshold per embedder (M2).
- `eval/chunking_matrix.yaml`: lab configurations (M6).

`.env.example` stays in sync with `Settings` (the existing test enforces it) and gains `DATABASE_URL`,
`ANTHROPIC_API_KEY`, `LLM_FALLBACK_*`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `LANGFUSE_*`.

---

## 8. Testing strategy

- **Unit:** pure logic with fakes (`FakeLLM`, `FakeEmbedder`, `FakeReranker`), as today.
- **DB:** marked `db`, against real Postgres + pgvector; skipped locally without a database, required
  in CI.
- **API:** FastAPI test client with dependency overrides; SSE parsed into events.
- **ACL matrix:** one parametrised suite over users × retrieval modes × access paths (ask, stream,
  cache, agent tool, MCP, documents, jobs, uploads). It grows with every milestone.
- **Evals:** retrieval (deterministic, CI gate), answer and judge (nightly, secrets), prompt suite,
  trajectory eval, red-team.
- **Real flow:** each milestone ends with a manual run in the browser, written down in the pull request.

---

## 9. Out of scope

Real authentication (OIDC/SSO), OCR, multi-tenant billing, fine-tuning models, a mobile client.

---

## 10. Open questions for Ivan

Defaults are chosen above; change them before the milestone that depends on them.

1. ~~Rename to `groundwork` in this repo, or move into `~/repos/groundwork`?~~ Decided: rename this repo and its folder; the scaffold is archived and deleted in M0.
2. **Chroma removed** after parity (D16), or kept as a second `ChunkStore` for local runs without Docker?
3. **Streaming** with a preview replaced by the checked answer (D24), or sentence-buffered?
4. **Northwind texts:** written by the agent and reviewed by Ivan, or drafted by Ivan?
5. **Corpus size for the engineering docs:** 150 pages keeps CI fast; more makes scaling visible.
6. **Contextual chunking budget:** which provider account and spend limit for the `llm` contextualizer
   runs over four collections.
