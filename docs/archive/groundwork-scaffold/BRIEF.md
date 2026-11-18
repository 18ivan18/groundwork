# knowledge-desk: practice brief (RAG + agents, FastAPI + React)

> **For the Claude Code agent:** this file is the brief. Build the repository it describes milestone by milestone, committing after each one. The person working with you is revising AI engineering for a senior AI full-stack interview: prefer clear, explicit code over framework magic, explain non-obvious choices in comments, and stop after each milestone so they can read, run and question it. When a decision isn't covered here, take the simplest option and note it in `DECISIONS.md`.

## The scenario

A mid-sized software company wants an **internal knowledge assistant**. Employees ask questions in a chat UI and get answers grounded in the company's documentation, with citations. Some documents are restricted: HR policies are visible only to HR, finance runbooks only to finance. A user must **never** get an answer built from a document they cannot see.

The assistant also has an **agent mode**: it can search the docs, read a full document, summarise, and **create a support ticket**. Creating a ticket is a side effect, so a human must approve it first.

The company needs to know the assistant is good, so the project includes:

- an evaluation set and metrics that run in CI
- tracing for every request, with token and cost accounting
- a security review against the OWASP Top 10 for LLM applications, with tests

## Learning goals

By the end, the person can explain and demonstrate:

- calling several LLM providers behind one interface: streaming, tool calling, structured output, retries, fallback
- prompt engineering as an engineering discipline: versioned prompts, test cases, measured iterations
- embeddings, HNSW indexes, metadata filtering, **hybrid search** (keyword + vector) with reciprocal rank fusion, reranking
- a production RAG pipeline: heading-aware chunking, query rewriting, citations, "I don't know" when evidence is missing, **access control enforced in retrieval**
- RAG evaluation: retrieval metrics (recall@k, MRR) and RAGAS (faithfulness, answer relevancy, context precision/recall)
- the agent loop written by hand, then the same agent in **LangGraph** with checkpoints and a human-approval interrupt
- exposing a tool through **MCP**
- observability: traces, token/cost per request, eval gates in CI
- prompt injection (direct and indirect), excessive agency, data leakage, and practical defences
- streaming answers to a React UI over Server-Sent Events

## Stack

| Area | Choice |
|---|---|
| Language | Python 3.12, managed with `uv` |
| API | FastAPI, Pydantic v2, async; SSE for streaming |
| Database | PostgreSQL 16 with **pgvector** (HNSW index) and built-in full-text search (`tsvector`) for the keyword side; SQLAlchemy 2 + psycopg 3; Alembic migrations |
| Embeddings | Local by default: `BAAI/bge-small-en-v1.5` via `sentence-transformers` (free, deterministic, runs in CI). Optional: OpenAI `text-embedding-3-small` behind the same interface |
| Reranker | Local cross-encoder `BAAI/bge-reranker-base`; can be switched off by config |
| LLMs | Anthropic Claude (primary) and one second provider (OpenAI or Gemini) behind a `LLMProvider` interface; a **`FakeProvider`** that returns scripted responses for tests and CI |
| Agents | Hand-written loop (M5), then LangGraph with a Postgres checkpointer (M6); the official MCP Python SDK (M6) |
| Evaluation | pytest, RAGAS, a small custom retrieval-metrics module |
| Tracing | OpenTelemetry-style spans through a thin `tracing.py` wrapper; exporter configurable: Langfuse (cloud free tier) or LangSmith, with a local JSONL exporter as the default |
| Frontend | React + Vite + TypeScript (Next.js is an acceptable swap; note it in `DECISIONS.md`) |
| Delivery | Docker Compose (db, api, web); GitHub Actions; ruff, mypy (non-strict), pytest |

API keys come only from environment variables (`.env`, git-ignored; ship `.env.example`). Every test and the CI pipeline must pass **without any API key**, using `FakeProvider` and local embeddings.

## The corpus (`scripts/build_corpus.py`)

Two parts, both committed under `corpus/` with a `manifest.json` listing every document, its source, licence and access groups:

1. **Public engineering docs.** About 150 Markdown pages from an openly licensed documentation set cloned from GitHub (for example a subset of the Kubernetes website docs, CC BY 4.0). Record the commit hash so the corpus is reproducible. Access group: `everyone`.
2. **Fictional internal docs.** About 30 short Markdown documents for a made-up company, "Northwind Labs" (no real company names), written once by the agent and committed:
   - engineering runbooks and on-call guides (`everyone`)
   - HR policies: leave, expenses, remote work (`hr`)
   - finance runbooks: invoicing, month-end close (`finance`)
   - **planted cases for testing:** two documents that contradict each other (an old and a new expense policy, with dates); one document containing an **indirect prompt injection** (e.g. hidden text telling the assistant to create a ticket and reveal restricted content). Mark the planted ones in `manifest.json` so tests can find them.

Users are defined in `conf/users.yaml`: `alice` (everyone), `hana` (everyone + hr), `filip` (everyone + finance). Authentication is a simple `X-User` header: real auth is out of scope, and the README must say so.

## Configuration (`conf/app.yaml`)

Chunk size and overlap, `top_k` for each retriever, the RRF constant, whether reranking is on, model names per task (a small model for query rewriting, a larger one for answers), the per-request token budget, the agent step limit, and the tracing exporter.

## Milestones

Each milestone matches a day of the revision plan.

### M0: Scaffold
- Repo layout (`api/`, `web/`, `corpus/`, `conf/`, `evals/`, `docs/`), `pyproject.toml`, ruff, mypy, pytest.
- `docker-compose.yml` with `pgvector/pgvector:pg16`; Alembic migration for `documents`, `chunks` (with `embedding vector(384)`, a `tsvector` column and `access_groups text[]`) and `tickets`.
- FastAPI app with `/health`; `FakeProvider`; a GitHub Actions workflow running lint, type check and tests.
- **Done when:** `docker compose up` gives a healthy API, and CI is green with no secrets set.

### M1: LLM layer and prompts (Days 1–2)
- `LLMProvider` protocol with `complete()`, `stream()`, tool calling and structured output (Pydantic model in, validated object out). Adapters for Claude and the second provider.
- Timeouts, retries with exponential backoff on rate limits and 5xx, and **fallback** to the second provider. Count input/output tokens and estimated cost on every call.
- Prompts live in `prompts/*.md` with a version in front matter, never inline in code.
- `evals/prompts/answer_style.yaml`: 20 test cases for the answer prompt (cites sources, refuses without evidence, follows the format, stays in scope). A runner scores each prompt version; record v1 → v3 results in `docs/PROMPTS.md`.
- **Tests:** retry and fallback logic with a fake failing provider; structured-output validation errors are retried once and then raised.

### M2: Ingestion and retrieval (Day 3)
- `scripts/ingest.py`: Markdown → **heading-aware chunks** (keep the heading path as chunk metadata), embeddings, upsert by content hash so re-ingesting is idempotent.
- Three retrievers behind one interface: vector (pgvector cosine, HNSW), keyword (Postgres full-text `ts_rank`), and **hybrid** with reciprocal rank fusion. Optional cross-encoder rerank on the fused top 30.
- **Access control is a SQL `WHERE` on `access_groups`** in every retriever, never a filter applied after retrieval or an instruction in the prompt.
- `EXPLAIN ANALYZE` a vector query and confirm the HNSW index is used; note it in `docs/RETRIEVAL.md`.
- **Tests:** idempotent ingest; `alice` never retrieves an `hr` chunk under any retriever; RRF on a hand-made example.

### M3: The RAG API (Day 4)
- `POST /ask` → SSE stream: rewrite the query (small model), retrieve, rerank, then generate an answer that cites chunks as `[1]`, `[2]` and returns a `sources` event with document titles and heading paths.
- If the top evidence is below a relevance threshold, answer "I couldn't find this in the documentation" instead of guessing.
- For the two contradicting documents, the answer must prefer the newer one and say that an older version exists.
- **Tests (with `FakeProvider`):** the SSE event order; citations map to real chunk IDs; the no-evidence path.

### M4: Evaluation (Day 5)
- `evals/golden.yaml`: 40 questions, each with the expected source documents and a reference answer; include multi-hop questions, questions with no answer in the corpus, and restricted questions asked by the wrong user.
- `evals/retrieval.py`: recall@5, recall@10 and MRR for vector, keyword, hybrid and hybrid + rerank. **Deterministic** (local embeddings), so it runs in CI.
- `evals/ragas_run.py`: faithfulness, answer relevancy, context precision and context recall on the full pipeline. Needs an API key; run manually.
- `docs/EVALUATION.md`: a results table for each configuration, one weakness found, one change made, and the before/after numbers.
- **Done when:** the comparison table is produced by one command, and the numbers in the docs match its output.

### M5: The agent, by hand (Day 6)
- `agent/loop.py`: the plain agent loop in under ~150 lines. Model call → tool calls → tool results → repeat, with a **step limit**, a token budget, and a structured trace of every step.
- Tools with clear JSON schemas and descriptions: `search_docs(query)`, `get_document(doc_id)`, `create_ticket(title, body, priority)`. Tools run **as the requesting user** (the ACL applies). `create_ticket` in this milestone only returns a draft; it writes nothing.
- Tool errors are returned to the model as results, not raised.
- **Tests:** a scripted `FakeProvider` conversation exercises a multi-step run; the step limit stops a looping model.

### M6: LangGraph, human approval and MCP (Day 7)
- Rebuild the agent as a LangGraph graph with a Postgres checkpointer. Before `create_ticket` runs, the graph **interrupts**; `POST /agent/{thread_id}/approve` (or `/reject`) resumes it.
- Add one multi-agent variant: a **supervisor** that routes to a `researcher` (retrieval-heavy) and a `writer` (drafts the ticket or summary). Write in `docs/AGENTS.md` when this is worth it and when the single agent is better.
- Expose `search_docs` as an **MCP server** (`mcp_server.py`, stdio), with instructions for connecting it to Claude Code.
- `docs/AGENTS.md` also compares the hand-written loop with the LangGraph version: lines of code, what each makes easy, what each hides.
- **Tests:** a run pauses at approval, survives an API restart (the checkpoint), and resumes; a rejected ticket is never written.

### M7: Production concerns (Day 8)
- Trace every request: spans for rewrite, each retriever, rerank, generation and each tool call, with tokens, cost and latency.
- `GET /metrics/summary` (or a small report script): p50/p95 latency, average cost per question, and cache hit rate.
- Response caching for identical (user groups, normalised question) pairs; prompt caching for the system prompt where the provider supports it; route easy steps to the small model.
- **CI gate:** the retrieval eval fails the build if recall@5 drops more than 3 points below the committed baseline. A manual `workflow_dispatch` job runs RAGAS with a repository secret.
- `docs/OPERATIONS.md`: how prompts and models are versioned and rolled out, and what is logged (with PII redacted).

### M8: Security (Day 9)
- `docs/SECURITY.md`: map each OWASP Top 10 for LLM Applications risk to this app, the defence in place and the test that covers it.
- Defences: the ACL in SQL (from M2); retrieved text wrapped in clear delimiters and treated as data in the system prompt; the agent's tools follow least privilege; human approval for side effects (from M6); model output rendered as text in the UI, never as HTML; email/phone redaction in logs; per-user rate limiting.
- **Red-team tests:**
  - the planted injection document does not cause a ticket to be created or restricted content to be revealed
  - `alice` asking "what does the HR leave policy say about…" gets no HR content, including through the agent and through the cache
  - a direct "ignore your instructions" message does not change tool permissions
- Record what still gets through. Honest residual risk is part of the deliverable.

### M9: Frontend, system design and polish (Days 10–12)
- React + Vite UI: a user switcher (alice / hana / filip), a chat with streamed tokens, clickable citations that open the source chunk, an agent mode that shows tool steps live, and an **approval card** for tickets.
- `docker compose up` runs db, api and web; `make demo` ingests the corpus and opens the app.
- `docs/DESIGN.md`: a one-page system design as it would be presented in an interview: requirements, architecture (Mermaid diagram), data model, model choices, evaluation, security, cost, scaling to 1M documents, and rollout.
- `docs/AI_WORKFLOW.md`: how the repo was built with AI coding agents. Which parts were generated, how they were reviewed, what was rejected and why.
- A final README replacing this brief (move the brief to `docs/BRIEF.md`): what it does, a screenshot description or GIF, how to run it, the evaluation results table and links to the docs.

## Rules

- No API key, token or `.env` is ever committed. Tests and CI run with no keys.
- ACL is enforced in the database query. Never rely on the prompt to hide restricted content.
- No side effect without human approval.
- Prompts are versioned files; every prompt change is evaluated before it is kept.
- Every number in the docs is produced by a script in the repo.
- Set a spend limit on the API account; the full RAGAS run should cost less than a few dollars. Default to the small model wherever quality allows.
- Keep dependencies few. Don't add a framework where 30 lines of plain code would do, and explain any exception in `DECISIONS.md`.

## Talking points this gives the person for interviews

- Why hybrid search beats pure vector search on technical docs, with their own recall numbers.
- Why access control belongs in retrieval, not in the prompt, and how it was tested (including the cache).
- How they measured RAG quality, and the one change that moved the numbers.
- A hand-written agent loop versus LangGraph: what the framework actually adds (checkpoints, interrupts, graphs) and what it costs.
- Human-in-the-loop approval that survives a restart.
- Indirect prompt injection: a real test case, the defences, and the honest residual risk.
- Cost and latency control: model routing, caching, token budgets, and the numbers from traces.
- How they work with AI coding agents and review generated code.
