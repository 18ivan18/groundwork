# Decisions

Choices not covered by the brief (`README.md`, moved to `docs/BRIEF.md` in M9). Simplest option wins; each entry says why.

## M0: Scaffold

| # | Decision | Why |
|---|---|---|
| 1 | Project is named **groundwork** (the brief's working title was "knowledge-desk"). | Owner's choice. |
| 2 | Single root `pyproject.toml`; the importable package is the `api/` directory itself (`from api.config import ...`). Tests in `tests/`. | Matches the brief's layout with no `src/` nesting; `scripts/` and `evals/` import the same package. |
| 3 | Config = plain Pydantic models loaded from YAML; no `pydantic-settings`. Only `DATABASE_URL` and API keys come from env. | One less dependency; behaviour (YAML, reviewed) and secrets (env, never committed) stay separate. |
| 4 | Migrations are hand-written SQL in Alembic (`op.execute`), no ORM autogenerate. | Every index and column type is explicit and explainable. |
| 5 | `chunks.access_groups` is **denormalised** from `documents`. | The ACL filter runs inside the same query as the vector / full-text search, with no join; this is what "ACL in the WHERE clause" needs. |
| 6 | `chunks.tsv` is a `GENERATED ALWAYS ... STORED` column over `content` only, `english` config. | Can't drift from content. Heading text can be folded into `content` at chunking time (M2) if keyword recall needs it. |
| 7 | HNSW parameters written out (`m=16, ef_construction=64`, pgvector defaults). | So they can be tuned and discussed in M2. |
| 8 | Sync SQLAlchemy engine for now. | Enough for the scaffold; revisit when SSE endpoints land in M3. |
| 9 | Host ports default to **API 8010, DB 5434** (overridable via `API_PORT` / `DB_PORT`). | 8000 and 5433 are taken by other local projects. Inside Compose the API still talks to `db:5432`. |
| 10 | API container runs `alembic upgrade head` before `uvicorn`. | Fine for one replica; with several, migrate as a separate one-off job. |
| 11 | DB tests are marked `db`: skipped locally when Postgres is down, but `GW_REQUIRE_DB=1` in CI makes a missing DB a failure. | `pytest` works on a fresh clone, and CI can't go green by silently skipping the schema tests. |
| 12 | `LLMProvider` is a `typing.Protocol`; M0 has only `complete` / `stream`. Tool calling, structured output, retries and fallback come in M1. | Structural typing keeps test doubles free of inheritance; the interface grows only when a milestone needs it. |
| 13 | `web/` is an empty placeholder until M9; Compose runs `db` + `api` only. | The M0 "done when" covers the API only. |
| 14 | Python pinned to 3.12 (`.python-version`, `requires-python <3.13`) even though the machine has 3.14. | The brief specifies 3.12; ML wheels (torch, sentence-transformers) in M2 are safest there. |
