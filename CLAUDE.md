# groundwork

Read `docs/HANDOFF.md` fully before writing code. It holds the milestones, rules and decisions D15 on;
`docs/DECISIONS.md` lists every decision, and `docs/archive/HANDOFF-novabridge.md` holds D1–D14.

## Commands

Backend (from `backend/`):

    uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
    uv run uvicorn groundwork.main:create_app --factory

Frontend (from `frontend/`):

    npm run lint && npm run typecheck && npm test

## Rules

- One milestone at a time, each on its own branch (`gw/m3-acl`, …) and pull request; run both check
  commands and wait for Ivan's approval before starting the next.
- Measure, don't assume: changes to ingestion, retrieval or prompts report before-and-after eval numbers.
- No LangChain or LlamaIndex in the RAG pipeline. Exceptions: LangGraph (and its `langchain-core`) only
  under `backend/src/groundwork/agent/graph/`; RAGAS only in the optional `ragas` group for one eval script.
- Ask before adding a dependency not listed in section 4 of `docs/HANDOFF.md`.
- The ACL is enforced inside every retriever, before results are ranked and cut; never after generation,
  never by the prompt. Every new way to reach document text gets an ACL test.
- No existence leaks: a user who can't see a document must not learn it exists (search, lists, upload
  deduplication, job status, cache hits, error messages).
- No side effect without human approval (creating a ticket).
- Prompts are versioned files under `backend/prompts/`; changing one without bumping its version fails CI.
- No secrets in the repo. No invented facts in fixtures, corpora or docs; the fictional Northwind corpus is
  the one exception and names no real company or person.
- Model IDs and prices are configuration, never facts in code.
- If the handoff is wrong or blocks you, stop and say so.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (18ivan18/novabridge-qa) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
