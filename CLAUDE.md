# NovaBridge Q&A

Read `docs/HANDOFF.md` fully before writing code. It holds the decisions, milestones and rules.

## Commands

Backend (from `backend/`):

    uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
    uv run uvicorn novabridge_qa.main:create_app --factory

Frontend (from `frontend/`, once it exists):

    npm run lint && npm run typecheck && npm test

## Rules

- One milestone at a time; run the checks and wait for Ivan's approval before starting the next.
- No LangChain or LlamaIndex.
- Ask before adding a dependency not listed in section 4 of the handoff.
- No secrets in the repo. No invented facts in fixtures or the README.
- If the handoff is wrong or blocks you, stop and say so.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (18ivan18/novabridge-qa) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
