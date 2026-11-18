# groundwork: notes for Claude Code

The brief is `README.md` (until M9, then `docs/BRIEF.md`). Build milestone by milestone, commit after each, and **stop after each milestone** so the owner can read, run and question it. The owner is revising for a senior AI full-stack interview: prefer explicit code over framework magic, and comment non-obvious choices. Record any decision the brief doesn't cover in `DECISIONS.md`.

Hard rules: no secrets committed, and tests/CI pass with no API keys (use `FakeProvider`). The ACL is enforced in SQL, never in the prompt. No side effect without human approval. Prompts are versioned files under `prompts/`. Every number in the docs comes from a script in the repo.

Commands: `make check` (lint + mypy + tests), `make up` / `make down` (Compose: API on :8010, DB on :5434), `make migrate`.
