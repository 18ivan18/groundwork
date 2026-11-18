# groundwork scaffold (frozen reference)

Files copied from the standalone `groundwork` scaffold (one commit, `e12c6c4`) before it was
deleted in M0. They are a **frozen reference**, not part of the build: nothing imports or runs them.

| File | Original path | Referenced by |
|---|---|---|
| `BRIEF.md` | `README.md` | `docs/HANDOFF.md` section 1 (the groundwork brief) |
| `DECISIONS.md` | `DECISIONS.md` | D18 (S4), M2 tests (S11); ported into `docs/DECISIONS.md` |
| `CLAUDE.md` | `CLAUDE.md` | history only |
| `docker-compose.yml` | `docker-compose.yml` | M2 compose services |
| `0001_initial.py` | `api/migrations/versions/0001_initial.py` | section 5: read for ideas, not copied |
| `conf/app.yaml`, `conf/users.yaml` | `conf/` | M1 users, M4 models |

The scaffold's code (`api/`, `web/`, `tests/`, `evals/`, `scripts/`) was not kept: this repo supersedes it.
