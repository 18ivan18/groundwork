"""Initial schema: documents, chunks (vector + full-text + ACL), tickets.

Revision ID: 0001
Revises:
Create Date: 2026-11-02

Written as plain SQL on purpose: every index and column type is visible here,
which is what you want to be able to explain in an interview.
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE documents (
            id            bigserial PRIMARY KEY,
            source_path   text        NOT NULL UNIQUE,   -- path under corpus/
            title         text        NOT NULL,
            source        text        NOT NULL,          -- e.g. 'kubernetes-docs', 'northwind'
            licence       text        NOT NULL,
            access_groups text[]      NOT NULL,          -- who may see it, e.g. {hr}
            content_hash  text        NOT NULL,          -- sha256 of the raw file: skip unchanged
            doc_date      date,                          -- for "prefer the newer policy"
            created_at    timestamptz NOT NULL DEFAULT now(),
            updated_at    timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE chunks (
            id            bigserial PRIMARY KEY,
            document_id   bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal       int         NOT NULL,          -- position within the document
            heading_path  text[]      NOT NULL DEFAULT '{}',  -- e.g. {Leave, Parental leave}
            content       text        NOT NULL,
            content_hash  text        NOT NULL,          -- idempotent upsert key
            token_count   int         NOT NULL,
            embedding     vector(384),                   -- bge-small-en-v1.5
            -- Keyword side of hybrid search. GENERATED so it can never drift from content.
            -- The two-argument to_tsvector form is IMMUTABLE, which generated columns need.
            tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            -- Denormalised from documents so every retriever can filter in ONE table scan,
            -- inside the index query, without a join: the ACL lives in the WHERE clause.
            access_groups text[]      NOT NULL,
            created_at    timestamptz NOT NULL DEFAULT now(),
            UNIQUE (document_id, content_hash)
        )
        """
    )

    # HNSW for approximate nearest neighbour on cosine distance (the `<=>` operator).
    # m / ef_construction are pgvector's defaults, spelled out so they can be discussed.
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv)")
    # GIN on the array supports `access_groups && ARRAY[...]` (overlap) filters.
    op.execute("CREATE INDEX chunks_access_groups_gin ON chunks USING gin (access_groups)")
    op.execute("CREATE INDEX chunks_document_id ON chunks (document_id)")

    op.execute(
        """
        CREATE TABLE tickets (
            id          bigserial PRIMARY KEY,
            created_by  text        NOT NULL,            -- the X-User who asked
            title       text        NOT NULL,
            body        text        NOT NULL,
            priority    text        NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
            status      text        NOT NULL DEFAULT 'open',
            thread_id   text,                            -- agent run that created it (M6)
            approved_by text,                            -- human who approved the side effect
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tickets")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
