"""The initial schema: section 5 of docs/HANDOFF.md.

Tables for later milestones (jobs, conversations, tickets and the rest) are created now so there
is one schema to read; the code that uses them arrives with its milestone. Access groups are
stored now and enforced from M3.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

    -- access_groups is stored sorted and without duplicates, so {hr,everyone} and {everyone,hr}
    -- are one value and the documents' unique constraint sees them as equal.
    CREATE FUNCTION normalized_groups(groups text[]) RETURNS text[]
        LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
        RETURN ARRAY(SELECT DISTINCT g FROM unnest(groups) AS g ORDER BY g);

    CREATE TABLE users (
        id text PRIMARY KEY,
        display_name text NOT NULL,
        groups text[] NOT NULL
    );

    CREATE TABLE collections (
        id text PRIMARY KEY,
        name text NOT NULL,
        description text,
        default_access_groups text[] NOT NULL DEFAULT '{everyone}'
    );

    CREATE TABLE documents (
        id bigserial PRIMARY KEY,
        content_hash text NOT NULL,
        collection_id text NOT NULL REFERENCES collections,
        title text NOT NULL,
        filename text NOT NULL,
        mime text,
        doc_type text NOT NULL,
        doc_date date,
        metadata jsonb NOT NULL DEFAULT '{}',
        access_groups text[] NOT NULL
            CHECK (cardinality(access_groups) > 0
                   AND access_groups = normalized_groups(access_groups)),
        owner_id text REFERENCES users,
        licence text,
        planted text,
        created_at timestamptz NOT NULL DEFAULT now(),
        -- NULLS NOT DISTINCT: corpus documents have no owner, and without it re-ingesting a corpus
        -- would insert every document again.
        UNIQUE NULLS NOT DISTINCT (content_hash, collection_id, access_groups, owner_id)
    );

    CREATE TABLE indexes (
        id serial PRIMARY KEY,
        name text NOT NULL UNIQUE,
        chunker text NOT NULL,
        chunker_params jsonb NOT NULL DEFAULT '{}',
        contextualizer text NOT NULL,
        contextualizer_params jsonb NOT NULL DEFAULT '{}',
        embedder_model text NOT NULL,
        dimension int NOT NULL CHECK (dimension > 0),
        status text NOT NULL DEFAULT 'building'
            CHECK (status IN ('building', 'ready', 'retired')),
        -- Moves when writes are published, so each API worker knows to reload BM25.
        generation bigint NOT NULL DEFAULT 0,
        -- Random, and new with each generation: a recreated or restored database can reach a
        -- generation number again, but never with the same token. A saved BM25 index names both.
        generation_token uuid NOT NULL DEFAULT gen_random_uuid(),
        -- Set in the transaction of every write that changes this index's chunks and cleared by
        -- publishing, so a later process can publish what a crashed one stored.
        unpublished bool NOT NULL DEFAULT false,
        is_active bool NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now()
    );
    CREATE UNIQUE INDEX indexes_one_active ON indexes ((true)) WHERE is_active;

    CREATE TABLE chunks (
        id bigserial PRIMARY KEY,
        index_id int NOT NULL REFERENCES indexes ON DELETE CASCADE,
        document_id bigint NOT NULL REFERENCES documents ON DELETE CASCADE,
        ordinal int NOT NULL,
        -- Verbatim: what users see and cite.
        text text NOT NULL,
        -- Contextualizer output: indexed, never cited.
        context text,
        section_path text[] NOT NULL DEFAULT '{}',
        page int,
        char_start int NOT NULL,
        char_end int NOT NULL,
        token_count int,
        -- Copied from the document so the ACL and filters run in the vector query itself.
        access_groups text[] NOT NULL,
        collection_id text NOT NULL,
        doc_date date,
        -- Untyped: each index has its own dimension, and its own partial HNSW index over
        -- (embedding::vector(dimension)) WHERE index_id = <id>, created by retrieval/indexes.py.
        embedding vector NOT NULL,
        tsv tsvector GENERATED ALWAYS AS
            (to_tsvector('english', coalesce(context, '') || ' ' || text)) STORED,
        UNIQUE (index_id, document_id, ordinal)
    );
    CREATE INDEX chunks_access_groups ON chunks USING gin (access_groups);
    CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
    CREATE INDEX chunks_filters ON chunks (index_id, collection_id, doc_date);
    CREATE INDEX chunks_document ON chunks (document_id);

    CREATE TABLE chunk_contexts (
        -- hash(document, chunk, prompt, model)
        cache_key text PRIMARY KEY,
        context text NOT NULL,
        model text NOT NULL,
        prompt_version text NOT NULL,
        input_tokens int,
        output_tokens int,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE jobs (
        id uuid PRIMARY KEY,
        kind text NOT NULL,
        payload jsonb NOT NULL DEFAULT '{}',
        status text NOT NULL,
        progress int NOT NULL DEFAULT 0,
        attempts int NOT NULL DEFAULT 0,
        max_attempts int NOT NULL DEFAULT 3,
        run_after timestamptz NOT NULL DEFAULT now(),
        locked_by text,
        locked_at timestamptz,
        error text,
        created_by text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX jobs_ready ON jobs (run_after) WHERE status = 'queued';

    CREATE TABLE conversations (
        id uuid PRIMARY KEY,
        user_id text NOT NULL,
        title text,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE messages (
        id bigserial PRIMARY KEY,
        conversation_id uuid NOT NULL REFERENCES conversations ON DELETE CASCADE,
        role text NOT NULL,
        content text NOT NULL,
        -- A snapshot of each cited passage, so a deleted document leaves "source deleted" rather
        -- than a broken answer.
        sources jsonb,
        refusal text,
        usage jsonb,
        trace_id text,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE feedback (
        id bigserial PRIMARY KEY,
        message_id bigint NOT NULL REFERENCES messages ON DELETE CASCADE,
        user_id text NOT NULL,
        rating smallint NOT NULL,
        reason text,
        promoted_to_eval bool NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE eval_labels (
        id bigserial PRIMARY KEY,
        -- 'claim' (judge calibration) or 'failure' (taxonomy)
        kind text NOT NULL,
        run_id text,
        question_id text,
        claim text,
        passage_id text,
        label text NOT NULL,
        note text,
        labelled_by text,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE answer_cache (
        key text PRIMARY KEY,
        response jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        hits int NOT NULL DEFAULT 0
    );

    CREATE TABLE tickets (
        id bigserial PRIMARY KEY,
        title text NOT NULL,
        body text NOT NULL,
        priority text,
        status text NOT NULL,
        created_by text,
        approved_by text,
        thread_id text,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE user_memories (
        id bigserial PRIMARY KEY,
        user_id text NOT NULL,
        content text NOT NULL,
        source text,
        created_at timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE rate_limits (
        user_id text NOT NULL,
        window_start timestamptz NOT NULL,
        route text NOT NULL,
        count int NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, route, window_start)
    );
    """)


def downgrade() -> None:
    op.execute("""
    DROP TABLE rate_limits, user_memories, tickets, answer_cache, eval_labels, feedback, messages,
        conversations, jobs, chunk_contexts, chunks, indexes, documents, collections, users;
    DROP FUNCTION normalized_groups(text[]);
    """)
