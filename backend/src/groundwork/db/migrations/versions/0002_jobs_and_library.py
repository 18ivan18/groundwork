"""Columns the job queue and the Library API need.

- jobs.document_id: the document a job produced, for a document's job history.
- jobs.cancel_requested: set by the API, read by the worker between embedding batches.
- jobs.result: what a finished job reports, such as its chunk count.
- documents.storage_path: the original file, for download, relative to the data directory.
- collections.protected: the demo corpora; deleting from them needs admin.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    ALTER TABLE jobs
        ADD COLUMN document_id bigint REFERENCES documents ON DELETE SET NULL,
        ADD COLUMN cancel_requested bool NOT NULL DEFAULT false,
        ADD COLUMN result jsonb NOT NULL DEFAULT '{}';
    CREATE INDEX jobs_document ON jobs (document_id);
    CREATE INDEX jobs_created_by ON jobs (created_by, created_at DESC);
    -- Claimed jobs whose worker stopped sending heartbeats.
    CREATE INDEX jobs_running ON jobs (locked_at) WHERE locked_by IS NOT NULL;

    ALTER TABLE documents ADD COLUMN storage_path text;
    ALTER TABLE collections ADD COLUMN protected bool NOT NULL DEFAULT false;
    """)


def downgrade() -> None:
    op.execute("""
    ALTER TABLE collections DROP COLUMN protected;
    ALTER TABLE documents DROP COLUMN storage_path;
    DROP INDEX jobs_running, jobs_created_by, jobs_document;
    ALTER TABLE jobs DROP COLUMN result, DROP COLUMN cancel_requested, DROP COLUMN document_id;
    """)
