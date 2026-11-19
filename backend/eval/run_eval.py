"""Measure retrieval quality and write the tables to eval/results.md.

Run from backend/: uv run python eval/run_eval.py

Needs Postgres (`docker compose up db`). For each chunk size the collections the questions are
about are indexed into an index of their own, `eval-window-<size>-...`, which is kept: a later run
only ingests the documents that changed, and the app's index is untouched.

A question counts as found when a passage from one of its `documents` (manifest keys) contains one
of its `evidence` patterns. That is stricter than matching a section, which can span dozens of
chunks.

Scores are reported per document format, which separates the TXT filings from the HTML ones.

Each retriever hands its top `retrieval_candidates` passages to fusion, so recall at that depth
is the most hybrid search can find. The candidate table compares it with what survives fusion
and the cut to `top_k`.

`--ci` is the regression gate: Success@6 of the full pipeline on the sec-10k and northwind
collections at the configured chunk size, with exact vector search so the result does not depend
on the HNSW graph. It fails when a collection falls more than 3 points below eval/baseline.json.
`--ci --update-baseline` rewrites that file: a deliberate, reviewed commit.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict
from tsrank import ts_rank_search

from groundwork.config import Settings
from groundwork.container import build_embedder, build_reranker, make_retriever
from groundwork.db.pool import Pool, open_pool
from groundwork.domain.models import Document, ScoredPassage
from groundwork.ingestion.corpus import CorpusEntry, corpus_entries
from groundwork.ingestion.loaders import load_file
from groundwork.ingestion.pipeline import ingest_file
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.hybrid import HybridRetriever, SearchMode
from groundwork.retrieval.indexes import drop_index, ensure_index, window_spec
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.reranker import Reranker
from groundwork.retrieval.sparse import FixedSparse, build_sparse

EVAL_DIR = Path(__file__).parent
BASELINE = EVAL_DIR / "baseline.json"
CI_COLLECTIONS = ("sec-10k", "northwind")
# Success@6 may fall this far below the baseline, in points, before CI fails.
CI_TOLERANCE = 0.03
CHUNK_SIZES = (200, 300, 400)
K_VALUES = (1, 3, 6, 8, 10, 20)
# Recall@k is reported at the answer's depth (top_k) and at the candidates' depth.
RECALL_K = (6, 20)
# MRR stops at rank 10, so it stays comparable with the runs recorded in decisions.md.
MRR_DEPTH = 10
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


QuestionType = Literal[
    "single",
    "multi_hop",
    "unanswerable_off_topic",
    "unanswerable_in_domain",
    "restricted",
    "contradiction",
    "injection",
]
UNANSWERABLE_TYPES = frozenset({"unanswerable_off_topic", "unanswerable_in_domain"})


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    collection: str
    question: str
    type: QuestionType
    # Manifest keys, within `collection`, of the documents whose evidence counts.
    documents: tuple[str, ...] = ()
    # One pattern per fact the answer needs: a multi-hop question needs all of them.
    evidence: tuple[str, ...] = ()
    # Facts the answer must state, each a case-insensitive pattern whose alternatives are
    # accepted ways of writing it. Used by run_answer_eval.py.
    expected: tuple[str, ...] = ()
    # Who asks: a user from conf/users.yaml. Access is enforced from M3.
    user: str = "alice"
    # For unanswerable questions: a pattern that must not occur anywhere in the collection, which
    # proves the answer is not there.
    absent: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


# (collection, manifest key)
DocumentKey = tuple[str, str]


@dataclass(frozen=True)
class Scores:
    # Share of questions with any of their evidence in the top k.
    success: dict[int, float]
    # Mean share of each question's evidence patterns found in the top k: below Success@k only
    # for multi-hop questions that found some of their facts but not all.
    recall: dict[int, float]
    mrr: float


def normalize(text: str) -> str:
    """Collapse whitespace, because lines in filings wrap at arbitrary points."""
    return " ".join(text.split())


def load_questions(path: Path) -> list[Question]:
    questions = [
        Question.model_validate(json.loads(line)) for line in path.read_text().splitlines()
    ]
    ids = [question.id for question in questions]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"Question ids used twice: {', '.join(duplicates)}")
    return questions


def eval_collections(questions: Sequence[Question]) -> frozenset[str]:
    """The collections to index: those the questions are about."""
    return frozenset(question.collection for question in questions)


class DocumentKeys:
    """Maps indexed documents back to their manifest keys."""

    def __init__(self, entries: Sequence[CorpusEntry]) -> None:
        self._by_file = {(e.meta.collection_id, e.meta.filename): e.key for e in entries}

    def __call__(self, document: Document) -> DocumentKey:
        collection = document.collection_id
        return collection, self._by_file.get((collection, document.filename), "")


def targets(question: Question) -> frozenset[DocumentKey]:
    return frozenset((question.collection, key) for key in question.documents)


# For each evidence pattern of a question, the rank of the first passage that holds it.
EvidenceRanks = tuple[int | None, ...]


def evidence_ranks(
    passages: Sequence[ScoredPassage], question: Question, keys: DocumentKeys
) -> EvidenceRanks:
    """Per evidence pattern, the rank of the first passage from one of the question's documents
    that holds it, or None."""
    wanted = targets(question)
    texts = [
        (rank, normalize(passage.chunk.text))
        for rank, passage in enumerate(passages, start=1)
        if keys(passage.document) in wanted
    ]
    return tuple(
        next((rank for rank, text in texts if re.search(pattern, text)), None)
        for pattern in question.evidence
    )


def first_hit(ranks: EvidenceRanks) -> int | None:
    """The rank at which any of the evidence was first found."""
    return min((rank for rank in ranks if rank is not None), default=None)


def first_hit_rank(
    passages: Sequence[ScoredPassage], question: Question, keys: DocumentKeys
) -> int | None:
    return first_hit(evidence_ranks(passages, question, keys))


def score(
    questions: Sequence[EvidenceRanks],
    k_values: Sequence[int],
    recall_k: Sequence[int] = RECALL_K,
    mrr_depth: int = MRR_DEPTH,
) -> Scores:
    n = len(questions)
    firsts = [first_hit(ranks) for ranks in questions]
    success = {k: sum(r is not None and r <= k for r in firsts) / n for k in k_values}
    recall = {
        k: sum(sum(r is not None and r <= k for r in ranks) / len(ranks) for ranks in questions) / n
        for k in recall_k
    }
    mrr = sum(1 / r for r in firsts if r is not None and r <= mrr_depth) / n
    return Scores(success=success, recall=recall, mrr=mrr)


@dataclass(frozen=True)
class CandidateRecall:
    """Which questions had their evidence found, by question id."""

    total: int
    dense: frozenset[str]
    sparse: frozenset[str]
    hybrid: frozenset[str]

    @property
    def either(self) -> frozenset[str]:
        return self.dense | self.sparse

    @property
    def dense_only(self) -> frozenset[str]:
        return self.dense - self.sparse

    @property
    def sparse_only(self) -> frozenset[str]:
        return self.sparse - self.dense

    @property
    def lost_by_fusion(self) -> frozenset[str]:
        """Found by a retriever, but not in the fused top k."""
        return self.either - self.hybrid


def candidate_recall(
    ids: Sequence[str], ranks: dict[str, Sequence[int | None]], candidates: int, top_k: int
) -> CandidateRecall:
    """`ranks` holds each mode's first-hit rank per question, in the order of `ids`."""

    def found(mode: str, depth: int) -> frozenset[str]:
        hits = zip(ids, ranks[mode], strict=True)
        return frozenset(i for i, r in hits if r is not None and r <= depth)

    return CandidateRecall(
        total=len(ids),
        dense=found("dense", candidates),
        sparse=found("sparse", candidates),
        hybrid=found("hybrid", top_k),
    )


def share_at_or_above(similarities: Sequence[float], threshold: float) -> int:
    return sum(similarity >= threshold for similarity in similarities)


def validate(questions: Sequence[Question], texts: dict[DocumentKey, str]) -> None:
    """Fail on a question whose ground truth does not hold in its documents.

    `texts` maps each document's (collection, key) to its normalized text.
    """
    for question in questions:
        documents = {}
        for key in targets(question):
            if key not in texts:
                raise ValueError(f"{question.id}: no document {key[1]} in {key[0]}")
            documents[key] = texts[key]
        if question.answerable and not documents:
            raise ValueError(f"{question.id}: evidence without documents")
        for pattern in question.evidence:
            if not any(re.search(pattern, text) for text in documents.values()):
                raise ValueError(f"{question.id}: evidence {pattern!r} is not in its documents")
        for fact in question.expected:
            if not any(re.search(fact, text, re.IGNORECASE) for text in documents.values()):
                raise ValueError(f"{question.id}: expected {fact!r} is not in its documents")
        collection = [text for (name, _), text in texts.items() if name == question.collection]
        if question.absent and any(
            re.search(question.absent, text, re.IGNORECASE) for text in collection
        ):
            raise ValueError(f"{question.id}: {question.absent!r} does occur in the collection")


async def eval_store(
    pool: Pool,
    settings: Settings,
    entries: Sequence[CorpusEntry],
    embedder: Embedder,
    chunk_tokens: int,
    *,
    exact: bool = False,
    prefix: str = "eval",
) -> PgVectorStore:
    """The eval index for this chunk size, holding exactly `entries`. Documents already in it
    are skipped; an index holding others (a file changed since) is rebuilt. The CI gate indexes
    fewer collections, so it uses its own `prefix` rather than rebuilding the full eval's index."""
    spec = window_spec(
        chunk_tokens, settings.chunk_overlap, settings.embedding_model, embedder.dimension
    )
    spec = replace(spec, name=f"{prefix}-{spec.name}")
    store = PgVectorStore(pool, await ensure_index(pool, spec), exact=exact)
    for entry in entries:
        await ingest_file(
            entry.path,
            entry.meta,
            embedder,
            store,
            chunk_tokens=chunk_tokens,
            chunk_overlap=settings.chunk_overlap,
        )
    if await store.count_documents() > len(entries):
        print(f"{spec.name} holds documents no longer in the corpus: rebuilding it.")
        await drop_index(pool, store.index.id)
        return await eval_store(
            pool, settings, entries, embedder, chunk_tokens, exact=exact, prefix=prefix
        )
    return store


async def build_retriever(
    settings: Settings,
    store: PgVectorStore,
    embedder: Embedder,
    reranker: Reranker | None,
) -> HybridRetriever:
    sparse = FixedSparse(await build_sparse(store))
    return make_retriever(settings, embedder, store, sparse, reranker)


@dataclass(frozen=True)
class Row:
    chunk_tokens: int
    chunks: int
    mode: str
    question_set: str
    scores: Scores


def retrieval_table(rows: Sequence[Row]) -> str:
    headers = [f"S@{k}" for k in K_VALUES] + [f"Recall@{k}" for k in RECALL_K]
    lines = [
        f"| Chunk tokens | Chunks | Mode | Questions | {' | '.join(headers)} | MRR |",
        "|---|---|---|---|" + "---|" * (len(headers) + 1),
    ]
    for row in rows:
        values = [row.scores.success[k] for k in K_VALUES] + [
            row.scores.recall[k] for k in RECALL_K
        ]
        cells = " | ".join(f"{value:.2f}" for value in values)
        lines.append(
            f"| {row.chunk_tokens} | {row.chunks} | {row.mode} | {row.question_set} "
            f"| {cells} | {row.scores.mrr:.3f} |"
        )
    return "\n".join(lines)


def candidate_table(
    rows: Sequence[tuple[int, str, CandidateRecall]], candidates: int, top_k: int
) -> str:
    def share(found: frozenset[str], total: int) -> str:
        return f"{len(found) / total:.2f}"

    def ids(found: frozenset[str]) -> str:
        return ", ".join(sorted(found)) or "none"

    lines = [
        f"| Chunk tokens | Questions | Dense S@{candidates} | BM25 S@{candidates} "
        f"| Either S@{candidates} | Hybrid S@{top_k} | Only dense | Only BM25 | Lost by fusion |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for chunk_tokens, name, c in rows:
        lines.append(
            f"| {chunk_tokens} | {name} | {share(c.dense, c.total)} | {share(c.sparse, c.total)} "
            f"| {share(c.either, c.total)} | {share(c.hybrid, c.total)} "
            f"| {ids(c.dense_only)} | {ids(c.sparse_only)} | {ids(c.lost_by_fusion)} |"
        )
    return "\n".join(lines)


def _labelled(groups: dict[str, list[int]]) -> dict[str, list[int]]:
    return {f"{name} ({len(group)})": group for name, group in sorted(groups.items())}


def question_sets(
    answerable: Sequence[Question], formats: dict[DocumentKey, str]
) -> dict[str, list[int]]:
    """Each question's position under its collection and the format of its first document, such
    as "sec-10k · TXT (21)". Every question is in exactly one set."""
    groups: dict[str, list[int]] = {}
    for position, question in enumerate(answerable):
        suffix = formats[(question.collection, question.documents[0])].lstrip(".").upper()
        groups.setdefault(f"{question.collection} · {suffix}", []).append(position)
    return _labelled(groups)


def report_groups(
    answerable: Sequence[Question], formats: dict[DocumentKey, str]
) -> dict[str, list[int]]:
    """The groups the retrieval report scores: each collection, then its formats and question
    types when it has more than one of either. A question appears in several groups."""
    by_collection: dict[str, list[int]] = {}
    by_type: dict[str, list[int]] = {}
    for position, question in enumerate(answerable):
        by_collection.setdefault(question.collection, []).append(position)
        by_type.setdefault(f"{question.collection} · {question.type}", []).append(position)
    by_format = question_sets(answerable, formats)
    groups: dict[str, list[int]] = {}
    for collection, positions in _labelled(by_collection).items():
        groups[collection] = positions
        name = collection.rsplit(" (", 1)[0]
        for subgroups in (by_format, _labelled(by_type)):
            mine = {label: p for label, p in subgroups.items() if label.startswith(f"{name} · ")}
            if len(mine) > 1:
                groups.update(mine)
    return groups


def gate_table(answerable: list[float], off_topic: list[float], in_domain: list[float]) -> str:
    lines = [
        "| Threshold | Answerable kept | Off-topic blocked | In-domain unanswerable blocked |",
        "|---|---|---|---|",
    ]
    for threshold in THRESHOLDS:
        kept = share_at_or_above(answerable, threshold)
        blocked_off = len(off_topic) - share_at_or_above(off_topic, threshold)
        blocked_in = len(in_domain) - share_at_or_above(in_domain, threshold)
        lines.append(
            f"| {threshold:.2f} | {kept}/{len(answerable)} | {blocked_off}/{len(off_topic)} "
            f"| {blocked_in}/{len(in_domain)} |"
        )
    return "\n".join(lines)


def baseline_check(
    found: dict[str, float], baseline: dict[str, float], tolerance: float = CI_TOLERANCE
) -> list[str]:
    """One line per collection that fell more than `tolerance` below its baseline, or has none."""
    failures = []
    for collection, value in sorted(found.items()):
        if collection not in baseline:
            failures.append(f"{collection}: no baseline. Run --ci --update-baseline.")
        elif value < baseline[collection] - tolerance - 1e-9:
            failures.append(
                f"{collection}: Success@6 {value:.3f}, baseline {baseline[collection]:.3f}"
            )
    return failures


async def ci(update_baseline: bool) -> int:
    settings = Settings()
    questions = [
        q for q in load_questions(EVAL_DIR / "questions.jsonl") if q.collection in CI_COLLECTIONS
    ]
    entries = corpus_entries(settings.data_dir / "raw", frozenset(CI_COLLECTIONS))
    keys = DocumentKeys(entries)
    validate(
        questions,
        {(e.meta.collection_id, e.key): normalize(load_file(e.path).text) for e in entries},
    )
    # Restricted questions are asked by a user who must not find the document: from M3 the ACL
    # makes them misses, which would read as a regression here.
    answerable = [q for q in questions if q.answerable and q.type != "restricted"]
    embedder = build_embedder(settings)
    pool = await open_pool(settings.database_url)
    try:
        store = await eval_store(
            pool, settings, entries, embedder, settings.chunk_tokens, exact=True, prefix="ci"
        )
        retriever = await build_retriever(settings, store, embedder, build_reranker(settings))
        found: dict[str, list[EvidenceRanks]] = {}
        for q in answerable:
            passages = await retriever.search(q.question, settings.top_k)
            found.setdefault(q.collection, []).append(evidence_ranks(passages, q, keys))
    finally:
        await pool.close()
    success = {
        c: score(ranks, [settings.top_k]).success[settings.top_k] for c, ranks in found.items()
    }
    for collection, value in sorted(success.items()):
        count = len(found[collection])
        print(f"{collection}: Success@{settings.top_k} {value:.3f} ({count} questions)")
    if update_baseline:
        BASELINE.write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {BASELINE.name}")
        return 0
    failures = baseline_check(success, json.loads(BASELINE.read_text()))
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


async def main() -> None:
    settings = Settings()
    questions = load_questions(EVAL_DIR / "questions.jsonl")
    entries = corpus_entries(settings.data_dir / "raw", eval_collections(questions))
    keys = DocumentKeys(entries)
    by_key = {(e.meta.collection_id, e.key): e for e in entries}
    validate(questions, {key: normalize(load_file(e.path).text) for key, e in by_key.items()})
    answerable = [q for q in questions if q.answerable]
    sets = report_groups(answerable, {key: e.path.suffix.lower() for key, e in by_key.items()})
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)

    rows: list[Row] = []
    candidate_rows: list[tuple[int, str, CandidateRecall]] = []
    misses: list[str] = []
    gate_sections: list[str] = []
    latencies: list[str] = []
    ts_rows: list[Row] = []
    pool = await open_pool(settings.database_url)
    for chunk_tokens in CHUNK_SIZES:
        dense = await eval_store(pool, settings, entries, embedder, chunk_tokens)
        retriever = await build_retriever(settings, dense, embedder, reranker)
        chunk_count = await dense.count_chunks()
        ranks_by_mode: dict[str, list[int | None]] = {}
        for mode in get_args(SearchMode):
            found: list[EvidenceRanks] = []
            seconds = []
            for q in answerable:
                started = time.perf_counter()
                passages = await retriever.search(q.question, max(K_VALUES), mode=mode)
                seconds.append(time.perf_counter() - started)
                found.append(evidence_ranks(passages, q, keys))
            ranks = [first_hit(evidence) for evidence in found]
            ranks_by_mode[mode] = ranks
            latencies.append(
                f"| {chunk_tokens} | {mode} | {statistics.median(seconds) * 1000:.0f} "
                f"| {max(seconds) * 1000:.0f} |"
            )
            for name, positions in sets.items():
                scores = score([found[p] for p in positions], K_VALUES)
                rows.append(Row(chunk_tokens, chunk_count, mode, name, scores))
            for question, rank in zip(answerable, ranks, strict=True):
                if rank is None or rank > settings.top_k:
                    misses.append(f"| {chunk_tokens} | {mode} | {question.id} | {rank} |")
        for name, positions in sets.items():
            candidates = candidate_recall(
                [answerable[p].id for p in positions],
                {m: [r[p] for p in positions] for m, r in ranks_by_mode.items()},
                settings.retrieval_candidates,
                settings.top_k,
            )
            candidate_rows.append((chunk_tokens, name, candidates))

        ts_found = [
            evidence_ranks(await ts_rank_search(dense, q.question, max(K_VALUES)), q, keys)
            for q in answerable
        ]
        for name, positions in sets.items():
            scores = score([ts_found[p] for p in positions], K_VALUES)
            ts_rows.append(Row(chunk_tokens, chunk_count, "ts_rank", name, scores))

        top: dict[str, float] = {}
        for q in questions:
            [hit] = await dense.search(embedder.embed_query(q.question), 1)
            top[q.id] = hit.similarity
        kept = [top[q.id] for q in answerable]
        off_topic = [top[q.id] for q in questions if q.type == "unanswerable_off_topic"]
        in_domain = [top[q.id] for q in questions if q.type == "unanswerable_in_domain"]
        gate_sections.append(
            f"### {chunk_tokens} tokens\n\n"
            f"Best dense similarity: answerable min {min(kept):.3f}, "
            f"median {statistics.median(kept):.3f}; off-topic max {max(off_topic):.3f}; "
            f"in-domain unanswerable max {max(in_domain):.3f}.\n\n"
            + gate_table(kept, off_topic, in_domain)
        )

    await pool.close()
    report = "\n\n".join(
        [
            "# Retrieval evaluation",
            f"Embedder: `{settings.embedding_model}`. Overlap: {settings.chunk_overlap} tokens. "
            f"Candidates per retriever: {settings.retrieval_candidates}. RRF k: {settings.rrf_k}. "
            f"Re-ranker: `{settings.reranker_model or 'off'}` on the top "
            f"{settings.rerank_candidates} after fusion. "
            f"Documents indexed: {len(entries)} in "
            f"{', '.join(sorted(eval_collections(questions)))}. "
            f"Questions: {len(answerable)} answerable, "
            f"{len(questions) - len(answerable)} unanswerable. Unfiltered searches over every "
            "indexed collection; a question counts as found only in the documents it names.",
            "## Success, recall and MRR",
            "**S@k** (Success@k): share of questions with any of their evidence in the top k "
            "(called R@k before M1). **Recall@k**: mean share of each question's evidence found "
            "in the top k; it differs from S@k only for questions that need several facts. "
            f"MRR counts ranks up to {MRR_DEPTH}.\n\n" + retrieval_table(rows),
            "## Candidates before fusion",
            f"Each retriever passes its top {settings.retrieval_candidates} passages to RRF. "
            "*Either* is the share found by at least one of them: the most hybrid search can "
            "return. *Lost by fusion* lists questions found by a retriever but not in the fused "
            f"top {settings.top_k}.\n\n"
            + candidate_table(candidate_rows, settings.retrieval_candidates, settings.top_k),
            f"## Questions not found within top {settings.top_k}",
            "A rank of None means the passage was not in the top "
            f"{max(K_VALUES)}.\n\n| Chunk tokens | Mode | Question | Rank |\n|---|---|---|---|\n"
            + "\n".join(misses),
            "## BM25 and Postgres ts_rank",
            "The same sparse search with Postgres full-text search in place of bm25s, for "
            "comparison only (D17): `ts_rank` has no inverse document frequency. Compare with "
            "the *sparse* rows above.\n\n"
            + retrieval_table([row for row in rows if row.mode == "sparse"] + ts_rows),
            "## Search latency",
            "Wall time of one search on this machine, embedding the question included. One run, "
            "so only differences of tens of milliseconds mean anything.\n\n"
            "| Chunk tokens | Mode | Median ms | Max ms |\n|---|---|---|---|\n"
            + "\n".join(latencies),
            "## Relevance gate (MIN_RELEVANCE)",
            *gate_sections,
        ]
    )
    (EVAL_DIR / "results.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ci", action="store_true", help="Run the regression gate.")
    parser.add_argument(
        "--update-baseline", action="store_true", help="With --ci: rewrite eval/baseline.json."
    )
    args = parser.parse_args()
    if args.ci:
        sys.exit(asyncio.run(ci(args.update_baseline)))
    if args.update_baseline:
        parser.error("--update-baseline needs --ci")
    asyncio.run(main())
