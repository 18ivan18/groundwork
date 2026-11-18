"""Measure retrieval quality and write the tables to eval/results.md.

Run from backend/: uv run python eval/run_eval.py

For each chunk size the collections the questions are about are indexed into a temporary
directory, so the real index is untouched. A question counts as found when a passage from one of
its `documents` (manifest keys) contains one of its `evidence` patterns. That is stricter than
matching a section, which can span dozens of chunks.

Scores are reported per document format, which separates the TXT filings from the HTML ones.

Each retriever hands its top `retrieval_candidates` passages to fusion, so recall at that depth
is the most hybrid search can find. The candidate table compares it with what survives fusion
and the cut to `top_k`.
"""

import json
import re
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

from groundwork.config import Settings
from groundwork.container import build_embedder, build_reranker, make_retriever
from groundwork.domain.models import Document, ScoredPassage
from groundwork.ingestion.corpus import CorpusEntry, corpus_entries
from groundwork.ingestion.loaders import load_file
from groundwork.ingestion.pipeline import ingest_file
from groundwork.retrieval.dense import DenseIndex
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.hybrid import HybridRetriever, SearchMode
from groundwork.retrieval.reranker import Reranker
from groundwork.retrieval.sparse import SparseIndex

EVAL_DIR = Path(__file__).parent
CHUNK_SIZES = (200, 300, 400)
K_VALUES = (1, 3, 6, 8, 10, 20)
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


def first_hit_rank(
    passages: Sequence[ScoredPassage], question: Question, keys: DocumentKeys
) -> int | None:
    """The rank of the first passage from one of the question's documents that holds any of its
    evidence."""
    wanted = targets(question)
    for rank, passage in enumerate(passages, start=1):
        if keys(passage.document) in wanted:
            text = normalize(passage.chunk.text)
            if any(re.search(pattern, text) for pattern in question.evidence):
                return rank
    return None


def score(
    ranks: Sequence[int | None], k_values: Sequence[int], mrr_depth: int = MRR_DEPTH
) -> Scores:
    recall = {k: sum(r is not None and r <= k for r in ranks) / len(ranks) for k in k_values}
    mrr = sum(1 / r for r in ranks if r is not None and r <= mrr_depth) / len(ranks)
    return Scores(recall=recall, mrr=mrr)


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


def build_retriever(
    settings: Settings,
    entries: Sequence[CorpusEntry],
    embedder: Embedder,
    reranker: Reranker | None,
    chunk_tokens: int,
    directory: Path,
) -> tuple[HybridRetriever, DenseIndex]:
    dense = DenseIndex(directory, settings.embedding_model)
    for entry in entries:
        ingest_file(
            entry.path,
            entry.meta,
            embedder,
            dense,
            chunk_tokens=chunk_tokens,
            chunk_overlap=settings.chunk_overlap,
        )
    sparse = SparseIndex.build(dense.all_passages())
    return make_retriever(settings, embedder, dense, sparse, reranker), dense


@dataclass(frozen=True)
class Row:
    chunk_tokens: int
    chunks: int
    mode: str
    question_set: str
    scores: Scores


def retrieval_table(rows: Sequence[Row]) -> str:
    recall_headers = " | ".join(f"R@{k}" for k in K_VALUES)
    lines = [
        f"| Chunk tokens | Chunks | Mode | Questions | {recall_headers} | MRR |",
        "|---|---|---|---|" + "---|" * (len(K_VALUES) + 1),
    ]
    for row in rows:
        recalls = " | ".join(f"{row.scores.recall[k]:.2f}" for k in K_VALUES)
        lines.append(
            f"| {row.chunk_tokens} | {row.chunks} | {row.mode} | {row.question_set} "
            f"| {recalls} | {row.scores.mrr:.3f} |"
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
        f"| Chunk tokens | Questions | Dense R@{candidates} | BM25 R@{candidates} "
        f"| Either R@{candidates} | Hybrid R@{top_k} | Only dense | Only BM25 | Lost by fusion |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for chunk_tokens, name, c in rows:
        lines.append(
            f"| {chunk_tokens} | {name} | {share(c.dense, c.total)} | {share(c.sparse, c.total)} "
            f"| {share(c.either, c.total)} | {share(c.hybrid, c.total)} "
            f"| {ids(c.dense_only)} | {ids(c.sparse_only)} | {ids(c.lost_by_fusion)} |"
        )
    return "\n".join(lines)


def question_sets(
    answerable: Sequence[Question], formats: dict[DocumentKey, str]
) -> dict[str, list[int]]:
    """Group question positions by the format of their first document, such as
    "TXT filings (21)"."""
    groups: dict[str, list[int]] = {}
    for position, question in enumerate(answerable):
        suffix = formats[(question.collection, question.documents[0])]
        groups.setdefault(suffix.lstrip(".").upper(), []).append(position)
    return {f"{name} filings ({len(group)})": group for name, group in sorted(groups.items())}


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


def main() -> None:
    settings = Settings()
    questions = load_questions(EVAL_DIR / "questions.jsonl")
    entries = corpus_entries(settings.data_dir / "raw", eval_collections(questions))
    keys = DocumentKeys(entries)
    by_key = {(e.meta.collection_id, e.key): e for e in entries}
    validate(questions, {key: normalize(load_file(e.path).text) for key, e in by_key.items()})
    answerable = [q for q in questions if q.answerable]
    sets = question_sets(answerable, {key: e.path.suffix.lower() for key, e in by_key.items()})
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)

    rows: list[Row] = []
    candidate_rows: list[tuple[int, str, CandidateRecall]] = []
    misses: list[str] = []
    gate_sections: list[str] = []
    latencies: list[str] = []
    for chunk_tokens in CHUNK_SIZES:
        with tempfile.TemporaryDirectory() as directory:
            retriever, dense = build_retriever(
                settings, entries, embedder, reranker, chunk_tokens, Path(directory)
            )
            ranks_by_mode: dict[str, list[int | None]] = {}
            for mode in get_args(SearchMode):
                ranks, seconds = [], []
                for q in answerable:
                    started = time.perf_counter()
                    passages = retriever.search(q.question, max(K_VALUES), mode=mode)
                    seconds.append(time.perf_counter() - started)
                    ranks.append(first_hit_rank(passages, q, keys))
                ranks_by_mode[mode] = ranks
                latencies.append(
                    f"| {chunk_tokens} | {mode} | {statistics.median(seconds) * 1000:.0f} "
                    f"| {max(seconds) * 1000:.0f} |"
                )
                for name, positions in sets.items():
                    scores = score([ranks[p] for p in positions], K_VALUES)
                    rows.append(Row(chunk_tokens, dense.count_chunks(), mode, name, scores))
                for question, rank in zip(answerable, ranks, strict=True):
                    if rank is None or rank > settings.top_k:
                        misses.append(f"| {chunk_tokens} | {mode} | {question.id} | {rank} |")
            for name, positions in sets.items():
                found = candidate_recall(
                    [answerable[p].id for p in positions],
                    {m: [r[p] for p in positions] for m, r in ranks_by_mode.items()},
                    settings.retrieval_candidates,
                    settings.top_k,
                )
                candidate_rows.append((chunk_tokens, name, found))

            top: dict[str, float] = {
                q.id: dense.search(embedder.embed_query(q.question), 1)[0].similarity
                for q in questions
            }
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

    report = "\n\n".join(
        [
            "# Retrieval evaluation",
            f"Embedder: `{settings.embedding_model}`. Overlap: {settings.chunk_overlap} tokens. "
            f"Candidates per retriever: {settings.retrieval_candidates}. RRF k: {settings.rrf_k}. "
            f"Re-ranker: `{settings.reranker_model or 'off'}` on the top "
            f"{settings.rerank_candidates} after fusion. "
            f"Filings indexed: {len(entries)}. Questions: {len(answerable)} answerable, "
            f"{len(questions) - len(answerable)} unanswerable. Unfiltered searches over all "
            "filings; a question counts as found only in the filing it names.",
            "## Recall and MRR",
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
    main()
