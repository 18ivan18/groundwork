import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from run_eval import (
    DocumentKeys,
    Question,
    baseline_check,
    candidate_recall,
    eval_collections,
    evidence_ranks,
    first_hit_rank,
    load_questions,
    normalize,
    question_sets,
    report_groups,
    score,
    share_at_or_above,
    validate,
)
from tsrank import or_query

from groundwork.domain.models import Chunk, Document, DocumentMeta, ScoredPassage
from groundwork.ingestion.corpus import CorpusEntry

FILES = {"acme-2025": "acme-2025.htm", "acme-2001": "acme-2001.txt", "beta-2025": "beta-2025.htm"}
KEYS = DocumentKeys(
    [
        CorpusEntry(
            Path(filename),
            key,
            DocumentMeta(
                title="t",
                collection_id="sec-10k",
                doc_type="10-K",
                doc_date=None,
                filename=filename,
            ),
        )
        for key, filename in FILES.items()
    ]
)


def passage(text: str, key: str = "acme-2025") -> ScoredPassage:
    document = Document(
        title="t",
        collection_id="sec-10k",
        doc_type="10-K",
        doc_date=date(2025, 1, 31),
        filename=FILES[key],
        doc_id="d",
        ingested_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    chunk = Chunk("d:0", "d", text, (), None, 0, 1)
    return ScoredPassage(chunk, document, 1, 1, 0.1, 0.5)


QUESTION = Question(
    id="q",
    collection="sec-10k",
    question="?",
    type="single",
    documents=("acme-2025",),
    evidence=("sold 10 anvils",),
)
FILING = ("sec-10k", "acme-2025")


def test_first_hit_rank_finds_the_first_matching_passage() -> None:
    passages = [passage("nothing"), passage("sold 10 anvils"), passage("sold 10 anvils")]

    assert first_hit_rank(passages, QUESTION, KEYS) == 2


def test_evidence_matches_across_wrapped_lines() -> None:
    assert first_hit_rank([passage("sold 10\n   anvils")], QUESTION, KEYS) == 1


def test_evidence_in_another_document_is_not_a_hit() -> None:
    passages = [passage("sold 10 anvils", "beta-2025"), passage("sold 10 anvils", "acme-2001")]

    assert first_hit_rank(passages, QUESTION, KEYS) is None


def test_any_document_and_any_evidence_of_the_question_count() -> None:
    question = QUESTION.model_copy(
        update={"documents": ("acme-2025", "acme-2001"), "evidence": ("sold 10 anvils", "hired 5")}
    )

    assert (
        first_hit_rank([passage("nothing"), passage("hired 5", "acme-2001")], question, KEYS) == 2
    )


def test_questions_are_validated_when_loaded(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"id": "q", "collection": "c", "question": "?", "type": "guess"}))

    with pytest.raises(ValidationError):
        load_questions(path)


def test_question_ids_must_be_unique(tmp_path: Path) -> None:
    line = json.dumps({"id": "q", "collection": "c", "question": "?", "type": "single"})
    path = tmp_path / "questions.jsonl"
    path.write_text(f"{line}\n{line}\n")

    with pytest.raises(ValueError, match="used twice: q"):
        load_questions(path)


def test_the_collections_to_index_come_from_the_questions() -> None:
    books = QUESTION.model_copy(update={"collection": "books"})

    assert eval_collections([QUESTION, books, QUESTION]) == {"sec-10k", "books"}


def test_the_committed_questions_load_and_ask_as_alice() -> None:
    questions = load_questions(Path(__file__).parents[2] / "eval" / "questions.jsonl")

    assert len(questions) >= 53
    assert all(q.answerable == bool(q.documents) for q in questions)


def test_the_held_out_questions_load_and_share_no_ids() -> None:
    folder = Path(__file__).parents[2] / "eval"
    tuned = load_questions(folder / "questions.jsonl")
    held_out = load_questions(folder / "heldout.jsonl")

    assert len(held_out) == 20
    assert not {q.id for q in tuned} & {q.id for q in held_out}


def test_score_computes_success_and_mrr() -> None:
    scores = score([(1,), (4,), (None,), (2,)], [1, 3])

    assert scores.success == {1: 0.25, 3: 0.5}
    assert scores.mrr == pytest.approx((1 + 1 / 4 + 1 / 2) / 4)


def test_mrr_ignores_ranks_past_its_depth() -> None:
    assert score([(2,), (20,)], [1], mrr_depth=10).mrr == pytest.approx(0.5 / 2)


def test_success_needs_any_evidence_and_recall_counts_each() -> None:
    # A two-fact question with one fact at rank 2, the other not found; a one-fact question.
    scores = score([(2, None), (1,)], [3], recall_k=[3])

    assert scores.success == {3: 1.0}
    assert scores.recall == {3: pytest.approx((0.5 + 1.0) / 2)}
    assert scores.mrr == pytest.approx((1 / 2 + 1) / 2)


def test_evidence_ranks_finds_each_pattern_separately() -> None:
    question = QUESTION.model_copy(update={"evidence": ("sold 10 anvils", "hired 5", "zebra")})
    passages = [passage("hired 5"), passage("nothing"), passage("sold 10 anvils")]

    assert evidence_ranks(passages, question, KEYS) == (3, 1, None)


def test_candidate_recall_separates_what_each_retriever_found_from_what_fusion_kept() -> None:
    ranks = {
        "dense": [1, None, 15, None],
        "sparse": [3, 2, None, 25],
        "hybrid": [1, 4, 9, None],
    }

    found = candidate_recall(["a", "b", "c", "d"], ranks, candidates=20, top_k=6)

    assert found.either == {"a", "b", "c"}
    assert found.dense_only == {"c"}
    assert found.sparse_only == {"b"}
    assert found.lost_by_fusion == {"c"}


def test_threshold_is_inclusive() -> None:
    assert share_at_or_above([0.5, 0.49, 0.7], 0.5) == 2


def test_validate_rejects_evidence_missing_from_the_document() -> None:
    with pytest.raises(ValueError, match="evidence"):
        validate([QUESTION], {FILING: "something else"})


def test_validate_rejects_a_question_about_a_document_not_in_the_corpus() -> None:
    with pytest.raises(ValueError, match="no document acme-2025"):
        validate([QUESTION], {("sec-10k", "acme-2001"): "sold 10 anvils"})


def test_validate_needs_every_evidence_pattern_in_some_document() -> None:
    question = QUESTION.model_copy(
        update={"documents": ("acme-2025", "acme-2001"), "evidence": ("sold 10", "hired 5")}
    )
    texts = {FILING: "sold 10 anvils", ("sec-10k", "acme-2001"): "hired 5 people"}

    validate([question], texts)
    with pytest.raises(ValueError, match="'hired 5'"):
        validate([question], {**texts, ("sec-10k", "acme-2001"): "hired 6 people"})


def test_validate_rejects_an_expected_fact_missing_from_the_document() -> None:
    question = QUESTION.model_copy(update={"expected": ("10 anvils", "11 anvils|eleven")})

    with pytest.raises(ValueError, match=re.escape("'11 anvils|eleven'")):
        validate([question], {FILING: "sold 10 anvils"})


def test_validate_accepts_an_expected_fact_by_any_alternative() -> None:
    question = QUESTION.model_copy(update={"expected": ("ten anvils|10 ANVILS",)})

    validate([question], {FILING: "sold 10 anvils"})


def unanswerable(absent: str) -> Question:
    return Question(
        id="u", collection="sec-10k", question="?", type="unanswerable_in_domain", absent=absent
    )


def test_validate_rejects_absent_term_that_occurs_in_the_collection() -> None:
    with pytest.raises(ValueError, match="does occur"):
        validate([unanswerable("anvil")], {FILING: "Anvil sales"})


def test_absent_terms_are_only_checked_in_the_question_collection() -> None:
    validate([unanswerable("anvil")], {("books", "moby-dick"): "an anvil"})


def test_validate_accepts_true_ground_truth() -> None:
    validate([QUESTION, unanswerable("zebra")], {FILING: normalize("sold 10\nanvils")})


FORMATS = {FILING: ".htm", ("sec-10k", "acme-2001"): ".txt", ("books", "moby-dick"): ".txt"}
OLD = QUESTION.model_copy(update={"id": "o", "documents": ("acme-2001",)})
BOOK = QUESTION.model_copy(update={"id": "b", "collection": "books", "documents": ("moby-dick",)})


def test_question_sets_put_each_question_under_its_collection_and_format() -> None:
    sets = question_sets([QUESTION, OLD, QUESTION, BOOK], FORMATS)

    assert sets == {"books · TXT (1)": [3], "sec-10k · HTM (2)": [0, 2], "sec-10k · TXT (1)": [1]}


def test_report_groups_split_a_collection_only_where_it_varies() -> None:
    multi = QUESTION.model_copy(update={"id": "m", "type": "multi_hop"})

    groups = report_groups([QUESTION, OLD, multi, BOOK], FORMATS)

    assert groups == {
        "books (1)": [3],
        "sec-10k (3)": [0, 1, 2],
        "sec-10k · HTM (2)": [0, 2],
        "sec-10k · TXT (1)": [1],
        "sec-10k · multi_hop (1)": [2],
        "sec-10k · single (2)": [0, 1],
    }


def test_the_ci_gate_allows_three_points_of_noise() -> None:
    baseline = {"northwind": 0.80, "sec-10k": 0.90}

    assert baseline_check({"northwind": 0.77, "sec-10k": 0.95}, baseline) == []
    [failure] = baseline_check({"northwind": 0.76, "sec-10k": 0.90}, baseline)
    assert failure.startswith("northwind: Success@6 0.760")


def test_the_ci_gate_fails_a_collection_without_a_baseline() -> None:
    [failure] = baseline_check({"books": 0.5}, {})

    assert "no baseline" in failure


def test_ts_rank_queries_or_the_words() -> None:
    assert or_query("What was ACME's 2024 revenue?") == "what | was | acme | s | 2024 | revenue"
    assert or_query("?!") == ""
