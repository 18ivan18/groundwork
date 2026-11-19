from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from groundwork.domain.models import (
    UNRESTRICTED,
    Access,
    Chunk,
    Document,
    ScoredPassage,
    SearchFilters,
)
from groundwork.llm.base import Completion, LLMError
from groundwork.llm.fake import FakeLLM
from groundwork.qa.prompt import SYSTEM_PROMPT
from groundwork.qa.service import Answer, QAService
from groundwork.retrieval.embedder import FakeEmbedder
from groundwork.retrieval.hybrid import HybridRetriever, SearchResult
from groundwork.retrieval.memory_store import MemoryStore, memory_index
from groundwork.retrieval.sparse import FixedSparse, build_sparse

DOCUMENT = Document(
    title="ACME Corp · Form 10-K",
    collection_id="uploads",
    doc_type="10-K",
    doc_date=None,
    metadata={"company": "ACME Corp", "ticker": "ACME"},
    filename="acme.txt",
    doc_id="doc",
    ingested_at=datetime(2025, 1, 1, tzinfo=UTC),
)


def passage(number: int, similarity: float | None = 0.7) -> ScoredPassage:
    chunk = Chunk(f"doc:{number}", "doc", f"Passage {number}.", (), None, 0, 1)
    return ScoredPassage(chunk, DOCUMENT, number, number, 0.03, similarity)


class StubRetriever:
    def __init__(self, passages: list[ScoredPassage]) -> None:
        self.passages = passages
        self.calls: list[tuple[str, int, SearchFilters | None, Access]] = []

    async def retrieve(
        self, query: str, k: int, filters: SearchFilters | None = None, *, access: Access
    ) -> SearchResult:
        self.calls.append((query, k, filters, access))
        similarities = [p.dense_similarity for p in self.passages if p.dense_similarity]
        return SearchResult(self.passages[:k], max(similarities, default=None))


def make_service(
    passages: Sequence[ScoredPassage], *replies: str | Completion | Exception
) -> QAService:
    return QAService(
        StubRetriever(list(passages)),
        FakeLLM(list(replies)),
        top_k=6,
        min_relevance=0.5,
        max_tokens=100,
    )


def assert_no_answer(answer: Answer, retrieved: int, refusal: str) -> None:
    assert answer.insufficient_evidence
    assert answer.refusal == refusal
    assert answer.text == ""
    assert answer.sources == ()
    assert len(answer.retrieved) == retrieved


async def test_normal_answer_returns_only_cited_passages_renumbered() -> None:
    passages = [passage(1), passage(2), passage(3)]

    answer = await make_service(passages, "Anvils [3]. Roofs [1].").ask("?", access=UNRESTRICTED)

    assert not answer.insufficient_evidence
    assert answer.text == "Anvils [1]. Roofs [2]."
    assert [(s.id, s.passage.chunk.chunk_id) for s in answer.sources] == [
        (1, "doc:3"),
        (2, "doc:1"),
    ]
    assert len(answer.retrieved) == 3


async def test_invalid_citations_are_dropped() -> None:
    answer = await make_service([passage(1)], "Anvils [1]. Made up [7].").ask(
        "?", access=UNRESTRICTED
    )

    assert answer.text == "Anvils [1]. Made up."
    assert len(answer.sources) == 1


async def test_invalid_citations_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    await make_service([passage(1)], "Anvils [1]. Made up [7].").ask("?", access=UNRESTRICTED)

    assert "(7,)" in caplog.text


async def test_no_valid_citations_means_insufficient_evidence() -> None:
    answer = await make_service([passage(1)], "Made up [7].").ask("?", access=UNRESTRICTED)

    assert_no_answer(answer, 1, "no_citations")


async def test_uncited_text_is_not_returned() -> None:
    assert_no_answer(
        (await make_service([passage(1)], "Anvils sell well.").ask("?", access=UNRESTRICTED)),
        1,
        "no_citations",
    )


async def test_a_discarded_reply_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    await make_service([passage(1)], "Anvils sell well.").ask("?", access=UNRESTRICTED)

    assert "no valid citations" in caplog.text


async def test_full_width_citations_from_gpt_oss_count_as_cited() -> None:
    answer = await make_service([passage(1), passage(2)], "Anvils【2】.").ask(
        "?", access=UNRESTRICTED
    )

    assert answer.text == "Anvils[1]."
    assert answer.sources[0].passage.chunk.chunk_id == "doc:2"


async def test_retrieval_gate_does_not_call_the_llm() -> None:
    service = make_service([passage(1, 0.49), passage(2, 0.3)])

    answer = await service.ask("?", access=UNRESTRICTED)

    assert_no_answer(answer, 2, "gate")
    assert answer.generation_ms == 0
    assert answer.input_tokens is None


async def test_gate_threshold_is_inclusive() -> None:
    assert not (
        await make_service([passage(1, 0.5)], "Anvils [1].").ask("?", access=UNRESTRICTED)
    ).insufficient_evidence


async def test_one_relevant_passage_opens_the_gate() -> None:
    passages = [passage(1, 0.1), passage(2, 0.9)]

    assert not (
        await make_service(passages, "Anvils [2].").ask("?", access=UNRESTRICTED)
    ).insufficient_evidence


async def test_a_relevant_candidate_outside_the_top_k_opens_the_gate() -> None:
    # The re-ranker may push the passage dense search liked best out of the top k.
    service = QAService(
        StubRetriever([passage(1, 0.1), passage(2, 0.9)]),
        FakeLLM(["Anvils [1]."]),
        top_k=1,
        min_relevance=0.5,
        max_tokens=100,
    )

    assert not (await service.ask("?", access=UNRESTRICTED)).insufficient_evidence


async def test_passages_found_only_by_bm25_do_not_open_the_gate() -> None:
    assert_no_answer(
        (await make_service([passage(1, None)]).ask("?", access=UNRESTRICTED)), 1, "gate"
    )


async def test_no_passages_does_not_call_the_llm() -> None:
    assert_no_answer((await make_service([]).ask("?", access=UNRESTRICTED)), 0, "gate")


@pytest.mark.parametrize(
    "reply", ["INSUFFICIENT_EVIDENCE", " INSUFFICIENT_EVIDENCE.\n", "INSUFFICIENT_EVIDENCE [1]"]
)
async def test_insufficient_evidence_reply(reply: str) -> None:
    assert_no_answer(
        (await make_service([passage(1)], reply).ask("?", access=UNRESTRICTED)), 1, "model"
    )


async def test_an_answer_has_no_refusal() -> None:
    assert (
        await make_service([passage(1)], "Anvils [1].").ask("?", access=UNRESTRICTED)
    ).refusal is None


async def test_token_usage_is_passed_on() -> None:
    answer = await make_service([passage(1)], Completion("Anvils [1].", 900, 40)).ask(
        "?", access=UNRESTRICTED
    )

    assert (answer.input_tokens, answer.output_tokens) == (900, 40)


async def test_token_usage_is_kept_when_the_model_refuses() -> None:
    answer = await make_service([passage(1)], Completion("INSUFFICIENT_EVIDENCE", 900, 3)).ask(
        "?", access=UNRESTRICTED
    )

    assert (answer.refusal, answer.input_tokens, answer.output_tokens) == ("model", 900, 3)


async def test_llm_errors_reach_the_caller() -> None:
    with pytest.raises(LLMError, match="boom"):
        await make_service([passage(1)], LLMError("boom")).ask("?", access=UNRESTRICTED)


async def test_the_llm_receives_the_prompt_and_token_limit() -> None:
    llm = FakeLLM(["Anvils [1]."])
    service = QAService(
        StubRetriever([passage(1)]), llm, top_k=6, min_relevance=0.5, max_tokens=123
    )

    await service.ask("Who sells anvils?", access=UNRESTRICTED)

    call = llm.calls[0]
    assert call.system == SYSTEM_PROMPT
    assert call.max_tokens == 123
    assert "Who sells anvils?" in call.messages[0].content
    assert "Passage 1." in call.messages[0].content


async def test_filters_and_top_k_reach_the_retriever() -> None:
    retriever = StubRetriever([passage(1)])
    service = QAService(retriever, FakeLLM(["A [1]."]), top_k=6, min_relevance=0.5, max_tokens=100)
    filters = SearchFilters(collections=("sec-10k",), metadata={"ticker": "ACME"})

    hana = Access.of({"everyone", "hr"})

    await service.ask("?", filters, top_k=3, access=hana)

    assert retriever.calls == [("?", 3, filters, hana)]


async def test_top_k_defaults_to_the_setting() -> None:
    retriever = StubRetriever([passage(1)])
    service = QAService(retriever, FakeLLM(["A [1]."]), top_k=6, min_relevance=0.5, max_tokens=100)

    await service.ask("?", access=UNRESTRICTED)

    assert retriever.calls[0][1] == 6


async def test_end_to_end_with_a_real_index() -> None:
    store = MemoryStore(memory_index(FakeEmbedder().dimension))
    texts = ["ACME sells anvils to coyotes.", "The warehouse roof leaks."]
    chunks = [Chunk("x", "x", t, (), None, 0, 1) for t in texts]
    embedder = FakeEmbedder()
    await store.add(DOCUMENT, "hash", chunks, embedder.embed_documents(texts))
    sparse = FixedSparse(await build_sparse(store))
    retriever = HybridRetriever(embedder, store, sparse, candidates=5, rrf_k=60)
    service = QAService(
        retriever, FakeLLM(["ACME sells anvils [1]."]), top_k=2, min_relevance=0.1, max_tokens=100
    )

    answer = await service.ask("Who sells anvils to coyotes?", access=UNRESTRICTED)

    assert answer.text == "ACME sells anvils [1]."
    assert answer.sources[0].passage.chunk.text == texts[0]
