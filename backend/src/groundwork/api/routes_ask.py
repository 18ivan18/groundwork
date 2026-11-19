"""Question answering endpoint."""

import logging

from fastapi import APIRouter

from groundwork.api.deps import CurrentUser, QAServiceDep
from groundwork.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    PassageResponse,
    RetrievalInfo,
    SourceResponse,
    Timings,
)
from groundwork.domain.models import ScoredPassage, SearchFilters
from groundwork.qa.service import Source

logger = logging.getLogger(__name__)

router = APIRouter(tags=["questions"])


def _to_passage(passage: ScoredPassage) -> PassageResponse:
    document, chunk = passage.document, passage.chunk
    return PassageResponse(
        doc_id=document.doc_id,
        title=document.title,
        collection_id=document.collection_id,
        doc_type=document.doc_type,
        doc_date=document.doc_date,
        metadata=dict(document.metadata),
        section_path=list(chunk.section_path),
        page=chunk.page,
        excerpt=chunk.text,
        chunk_id=chunk.chunk_id,
        retrieval=RetrievalInfo(
            dense_rank=passage.dense_rank,
            sparse_rank=passage.sparse_rank,
            rrf_score=passage.rrf_score,
            rerank_rank=passage.rerank_rank,
        ),
    )


def _to_source(source: Source) -> SourceResponse:
    return SourceResponse(id=source.id, **_to_passage(source.passage).model_dump())


def _to_filters(request: AskRequest) -> SearchFilters | None:
    if request.filters is None:
        return None
    return SearchFilters(
        collections=tuple(request.filters.collections),
        doc_types=tuple(request.filters.doc_types),
        date_from=request.filters.date_from,
        date_to=request.filters.date_to,
        metadata=dict(request.filters.metadata),
    )


# Async all the way down: the embedder, BM25, the re-ranker and the LLM client run in threads.
@router.post(
    "/ask",
    summary="Ask a question",
    description=(
        "Retrieves passages with hybrid search and writes an answer from them. Citations such as "
        "`[1]` are checked on the server: the number matches `id` in `sources`.\n\n"
        "When the documents do not support an answer, this is still a 200: "
        "`insufficient_evidence` is true, `answer` is empty and `retrieved` holds the closest "
        "passages."
    ),
    responses={502: {"model": ErrorResponse, "description": "The LLM provider failed."}},
)
async def ask(request: AskRequest, service: QAServiceDep, user: CurrentUser) -> AskResponse:
    # Questions can reveal client work, so they stay out of logs in production.
    logger.debug("Question: %s", request.question)
    answer = await service.ask(
        request.question, _to_filters(request), request.top_k, access=user.access
    )
    return AskResponse(
        answer=answer.text,
        insufficient_evidence=answer.insufficient_evidence,
        sources=[_to_source(source) for source in answer.sources],
        retrieved=[_to_passage(passage) for passage in answer.retrieved],
        timings_ms=Timings(retrieval=answer.retrieval_ms, generation=answer.generation_ms),
    )
