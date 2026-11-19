"""Health endpoint."""

from fastapi import APIRouter

from groundwork.api.deps import ContainerDep
from groundwork.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Check the service and count what is indexed",
    description="Answers once startup has finished, so it also serves as the Docker healthcheck.",
)
async def health(container: ContainerDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=container.settings.llm_provider,
        embedder=container.settings.embedding_model,
        documents=await container.store.count_documents(),
        chunks=await container.store.count_chunks(),
    )
