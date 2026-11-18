"""Client configuration endpoint."""

from fastapi import APIRouter

from groundwork.api.deps import SettingsDep
from groundwork.api.schemas import ConfigResponse
from groundwork.ingestion.loaders import SUPPORTED_SUFFIXES

router = APIRouter(tags=["config"])


@router.get(
    "/config",
    summary="Get the limits a client should apply",
    description=(
        "What the upload endpoint accepts, so a client can say so before a file is sent. "
        "The upload endpoint checks the same limits itself."
    ),
)
async def config(settings: SettingsDep) -> ConfigResponse:
    return ConfigResponse(
        upload_types=sorted(SUPPORTED_SUFFIXES),
        max_upload_mb=settings.max_upload_mb,
    )
