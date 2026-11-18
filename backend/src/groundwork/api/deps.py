"""FastAPI dependencies."""

from typing import Annotated, cast

from fastapi import Depends, Request

from groundwork.config import Settings
from groundwork.container import Container
from groundwork.ingestion.uploads import UploadService
from groundwork.qa.service import QAService


# app.state is untyped; main.py is the only writer, so these casts are safe.
def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_qa_service(request: Request) -> QAService:
    return cast(QAService, request.app.state.qa_service)


def get_upload_service(request: Request) -> UploadService:
    return cast(UploadService, request.app.state.upload_service)


ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]
