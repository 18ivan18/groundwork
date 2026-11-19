"""FastAPI dependencies."""

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status

from groundwork.auth import Authenticator, UnauthenticatedError
from groundwork.config import Settings
from groundwork.container import Container
from groundwork.domain.users import User
from groundwork.ingestion.uploads import UploadService
from groundwork.library import LibraryService
from groundwork.qa.service import QAService


# app.state is untyped; main.py is the only writer, so these casts are safe.
def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_user(request: Request) -> User:
    authenticator = cast(Authenticator, request.app.state.authenticator)
    try:
        return authenticator.authenticate(request.headers)
    except UnauthenticatedError as error:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(error)) from error


def get_qa_service(request: Request) -> QAService:
    return cast(QAService, request.app.state.qa_service)


def get_upload_service(request: Request) -> UploadService:
    return cast(UploadService, request.app.state.upload_service)


def get_library(request: Request) -> LibraryService:
    container = get_container(request)
    return LibraryService(container.store, container.jobs, data_dir=container.settings.data_dir)


CurrentUser = Annotated[User, Depends(get_user)]
LibraryDep = Annotated[LibraryService, Depends(get_library)]
ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]
