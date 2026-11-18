"""FastAPI application factory."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from groundwork.api import (
    routes_ask,
    routes_config,
    routes_documents,
    routes_health,
    routes_users,
)
from groundwork.api.deps import get_user
from groundwork.api.schemas import ErrorResponse
from groundwork.auth import USER_HEADER, HeaderAuthenticator
from groundwork.config import Settings
from groundwork.container import Container, app_index, build_embedder, build_services
from groundwork.db.pool import open_pool
from groundwork.domain.users import load_users
from groundwork.llm.base import LLMClient, LLMError
from groundwork.llm.factory import build_llm
from groundwork.retrieval.embedder import Embedder
from groundwork.retrieval.pgvector_store import PgVectorStore
from groundwork.retrieval.store import ChunkStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the database and build the heavy objects when a server starts.

    A bad configuration stops the server here, before it takes traffic. Tests that pass their own
    store, or use dependency_overrides without entering this, need no database, no model
    download and no API keys.
    """
    settings: Settings = app.state.settings
    logging.basicConfig(level=settings.log_level)
    pool = None
    embedder: Embedder | None = None
    llm: LLMClient | None = None
    try:
        if app.state.container is None:
            # Before the database: a missing LLM setting should stop startup at once, not after
            # a wait for Postgres.
            llm = build_llm(settings)
            pool = await open_pool(settings.database_url)
            embedder = await asyncio.to_thread(build_embedder, settings)
            index = await app_index(pool, settings, embedder.dimension)
            app.state.container = Container(settings, PgVectorStore(pool, index), app.state.users)
        container: Container = app.state.container
        services = build_services(settings, container.store, embedder, llm=llm)
        app.state.qa_service = services.qa
        app.state.upload_service = services.uploads
        yield
    finally:
        if pool is not None:
            await pool.close()


async def log_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log the path only: bodies hold questions, which must stay out of production logs."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s %d %.0fms", request.method, request.url.path, response.status_code, elapsed_ms
    )
    return response


def handle_llm_error(_: Request, error: Exception) -> JSONResponse:
    # __cause__ goes to the log only. LLMError's own message is written to be safe for users.
    logger.error("LLM request failed: %s", error, exc_info=error.__cause__)
    return JSONResponse(status_code=502, content={"detail": str(error)})


def handle_unexpected_error(_: Request, error: Exception) -> JSONResponse:
    logger.error("Unhandled error", exc_info=error)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong."})


def create_app(settings: Settings | None = None, *, store: ChunkStore | None = None) -> FastAPI:
    """Build the app. uvicorn's `--factory` passes nothing: the store is opened on Postgres when
    the server starts. Tests pass their own settings, and a store to run without a database."""
    settings = settings if settings is not None else Settings()
    app = FastAPI(
        title="groundwork API",
        description=(
            "Answers questions from the indexed documents, with numbered citations that are "
            "checked on the server."
        ),
        version="0.1.0",
        openapi_tags=[
            {"name": "questions", "description": "Ask questions of the library."},
            {"name": "documents", "description": "List the library and add to it."},
            {"name": "health", "description": "Service status."},
            {"name": "users", "description": "The demo users and the current user."},
            {"name": "config", "description": "Limits for clients to apply."},
        ],
        responses={500: {"model": ErrorResponse, "description": "An unexpected error."}},
        lifespan=lifespan,
        # Under /api because that is the only path nginx proxies to this service.
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.users = load_users(settings.users_file)
    app.state.authenticator = HeaderAuthenticator(app.state.users)
    app.state.container = None if store is None else Container(settings, store, app.state.users)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", USER_HEADER],
    )
    app.middleware("http")(log_request)
    app.add_exception_handler(LLMError, handle_llm_error)
    app.add_exception_handler(Exception, handle_unexpected_error)

    # Open: the Docker healthcheck sends no user, and a client lists the users to choose one.
    app.include_router(routes_health.router, prefix="/api")
    app.include_router(routes_users.router, prefix="/api")
    # Everything else needs a known user. Set here rather than per route, so a new route cannot
    # forget it; routes that use the user declare `CurrentUser` too, which FastAPI resolves once.
    authenticated = [Depends(get_user)]
    unauthenticated: dict[int | str, dict[str, Any]] = {
        401: {"model": ErrorResponse, "description": "No `X-User` header, or an unknown user."}
    }
    for module in (routes_config, routes_ask, routes_documents):
        app.include_router(
            module.router, prefix="/api", dependencies=authenticated, responses=unauthenticated
        )
    return app
