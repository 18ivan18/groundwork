"""The demo users, for the user picker, and the current user."""

from fastapi import APIRouter, Request

from groundwork.api.deps import CurrentUser
from groundwork.api.schemas import ErrorResponse, UserResponse
from groundwork.domain.users import User

router = APIRouter(tags=["users"])


def _to_response(user: User) -> UserResponse:
    return UserResponse(
        user_id=user.user_id, display_name=user.display_name, groups=sorted(user.groups)
    )


@router.get(
    "/users",
    summary="List the demo users",
    description=(
        "Open to anyone, so a client can offer the users before one is chosen. Authentication is "
        "the `X-User` header naming one of them: there are no passwords in this demo."
    ),
)
async def list_users(request: Request) -> list[UserResponse]:
    return [_to_response(user) for user in request.app.state.users.values()]


@router.get(
    "/me",
    summary="The current user",
    description="Who the `X-User` header names.",
    responses={401: {"model": ErrorResponse, "description": "No `X-User`, or an unknown user."}},
)
async def me(user: CurrentUser) -> UserResponse:
    return _to_response(user)
