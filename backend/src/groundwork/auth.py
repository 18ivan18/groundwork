"""Who is asking. Authentication is a plain `X-User` header naming a demo user (D23): the point of
the demo is what each user may retrieve, not how they sign in. Swapping in real authentication
means another Authenticator; nothing downstream reads the header."""

from collections.abc import Mapping
from typing import Protocol

from groundwork.domain.users import User

USER_HEADER = "X-User"


class UnauthenticatedError(Exception):
    """No user, or one that does not exist. The message is safe to show."""


class Authenticator(Protocol):
    def authenticate(self, headers: Mapping[str, str]) -> User: ...


class HeaderAuthenticator:
    def __init__(self, users: Mapping[str, User]) -> None:
        self._users = users

    def authenticate(self, headers: Mapping[str, str]) -> User:
        user_id = headers.get(USER_HEADER.lower()) or headers.get(USER_HEADER)
        if not user_id:
            raise UnauthenticatedError(f"Send the {USER_HEADER} header with a user id.")
        user = self._users.get(user_id)
        if user is None:
            raise UnauthenticatedError(f"Unknown user {user_id!r}.")
        return user
