"""Users and the groups they belong to, from conf/users.yaml."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from groundwork.domain.models import EVERYONE, UNRESTRICTED, Access

_Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=40)]


# Members see every user's jobs and may delete any document they can read, and documents in
# protected collections. Reading is still by group: admin is not a superuser for retrieval.
ADMIN = "admin"


@dataclass(frozen=True)
class User:
    user_id: str
    display_name: str
    groups: frozenset[str]

    @property
    def is_admin(self) -> bool:
        return ADMIN in self.groups


class UsersConfigError(ValueError):
    """conf/users.yaml is missing or malformed."""


class _UserEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    groups: list[_Slug]


class _UsersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    users: dict[_Slug, _UserEntry]


def private_group(user_id: str) -> str:
    """The group only this user is in: the default for what they upload."""
    return f"private:{user_id}"


def load_users(path: Path) -> Mapping[str, User]:
    """Users by id. Each user is in `everyone` and their own private group, whether or not the
    file lists them."""
    try:
        parsed = _UsersFile.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise UsersConfigError(f"{path}: {error}") from error
    if not parsed.users:
        raise UsersConfigError(f"{path}: no users")
    return {
        user_id: User(
            user_id,
            entry.display_name,
            frozenset({EVERYONE, private_group(user_id), *entry.groups}),
        )
        for user_id, entry in parsed.users.items()
    }


def access_for(users: Mapping[str, User], user_id: str | None) -> Access:
    """What `user_id` may read; everything when None. For scripts and the evaluation: the API
    gets its user from the Authenticator."""
    if user_id is None:
        return UNRESTRICTED
    if user_id not in users:
        raise UsersConfigError(f"Unknown user {user_id!r}. Known: {', '.join(sorted(users))}.")
    return Access.of(users[user_id].groups)
