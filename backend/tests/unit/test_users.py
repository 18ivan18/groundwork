from pathlib import Path

import pytest

from groundwork.domain.models import EVERYONE
from groundwork.domain.users import User, UsersConfigError, load_users

REPO_USERS = Path(__file__).parents[2] / "conf" / "users.yaml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "users.yaml"
    path.write_text(text)
    return path


def test_the_committed_users_have_their_groups() -> None:
    users = load_users(REPO_USERS)

    assert {user_id: set(user.groups) for user_id, user in users.items()} == {
        "alice": {"everyone"},
        "hana": {"everyone", "hr"},
        "filip": {"everyone", "finance"},
        "admin": {"everyone", "hr", "finance", "admin"},
    }


def test_everyone_is_added_when_not_listed(tmp_path: Path) -> None:
    users = load_users(write(tmp_path, "users:\n  bo:\n    display_name: Bo\n    groups: [hr]\n"))

    assert users["bo"] == User("bo", "Bo", frozenset({EVERYONE, "hr"}))


@pytest.mark.parametrize(
    "text",
    [
        "users: {}\n",
        "people:\n  bo: {display_name: Bo, groups: []}\n",
        "users:\n  Bo Smith: {display_name: Bo, groups: []}\n",
        "users:\n  bo: {display_name: Bo, groups: [Human Resources]}\n",
        "users:\n  bo: {display_name: '', groups: []}\n",
        "users:\n  bo: {display_name: Bo, groups: [], admin: true}\n",
        "users: [\n",
    ],
)
def test_a_malformed_file_is_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(UsersConfigError):
        load_users(write(tmp_path, text))


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UsersConfigError, match=r"users\.yaml"):
        load_users(tmp_path / "users.yaml")
