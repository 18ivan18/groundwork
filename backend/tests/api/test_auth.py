"""Every route but the open ones needs a known user in `X-User` (#38)."""

import pytest
from api_helpers import as_user
from fastapi import FastAPI
from fastapi.testclient import TestClient

OPEN = {"/api/health", "/api/users"}


def routes(app: FastAPI) -> list[tuple[str, str]]:
    schema = TestClient(app).get("/api/openapi.json").json()
    return [
        (method, path.replace("{job_id}", "some-job"))
        for path, operations in schema["paths"].items()
        for method in operations
    ]


@pytest.mark.parametrize("headers", [{}, {"X-User": "mallory"}, {"X-User": ""}])
def test_every_route_but_the_open_ones_refuses_an_unknown_user(
    app: FastAPI, headers: dict[str, str]
) -> None:
    client = TestClient(app, headers=headers)
    checked = []
    for method, path in routes(app):
        if path in OPEN:
            continue
        response = client.request(method, path)
        assert response.status_code == 401, f"{method} {path}"
        assert "X-User" in response.json()["detail"] or "Unknown user" in response.json()["detail"]
        checked.append(path)
    assert "/api/ask" in checked and "/api/documents" in checked


def test_the_open_routes_need_no_user(app: FastAPI) -> None:
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert {user["user_id"] for user in client.get("/api/users").json()} == {
        "alice",
        "hana",
        "filip",
        "admin",
    }


def test_me_is_the_user_the_header_names(app: FastAPI) -> None:
    body = as_user(app, "hana").get("/api/me").json()

    assert body == {
        "user_id": "hana",
        "display_name": "Hana (HR)",
        "groups": ["everyone", "hr", "private:hana"],
    }
