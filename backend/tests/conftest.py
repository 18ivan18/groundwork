"""Shared fixtures."""

import pytest

from groundwork.config import Settings
from groundwork.ingestion.corpora_bundle import LOCAL_CORPORA_ENV


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's shell variables from changing test results."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    monkeypatch.delenv(LOCAL_CORPORA_ENV, raising=False)
