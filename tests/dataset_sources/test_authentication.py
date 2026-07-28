"""Tests for session-only METASPACE authentication."""

from __future__ import annotations

import pytest

from msi_autoencoder_wrapper.dataset_management.sources.strategies.metaspace_authentication import (
    METASPACE_API_KEY_ENV,
    metaspace_client_options,
    validate_metaspace_session,
)
from msi_autoencoder_wrapper.dataset_management.sources.strategies.metaspace import (
    MetaspaceDatasetSource,
)
from msi_autoencoder_wrapper.utils.exceptions import ProjectConfigError


class FakeAuthenticatedClient:
    """Minimal client reporting a successful login."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def logged_in(self) -> bool:
        return True


def test_session_key_is_injected_without_modifying_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment supplies runtime auth while explicit options are copied."""
    monkeypatch.setenv(METASPACE_API_KEY_ENV, "session-secret")
    original = {"host": "https://metaspace.example"}

    resolved = metaspace_client_options(original)

    assert resolved == {
        "host": "https://metaspace.example",
        "api_key": "session-secret",
    }
    assert original == {"host": "https://metaspace.example"}


def test_session_validation_requires_and_verifies_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing keys fail and authenticated runtime-only keys succeed."""
    monkeypatch.delenv(METASPACE_API_KEY_ENV, raising=False)
    with pytest.raises(ProjectConfigError, match=METASPACE_API_KEY_ENV):
        validate_metaspace_session(FakeAuthenticatedClient)

    monkeypatch.setenv(METASPACE_API_KEY_ENV, "session-secret")
    assert validate_metaspace_session(FakeAuthenticatedClient) is True


def test_source_configuration_redacts_explicit_secrets() -> None:
    """Serializable component configuration never exposes credentials."""
    source = MetaspaceDatasetSource(
        client=object(),
        client_options={"api_key": "secret", "password": "secret", "host": "example"},
    )

    assert source.get_config()["client_options"] == {
        "api_key": "***",
        "password": "***",
        "host": "example",
    }
