"""Tests for quota-aware provider profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

from msi_dataset_manager.sources.base import DatasetSource
from msi_dataset_manager.sources.profiles import (
    RotatingDatasetSource,
    read_source_profiles,
)
from msi_dataset_manager.utils.exceptions import DownloadLimitError, ValidationError


class ProfileSource(DatasetSource):
    """Minimal source exposing one quota-sensitive operation."""

    source_name = "profile"

    def __init__(self, key: str) -> None:
        self.key = key
        self._config = {}

    def get_available_filters(self) -> Dict[str, Any]:
        return {}

    def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        return []

    def get_accepted_records(self) -> List[Dict[str, Any]]:
        return []

    def get_rejected_records(self) -> List[Dict[str, Any]]:
        return []

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        return {"dataset_id": dataset_id}

    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return []

    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        if self.key == "limited":
            raise DownloadLimitError("quota")
        return Path(destination)


def test_profile_csv_requires_key_and_preserves_arbitrary_metadata(tmp_path: Path) -> None:
    """Only key has schema meaning; comments and ownership columns survive."""
    profiles_path = tmp_path / "profiles.csv"
    profiles_path.write_text(
        "key,comment,account_owner\nlimited,first quota,Alice\nactive,,Bob\n",
        encoding="utf-8",
    )

    profiles = read_source_profiles(profiles_path)

    assert profiles[0] == {
        "key": "limited",
        "comment": "first quota",
        "account_owner": "Alice",
    }
    assert profiles[1]["account_owner"] == "Bob"


def test_profile_csv_rejects_missing_or_empty_keys(tmp_path: Path) -> None:
    """Invalid credential files fail before any provider call."""
    missing = tmp_path / "missing.csv"
    missing.write_text("comment\nowner\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="'key' column"):
        read_source_profiles(missing)

    empty = tmp_path / "empty.csv"
    empty.write_text("key,comment\n,owner\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="empty"):
        read_source_profiles(empty)


def test_rotating_source_retries_same_operation_with_next_key() -> None:
    """A quota exception advances one profile without swallowing other errors."""
    created: List[str] = []

    def factory(key: str, profile: Mapping[str, str]) -> DatasetSource:
        created.append(key)
        return ProfileSource(key)

    source = RotatingDatasetSource(
        [{"key": "limited", "comment": "first"}, {"key": "active"}],
        factory,
    )

    result = source.call("download_dataset", "dataset", "/tmp/dataset")

    assert result == Path("/tmp/dataset")
    assert created == ["limited", "active"]
    assert source.exhausted_profile_count == 1
