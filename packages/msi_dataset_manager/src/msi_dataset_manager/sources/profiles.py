"""CSV-backed provider profiles and quota-aware source rotation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from ..utils.exceptions import DownloadLimitError, raise_validation_error
from ..utils.logger import get_custom_logger
from .base import DatasetSource


logger = get_custom_logger(__name__)


def read_source_profiles(path: Path | str) -> List[Dict[str, str]]:
    """Read provider profiles from CSV.

    :param path: CSV file containing the mandatory ``key`` column. All other
        columns are retained as optional operator metadata.
    :type path: pathlib.Path | str
    :return: Non-empty profiles in file order.
    :rtype: List[Dict[str, str]]
    :raises ValidationError: If the file has no ``key`` column or contains an
        empty key.
    """
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "key" not in reader.fieldnames:
            raise_validation_error(
                "SourceProfiles",
                "The profiles CSV must contain a 'key' column.",
            )
        profiles = []
        for row_number, row in enumerate(reader, start=2):
            profile = {
                str(name): str(value or "").strip()
                for name, value in row.items()
                if name is not None
            }
            if not profile["key"]:
                raise_validation_error(
                    "SourceProfiles",
                    f"The profile key is empty in CSV row {row_number}.",
                )
            profiles.append(profile)
    if not profiles:
        raise_validation_error("SourceProfiles", "The profiles CSV is empty.")
    return profiles


class RotatingDatasetSource:
    """Delegate provider calls and rotate credentials after quota exhaustion.

    :param profiles: Profiles returned by :func:`read_source_profiles`.
    :type profiles: List[Mapping[str, str]]
    :param source_factory: Factory receiving the API key and complete profile.
    :type source_factory: Callable[[str, Mapping[str, str]], DatasetSource]
    :param initial_source: Optional source already configured with the first
        profile, avoiding a duplicate provider-client construction.
    :type initial_source: DatasetSource | None

    Only :class:`DownloadLimitError` advances the profile. Authentication,
    validation, and network failures remain visible to the caller.
    """

    def __init__(
        self,
        profiles: List[Mapping[str, str]],
        source_factory: Callable[[str, Mapping[str, str]], DatasetSource],
        initial_source: DatasetSource | None = None,
    ) -> None:
        self._profiles = [dict(profile) for profile in profiles]
        self._source_factory = source_factory
        self._index = 0
        self._source = initial_source or self._make_source()

    @property
    def source(self) -> DatasetSource:
        """Return the source configured with the active profile."""
        return self._source

    @property
    def exhausted_profile_count(self) -> int:
        """Return the number of profiles rejected due to quota exhaustion."""
        return self._index

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a source method, retrying with the next profile on quota errors."""
        while True:
            try:
                return getattr(self._source, method_name)(*args, **kwargs)
            except DownloadLimitError:
                if self._index + 1 >= len(self._profiles):
                    raise
                self._index += 1
                logger.warning(
                    "Provider quota exhausted for profile %s; switching to profile %s",
                    self._index,
                    self._index + 1,
                )
                self._source = self._make_source()

    def _make_source(self) -> DatasetSource:
        profile = self._profiles[self._index]
        return self._source_factory(profile["key"], profile)
