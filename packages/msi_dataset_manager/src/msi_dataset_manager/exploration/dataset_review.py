"""Local, review-oriented analysis of already discovered dataset records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetReviewProfile:
    """Configure advisory and explicit dataset-review rules.

    :param low_pixel_threshold: Pixel-count threshold used only for an
        advisory flag.
    :type low_pixel_threshold: int | None
    :param morphology_pattern: Regex marking a possibly regional acquisition.
    :type morphology_pattern: str | None
    :param explicit_regional_names: Dataset names approved for automatic
        regional-fragment exclusion.
    :type explicit_regional_names: frozenset[str]
    """

    low_pixel_threshold: Optional[int] = None
    morphology_pattern: Optional[str] = None
    explicit_regional_names: frozenset[str] = frozenset()


_BRAIN_PROFILE = DatasetReviewProfile(
    low_pixel_threshold=500,
    morphology_pattern=(
        r"(?:purkinje|granular_layer|molecular_layer|fibers_layer|_hip_|_hip$|"
        r"_cer_|_cer$|cerebellum|hippocamp|striatum|olfactory|midbrain|"
        r"substantia|hypothalamus|thalamus|cortex(?!.{0,15}coronal))"
    ),
    explicit_regional_names=frozenset(
        {
            "molecular_layer_brain_mouse",
            "purkinje_fibers_mouse_brain",
            "granular_layer_mouse_brain",
            "fibers_layer_mouse_brain",
        }
    ),
)

_LIVER_PROFILE = DatasetReviewProfile(
    low_pixel_threshold=200,
    morphology_pattern=r"(?:\blobe\b|periportal|pericentral|zonation|capsule|\bportal\b)",
)

_PROFILES = {"brain": _BRAIN_PROFILE, "liver": _LIVER_PROFILE}

_TECHNICAL_TOKEN_PATTERNS = (
    (r"^\d{4}-\d{2}-\d{2}[_ ]", ""),
    (r"^\d{8}_+", ""),
    (r"[-_]?\d+ ?ppm\b", ""),
    (r"\btic\b", ""),
    (r"_(?:aq_ml|aq|ml)$", ""),
    (r"-total ion count$", ""),
    (r" - root mean square$", ""),
)
_SERIES_TOKEN_PATTERNS = _TECHNICAL_TOKEN_PATTERNS + (
    (r"_replicate\d+$", ""),
    (r"_s\d+$", ""),
    (r"_\d+$", ""),
)
_KEEPER_PENALTY = re.compile(
    r"(?:_ml$|_v$|ppm$|-total ion count$)", re.IGNORECASE
)
_QC_MZ_SHIFT = re.compile(r"null_mz_shift", re.IGNORECASE)


@dataclass
class DatasetReview:
    """Table and explicit exclusion recommendations for one explorer result."""

    table: pd.DataFrame
    _rule_ids: Mapping[str, frozenset[str]]

    @property
    def available_rules(self) -> tuple[str, ...]:
        """Return names accepted by :meth:`exclusion_ids`."""
        return tuple(self._rule_ids)

    def exclusion_ids(self, rules: Iterable[str]) -> list[str]:
        """Return deterministic unique IDs recommended by selected rules.

        :param rules: Review-rule names from :attr:`available_rules`.
        :type rules: Iterable[str]
        :return: Dataset IDs selected by at least one requested rule.
        :rtype: list[str]
        :raises ValueError: If a rule is unknown.
        """
        selected: set[str] = set()
        for rule in rules:
            if rule not in self._rule_ids:
                raise ValueError(
                    f"Unknown dataset review rule '{rule}'. "
                    f"Available rules: {sorted(self._rule_ids)}"
                )
            selected.update(self._rule_ids[rule])
        return sorted(selected)

    def summary(self) -> pd.DataFrame:
        """Return one count row per exclusion recommendation rule."""
        return pd.DataFrame(
            [
                {"rule": rule, "dataset_count": len(dataset_ids)}
                for rule, dataset_ids in self._rule_ids.items()
            ]
        )


def resolve_review_profile(
    profile: str | DatasetReviewProfile | Mapping[str, Any] | None,
) -> DatasetReviewProfile:
    """Resolve a built-in or notebook-supplied review profile."""
    if profile is None:
        return DatasetReviewProfile()
    if isinstance(profile, DatasetReviewProfile):
        return profile
    if isinstance(profile, str):
        try:
            return _PROFILES[profile.casefold()]
        except KeyError as error:
            raise ValueError(
                f"Unknown dataset review profile '{profile}'. "
                f"Available profiles: {sorted(_PROFILES)}"
            ) from error
    if isinstance(profile, Mapping):
        return DatasetReviewProfile(
            low_pixel_threshold=profile.get("low_pixel_threshold"),
            morphology_pattern=profile.get("morphology_pattern"),
            explicit_regional_names=frozenset(
                str(name).casefold()
                for name in profile.get("explicit_regional_names", ())
            ),
        )
    raise TypeError("profile must be a name, DatasetReviewProfile, mapping, or None")


def build_dataset_review(
    records: pd.DataFrame,
    *,
    profile: DatasetReviewProfile,
    mz_precision: int,
) -> DatasetReview:
    """Annotate one already-filtered result table without changing it."""
    if "dataset_id" not in records or "name" not in records:
        raise ValueError("Dataset review requires dataset_id and name columns.")
    if mz_precision < 0:
        raise ValueError("mz_precision must be non-negative.")

    table = records.copy().reset_index(drop=True)
    names = table["name"].fillna("").astype(str)
    table["technical_name_key"] = names.map(_technical_name_key)
    table["biological_series_id"] = names.map(_biological_series_key)
    _annotate_duplicate_clusters(table, mz_precision=mz_precision)

    table["mz_shift_qc_variant"] = names.str.contains(_QC_MZ_SHIFT, na=False)
    if profile.morphology_pattern is None:
        table["morphology_hint"] = "not_configured"
    else:
        regional = names.str.contains(profile.morphology_pattern, case=False, regex=True)
        table["morphology_hint"] = regional.map(
            {True: "regional_or_microregion", False: "whole_section_likely"}
        )
    normalized_names = names.str.casefold()
    table["explicit_regional_fragment"] = normalized_names.isin(
        profile.explicit_regional_names
    )

    pixel_count = pd.to_numeric(table.get("pixel_count"), errors="coerce")
    table["low_pixel_flag"] = (
        False
        if profile.low_pixel_threshold is None
        else pixel_count.lt(profile.low_pixel_threshold).fillna(False)
    )

    rule_ids = {
        "high_confidence_duplicates": frozenset(
            table.loc[table["duplicate_excluded"], "dataset_id"].astype(str)
        ),
        "mz_shift_qc_variants": frozenset(
            table.loc[table["mz_shift_qc_variant"], "dataset_id"].astype(str)
        ),
        "explicit_regional_fragments": frozenset(
            table.loc[table["explicit_regional_fragment"], "dataset_id"].astype(str)
        ),
    }
    return DatasetReview(table=table, _rule_ids=rule_ids)


def _annotate_duplicate_clusters(table: pd.DataFrame, *, mz_precision: int) -> None:
    """Add conservative technical-duplicate annotations in place."""
    pixel_count = pd.to_numeric(table.get("pixel_count"), errors="coerce")
    mz_min = pd.to_numeric(table.get("mz_min"), errors="coerce")
    mz_max = pd.to_numeric(table.get("mz_max"), errors="coerce")
    valid = pixel_count.notna() & mz_min.notna() & mz_max.notna()
    table["duplicate_cluster_id"] = pd.NA
    table["duplicate_cluster_size"] = 0
    table["duplicate_confidence"] = "not_clustered"
    table["recommended_keeper_dataset_id"] = pd.NA
    table["duplicate_excluded"] = False
    if not valid.any():
        return

    candidates = pd.DataFrame(
        {
            "pixel_count": pixel_count.loc[valid],
            "mz_min": mz_min.loc[valid].round(mz_precision),
            "mz_max": mz_max.loc[valid].round(mz_precision),
        }
    )
    grouped = candidates.groupby(["pixel_count", "mz_min", "mz_max"], sort=True)
    for cluster_number, (_, indices) in enumerate(grouped.groups.items(), start=1):
        indices = list(indices)
        size = len(indices)
        if size < 2:
            continue
        cluster_id = f"technical-{cluster_number:04d}"
        table.loc[indices, "duplicate_cluster_id"] = cluster_id
        table.loc[indices, "duplicate_cluster_size"] = size
        residuals = set(table.loc[indices, "technical_name_key"])
        if len(residuals) != 1:
            table.loc[indices, "duplicate_confidence"] = "ambiguous_shared_template"
            continue
        table.loc[indices, "duplicate_confidence"] = "high_confidence_duplicate"
        keeper = _pick_keeper(table.loc[indices])
        table.loc[indices, "recommended_keeper_dataset_id"] = keeper
        table.loc[
            table.index.isin(indices) & table["dataset_id"].ne(keeper),
            "duplicate_excluded",
        ] = True


def _pick_keeper(group: pd.DataFrame) -> str:
    """Prefer a name without known technical suffixes, then dataset ID."""
    ranked = group.assign(
        _technical_penalty=group["name"].fillna("").astype(str).str.contains(
            _KEEPER_PENALTY, regex=True
        ).astype(int)
    ).sort_values(["_technical_penalty", "dataset_id"])
    return str(ranked.iloc[0]["dataset_id"])


def _technical_name_key(name: str) -> str:
    """Remove only known processing tokens from a dataset name."""
    value = name.casefold()
    for pattern, replacement in _TECHNICAL_TOKEN_PATTERNS:
        value = re.sub(pattern, replacement, value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _biological_series_key(name: str) -> str:
    """Derive a conservative, name-based candidate series identifier."""
    value = name.casefold()
    for pattern, replacement in _SERIES_TOKEN_PATTERNS:
        value = re.sub(pattern, replacement, value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")
