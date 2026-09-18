"""Manifest-inventory selectors and stable display/visualization aliases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ....visualization.theme import THEME_PRESETS


@dataclass(frozen=True)
class ResolvedModelCatalog:
    """Concrete model records selected by an analysis YAML.

    :param records: One row per selected model/repetition, retaining the original
        inventory columns plus alias and visualization columns.
    :type records: pandas.DataFrame
    :param styles: One row per logical alias with its display and plot encoding.
    :type styles: pandas.DataFrame
    :param groups: Named YAML groups resolved to ordered model aliases.
    :type groups: dict[str, tuple[str, ...]]
    """

    records: pd.DataFrame
    styles: pd.DataFrame
    groups: dict[str, tuple[str, ...]]

    def aliases(self, group: str | None = None) -> tuple[str, ...]:
        """Return aliases in declared display order, optionally for a named group."""
        if group is not None:
            return self.groups[group]
        return tuple(self.styles.sort_values("display_order").model_alias)

    def select(self, aliases: list[str] | tuple[str, ...], *, repetition: int | None = None) -> pd.DataFrame:
        """Return selected records for aliases and an optional exact repetition."""
        frame = self.records[self.records.model_alias.isin(aliases)].copy()
        if repetition is not None:
            frame = frame[frame.repetition == repetition].copy()
        return frame.sort_values(["display_order", "repetition", "model_alias"])


def _matches(frame: pd.DataFrame, selector: dict[str, Any], alias: str) -> pd.DataFrame:
    """Apply exact, generic inventory-column matching for one YAML alias."""
    if not selector:
        raise ValueError(f"Model alias '{alias}' needs a nonempty 'select' mapping.")
    selected = frame
    unknown = sorted(set(selector) - set(frame.columns))
    if unknown:
        raise ValueError(
            f"Model alias '{alias}' selects unknown inventory column(s): {unknown}. "
            f"Available columns include: {sorted(frame.columns)[:12]}."
        )
    for column, expected in selector.items():
        values = expected if isinstance(expected, list) else [expected]
        selected = selected[selected[column].isin(values)]
    return selected


def _style(alias: str, definition: dict[str, Any], position: int) -> dict[str, Any]:
    """Resolve one alias's stable display fields without interpreting training tags."""
    visual = dict(definition.get("visualization") or {})
    palette = THEME_PRESETS["diagnostic_light"].model_palette
    tags = dict(definition.get("tags") or {})
    return {
        "model_alias": alias,
        "display_label": str(definition.get("display_label", alias)),
        "display_order": int(visual.get("order", position)),
        "color": str(visual.get("color", palette[position % len(palette)])),
        "line_style": str(visual.get("line_style", "solid")),
        "marker": str(visual.get("marker", "o")),
        "tags_json": json.dumps(tags, sort_keys=True),
    }


def _resolve_groups(settings: dict[str, Any], aliases: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Validate explicit model groups used by particular notebook analyses."""
    known = set(aliases)
    groups: dict[str, tuple[str, ...]] = {}
    for name, members in (settings.get("groups") or {}).items():
        if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
            raise ValueError(f"Group '{name}' must be a list of model aliases.")
        unknown = sorted(set(members) - known)
        if unknown:
            raise ValueError(f"Group '{name}' references unknown model aliases: {unknown}.")
        groups[name] = tuple(members)
    return groups


def resolve_model_catalog(settings: dict[str, Any], inventory: pd.DataFrame) -> ResolvedModelCatalog:
    """Resolve YAML model aliases against the combined source inventory.

    A selector is intentionally generic exact matching over inventory columns. It can
    use deterministic ``grid_id``/``task_id``/``model_id`` values today, and new model
    families can expose additional columns tomorrow without adding special branches for
    contrastive, contractive, Jaccard, or any other training condition.

    :param settings: Fully resolved analysis settings.
    :type settings: dict[str, typing.Any]
    :param inventory: Unified campaign inventory returned by the selected strategy.
    :type inventory: pandas.DataFrame
    :return: Selected records, display styles, and alias groups.
    :rtype: ResolvedModelCatalog
    :raises ValueError: If selectors are ambiguous, empty, or reuse a concrete model.
    """
    definitions = settings.get("models") or {}
    if not definitions:
        # Legacy settings remain runnable while strategies are migrated. Their raw
        # inventory identifiers become aliases; no text label is treated as unique.
        records = inventory.copy()
        records["model_alias"] = records["model_id"]
        records["display_label"] = records["label"]
        records["display_order"] = range(len(records))
        records["color"] = ""
        records["line_style"] = "solid"
        records["marker"] = "o"
        records["tags_json"] = "{}"
        styles = records[["model_alias", "display_label", "display_order", "color", "line_style", "marker", "tags_json"]].drop_duplicates()
        return ResolvedModelCatalog(records, styles, {})

    if not isinstance(definitions, dict):
        raise ValueError("'models' must be a mapping from a stable alias to its selector.")
    ready = inventory[inventory.ready].copy() if "ready" in inventory else inventory.copy()
    selected_frames: list[pd.DataFrame] = []
    style_rows: list[dict[str, Any]] = []
    used_model_ids: dict[str, str] = {}
    for position, (alias, definition) in enumerate(definitions.items()):
        if not isinstance(alias, str) or not isinstance(definition, dict):
            raise ValueError("Every 'models' entry must map a string alias to a mapping.")
        chosen = _matches(ready, dict(definition.get("select") or {}), alias).copy()
        if chosen.empty:
            raise ValueError(f"Model alias '{alias}' matched no ready model records.")
        duplicate_repetitions = chosen.groupby("repetition").size()
        ambiguous = duplicate_repetitions[duplicate_repetitions != 1]
        if not ambiguous.empty:
            candidates = chosen[["source", "grid_id", "task_id", "model_id", "repetition"]].to_dict("records")
            raise ValueError(
                f"Model alias '{alias}' must resolve exactly once per repetition; "
                f"ambiguous repetitions are {ambiguous.index.tolist()}. Candidates: {candidates}"
            )
        for model_id in chosen.model_id:
            previous = used_model_ids.get(model_id)
            if previous is not None:
                raise ValueError(
                    f"Concrete model '{model_id}' is selected by both '{previous}' and '{alias}'."
                )
            used_model_ids[model_id] = alias
        style = _style(alias, definition, position)
        chosen["raw_label"] = chosen["label"]
        for key, value in style.items():
            chosen[key] = value
        chosen["label"] = chosen["display_label"]
        selected_frames.append(chosen)
        style_rows.append(style)

    styles = pd.DataFrame(style_rows).sort_values(["display_order", "model_alias"]).reset_index(drop=True)
    records = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["display_order", "repetition", "model_alias"]
    ).reset_index(drop=True)
    groups = _resolve_groups(settings, tuple(styles.model_alias))
    return ResolvedModelCatalog(records, styles, groups)
