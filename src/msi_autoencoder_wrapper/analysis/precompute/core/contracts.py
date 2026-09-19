"""Typed contracts shared by all analysis precompute strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .context import AnalysisContext


@dataclass(frozen=True)
class ArtifactSpec:
    """One logical resource supplied by a precompute plugin.

    :param name: Stable resource name used by plugin dependency declarations.
    :type name: str
    :param analysis_name: Analysis YAML key owning the output directory, or ``None``
        for a shared artifact such as inference cache.
    :type analysis_name: str | None
    :param required_files: File names that must exist after a successful plugin run.
    :type required_files: tuple[str, ...]
    :param root_setting: Settings key holding the path for a shared artifact when
        ``analysis_name`` is ``None``.
    :type root_setting: str | None
    :param path_kind: Whether ``root_setting`` identifies a directory or one output
        file. Directory artifacts may declare ``required_files`` below their root;
        file artifacts verify the configured path itself.
    :type path_kind: typing.Literal["directory", "file"]
    """

    name: str
    analysis_name: str | None = None
    required_files: tuple[str, ...] = ()
    root_setting: str | None = None
    path_kind: Literal["directory", "file"] = "directory"


@dataclass(frozen=True)
class AnalysisPlugin:
    """Adapter exposing one domain analysis to the common precompute runner.

    The plugin owns neither model selection nor global scheduling.  It receives an
    already resolved :class:`AnalysisContext`, computes its domain-specific tables,
    and returns only after writing its declared outputs through the existing domain
    implementation.

    :param name: Stable plugin name.
    :type name: str
    :param requires: Logical artifacts required before this plugin may run.
    :type requires: tuple[str, ...]
    :param provides: Logical artifacts supplied by this plugin.
    :type provides: tuple[ArtifactSpec, ...]
    :param run: Domain computation entry point.
    :type run: collections.abc.Callable[[AnalysisContext], None]
    :param enabled_when_configured: Skip the plugin when its analysis YAML section is
        absent. Shared plugins set this to ``False``.
    :type enabled_when_configured: bool
    :param is_enabled_by: Optional strategy-level enablement predicate. This is used
        for shared artifacts configured outside the ``analyses`` block.
    :type is_enabled_by: collections.abc.Callable[[AnalysisContext], bool] | None
    """

    name: str
    requires: tuple[str, ...]
    provides: tuple[ArtifactSpec, ...]
    run: Callable[["AnalysisContext"], None]
    enabled_when_configured: bool = True
    is_enabled_by: Callable[["AnalysisContext"], bool] | None = None

    def is_enabled(self, context: "AnalysisContext") -> bool:
        """Return whether this plugin is enabled by the current YAML settings."""
        if self.is_enabled_by is not None:
            return self.is_enabled_by(context)
        if not self.enabled_when_configured:
            return True
        return any(spec.analysis_name in context.settings.get("analyses", {}) for spec in self.provides)


@dataclass(frozen=True)
class PrecomputeStrategy:
    """One complete, ordered precompute workflow for a notebook collection.

    :param name: Fully qualified strategy identifier selected in YAML.
    :type name: str
    :param model_type: Model family served by the strategy, for example
        ``autoencoder``.
    :type model_type: str
    :param stages: Explicitly ordered domain plugins. The planner validates this
        order; it does not silently invent a different scientific workflow.
    :type stages: collections.abc.Sequence[AnalysisPlugin]
    :param load_settings: Function resolving settings paths and strategy defaults.
    :type load_settings: collections.abc.Callable[[str], dict]
    :param inventory: Function loading the raw model inventory from configured sources.
    :type inventory: collections.abc.Callable[[dict], object]
    """

    name: str
    model_type: str
    stages: Sequence[AnalysisPlugin]
    load_settings: Callable[[str], dict]
    inventory: Callable[[dict], object]
