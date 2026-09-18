"""Run one configured strategy from source inventory to notebook result tables."""

from __future__ import annotations

from pathlib import Path

from ....utils.logger import get_custom_logger
from ..model_catalog.exports import write_catalog_artifacts
from ..model_catalog.resolver import resolve_model_catalog
from .artifacts import ArtifactStore
from .context import AnalysisContext
from .planner import build_plan
from .resources import ResourceManager
from .strategy_loader import load_strategy

logger = get_custom_logger(__name__)


def _resolve_output_directories(settings: dict, root: Path, default_precompute_directory: Path) -> None:
    """Make result paths independent from the caller's current directory.

    :param settings: Loaded strategy settings, mutated only for runtime path values.
    :type settings: dict
    :param root: Repository root resolved by the domain settings loader.
    :type root: pathlib.Path
    :param default_precompute_directory: Control-artifact directory beside the
        settings file, used only when the YAML does not configure one.
    :type default_precompute_directory: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    precompute = settings.setdefault("precompute", {})
    output = Path(precompute.get("output_directory", default_precompute_directory))
    precompute["output_directory"] = str(output if output.is_absolute() else root / output)
    for configured in settings.get("analyses", {}).values():
        if "output_directory" not in configured:
            continue
        output = Path(configured["output_directory"])
        configured["output_directory"] = str(output if output.is_absolute() else root / output)


def run_precompute(
    settings_path: Path | str,
    strategy_identifier: str,
    *,
    allow_cpu: bool = False,
    dry_run: bool = False,
) -> AnalysisContext:
    """Resolve one experiment and execute its complete declared precompute strategy.

    :param settings_path: Analysis YAML beside the existing notebook collection.
    :type settings_path: pathlib.Path | str
    :param strategy_identifier: Registered strategy name below ``strategies``.
    :type strategy_identifier: str
    :param allow_cpu: Permit a deliberate CPU fallback where the domain strategy
        supports it.
    :type allow_cpu: bool
    :param dry_run: Validate models, folders, and stage order without computation.
    :type dry_run: bool
    :return: Fully resolved context, useful to focused integration tests.
    :rtype: AnalysisContext
    """
    path = Path(settings_path).resolve()
    strategy = load_strategy(strategy_identifier)
    settings = strategy.load_settings(str(path))
    settings["settings_path"] = str(path)
    _resolve_output_directories(settings, Path(settings["repository_root"]), path.parent / "precompute")
    configured_strategy = settings.get("precompute", {}).get("strategy")
    if configured_strategy and configured_strategy != strategy_identifier:
        raise ValueError(
            f"Settings select strategy '{configured_strategy}', not '{strategy_identifier}'."
        )

    inventory_result = strategy.inventory(settings)
    inventory = inventory_result[0] if isinstance(inventory_result, tuple) else inventory_result
    catalog = resolve_model_catalog(settings, inventory)
    settings["_resolved_model_catalog"] = catalog.records
    store = ArtifactStore(Path(settings["precompute"]["output_directory"]))
    context = AnalysisContext(settings, catalog, store, ResourceManager(), allow_cpu=allow_cpu)
    # Domain modules receive this opaque service through settings during migration.
    # It keeps decoded/model resources shared without coupling their calculations to
    # the orchestration package's concrete classes.
    settings["_precompute_resources"] = context.resources
    plan = build_plan(strategy, context)
    store.prepare(context, strategy)
    write_catalog_artifacts(store.root, catalog)
    store.write_plan(strategy, [stage.name for stage in plan.stages])
    logger.info("Precompute strategy '%s': %s stage(s) validated.", strategy.name, len(plan.stages))
    if dry_run:
        return context

    try:
        for stage in plan.stages:
            logger.info("Running precompute stage '%s'.", stage.name)
            stage.run(context)
            for spec in stage.provides:
                store.verify(context, spec)
    finally:
        context.resources.clear()
    return context
