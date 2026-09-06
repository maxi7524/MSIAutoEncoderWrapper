"""Numerically profile and compare two declarative training campaigns."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Callable, Sequence

import torch
import yaml
from scipy.stats import t as student_t
from tqdm.auto import tqdm

from ...runtime import build_plan, load_experiment_config
from ...runtime.planning.resolution import resolve_plan
from ...runtime.workflows.entrypoints import resolve_entrypoint


_SPECTRAL_POWER_ITERATIONS = 3
_TEST_PROBE_BATCHES = 2


@dataclass(frozen=True)
class Measurement:
    """Summary of repeated timing and memory observations."""

    step_id: str
    samples: int
    mean_seconds: float
    standard_deviation_seconds: float
    ci95_low_seconds: float
    ci95_high_seconds: float
    p95_seconds: float
    mean_rss_delta_bytes: float
    mean_cuda_peak_bytes: float
    complexity: str


class BenchmarkProgress:
    """Render overall and current-stage benchmark progress in the terminal."""

    def __init__(self, total_measurements: int) -> None:
        """Initialize the two persistent progress rows.

        :param total_measurements: Total repeated measurements across both campaigns.
        :type total_measurements: int
        """
        self._overall = tqdm(
            total=total_measurements,
            desc="Overall",
            unit="measurement",
            position=0,
            dynamic_ncols=True,
        )
        self._current = tqdm(
            total=1,
            desc="Current",
            unit="sample",
            position=1,
            dynamic_ncols=True,
        )

    def start_step(self, step_id: str, samples: int) -> None:
        """Replace the current-stage row for one repeated operation."""
        self._current.reset(total=samples)
        self._current.set_description(f"Current: {step_id}")

    def advance(self) -> None:
        """Advance both progress rows after one completed observation."""
        self._current.update(1)
        self._overall.update(1)

    def close(self) -> None:
        """Finalize the two terminal progress rows."""
        self._current.close()
        self._overall.close()


def confidence_interval(samples: Sequence[float]) -> tuple[float, float, float, float]:
    """Return mean, sample standard deviation, and a two-sided 95% t interval.

    :param samples: Independent finite observations.
    :type samples: Sequence[float]
    :return: Mean, standard deviation, lower confidence endpoint, upper endpoint.
    :rtype: tuple[float, float, float, float]
    :raises ValueError: If no finite observations are supplied.
    """
    values = [float(value) for value in samples]
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Benchmark samples must be finite non-negative values.")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, mean, mean
    deviation = statistics.stdev(values)
    margin = float(student_t.ppf(0.975, len(values) - 1)) * deviation / math.sqrt(len(values))
    return mean, deviation, mean - margin, mean + margin


def contractive_complexity(params: dict[str, Any] | None, latent_dimension: int) -> str:
    """Describe Jacobian work implied by one contractive batch.

    :param params: Contractive criterion parameters, if the campaign uses one.
    :type params: dict[str, Any] | None
    :param latent_dimension: Dimension ``D`` of the penalized latent space.
    :type latent_dimension: int
    :return: Human-readable operation count for one batch.
    :rtype: str
    """
    if params is None:
        return "O(C_train_step(B, input_shape, model))"
    metric = params.get("penalty_metric", "frobenius")
    if metric in {"spectral", "spectral_plus_hinged"}:
        return (
            f"O({_SPECTRAL_POWER_ITERATIONS}·C_VJP(B,M,D) + "
            f"{_SPECTRAL_POWER_ITERATIONS + 1}·C_JVP(B,M,D))"
        )
    method = params.get("calculation_method", "approximate_hutchinson_vjp")
    derivatives = latent_dimension if method == "exact_autograd_jacobian" else int(
        params.get("num_probes", 1)
    )
    return f"O({derivatives}·C_VJP(B,M,D)) via {derivatives} reverse-mode VJP passes"


def _rss_bytes() -> int:
    """Return current process RSS without making psutil mandatory."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


def _synchronize(device: torch.device | None) -> None:
    """Synchronize CUDA before a wall-clock measurement when applicable."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    step_id: str,
    operation: Callable[[], Any],
    repeats: int,
    device: torch.device | None,
    complexity: str,
    progress: BenchmarkProgress | None = None,
) -> tuple[Measurement, Any]:
    """Measure an operation repeatedly and return its last result.

    :param step_id: Stable report identifier.
    :type step_id: str
    :param operation: Side-effect-contained operation under measurement.
    :type operation: Callable[[], Any]
    :param repeats: Number of independent observations.
    :type repeats: int
    :param device: Optional CUDA device used by the operation.
    :type device: torch.device | None
    :param complexity: Symbolic complexity expression.
    :type complexity: str
    :return: Measurement summary and final operation result.
    :rtype: tuple[Measurement, Any]
    """
    durations: list[float] = []
    rss_deltas: list[int] = []
    cuda_peaks: list[int] = []
    result = None
    if progress is not None:
        progress.start_step(step_id, repeats)
    for _ in range(repeats):
        _synchronize(device)
        if device is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rss_before = _rss_bytes()
        started = time.perf_counter()
        result = operation()
        _synchronize(device)
        durations.append(time.perf_counter() - started)
        rss_deltas.append(max(0, _rss_bytes() - rss_before))
        cuda_peaks.append(
            int(torch.cuda.max_memory_allocated(device))
            if device is not None and device.type == "cuda"
            else 0
        )
        if progress is not None:
            progress.advance()
    mean, deviation, lower, upper = confidence_interval(durations)
    return Measurement(
        step_id=step_id,
        samples=repeats,
        mean_seconds=mean,
        standard_deviation_seconds=deviation,
        ci95_low_seconds=lower,
        ci95_high_seconds=upper,
        p95_seconds=max(durations) if repeats < 20 else float(torch.quantile(torch.tensor(durations), 0.95)),
        mean_rss_delta_bytes=statistics.fmean(rss_deltas),
        mean_cuda_peak_bytes=statistics.fmean(cuda_peaks),
        complexity=complexity,
    ), result


def _first_representative_task(plan: Any) -> dict[str, Any]:
    """Return one resolved task with the runtime state required by test mode."""
    task = asdict(plan.tasks[0])
    task["runtime"] = {
        **task.get("runtime", {}),
        "task_label": "campaign-benchmark",
        "resume": False,
        "checkpoint_path": "",
        "progress_path": "",
    }
    return task


def _latent_dimension(task: dict[str, Any]) -> int:
    """Read the configured latent dimension from one materialized task."""
    architecture = task.get("grid_parameters", {}).get("architectures", {})
    return int(architecture.get("parameters", {}).get("latent_dim", 1))


def _contractive_parameters(task: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the first configured contractive criterion from a task."""
    phases = task.get("parameters", {}).get("training", {}).get("phases", [])
    criterions = phases[0].get("criterions", {}) if phases else {}
    regularization = criterions.get("regularization", {})
    for name, criterion in regularization.items():
        if criterion.get("target", name) == "ContractiveLoss":
            return dict(criterion.get("params", {}))
    return None


def _format_bytes(value: float) -> str:
    """Format a byte estimate using binary units."""
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def _markdown_table(measurements: Sequence[Measurement]) -> str:
    """Render stage measurements as a Markdown table."""
    rows = [
        "| Step ID | Samples | Mean time (s) | 95% CI (s) | P95 time (s) | RSS allocation | CUDA peak allocation | Complexity |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for measurement in measurements:
        rows.append(
            "| {step} | {samples} | {mean:.4f} | [{low:.4f}, {high:.4f}] | "
            "{p95:.4f} | {rss} | {cuda} | {complexity} |".format(
                step=measurement.step_id,
                samples=measurement.samples,
                mean=measurement.mean_seconds,
                low=measurement.ci95_low_seconds,
                high=measurement.ci95_high_seconds,
                p95=measurement.p95_seconds,
                rss=_format_bytes(measurement.mean_rss_delta_bytes),
                cuda=_format_bytes(measurement.mean_cuda_peak_bytes),
                complexity=measurement.complexity,
            )
        )
    return "\n".join(rows)


def _profile_campaign(
    label: str,
    config_path: Path,
    repeats: int,
    device_override: str | None,
    temporary_root: Path,
    progress: BenchmarkProgress,
) -> tuple[list[Measurement], dict[str, Any]]:
    """Profile one campaign through plan resolution and bounded train probes."""
    def load_and_expand() -> tuple[dict[str, Any], Any]:
        config = load_experiment_config(config_path)
        if device_override is not None:
            factory_parameters = config.get("task", {}).get("parameters", {}).get(
                "factory_parameters"
            )
            if isinstance(factory_parameters, dict):
                factory_parameters["device"] = device_override
        return config, build_plan(config)

    expansion, last = _measure(
        f"{label}.load_validate_expand",
        load_and_expand,
        repeats,
        None,
        "O(number_of_grid_cells × repetitions)",
        progress,
    )
    config, unresolved_plan = last
    resolution_index = 0

    def resolve_components() -> Any:
        """Resolve into an isolated directory for an independent timing sample."""
        nonlocal resolution_index
        directory = temporary_root / label / f"resolution-{resolution_index:03d}"
        resolution_index += 1
        return resolve_plan(
            unresolved_plan,
            directory,
            config["task"].get("plan_entrypoint"),
        )

    resolution, resolved_plan = _measure(
        f"{label}.resolve_components",
        resolve_components,
        repeats,
        None,
        "O(selected spectra + annotation index + split construction)",
        progress,
    )
    task = _first_representative_task(resolved_plan)
    params = _contractive_parameters(task)
    complexity = contractive_complexity(params, _latent_dimension(task))
    factory_parameters = task.get("parameters", {}).get("factory_parameters", {})
    if not isinstance(factory_parameters, dict):
        factory_parameters = {}
    requested_device = device_override or factory_parameters.get("device")
    device = torch.device(requested_device) if requested_device else None
    test_entrypoint = config["task"].get("test_entrypoint")
    if not isinstance(test_entrypoint, str):
        raise ValueError("Benchmarking requires task.test_entrypoint.")
    test_operation = resolve_entrypoint(test_entrypoint)
    probe, _ = _measure(
        f"{label}.two_batch_train_probe",
        lambda: test_operation(deepcopy(task)),
        repeats,
        device,
        complexity,
        progress,
    )
    return [expansion, resolution, probe], {
        "tasks": len(resolved_plan.tasks),
        "params": params,
        "probe": probe,
        "input_bytes": _input_bytes(config),
        "workload": _planned_workload(task, len(resolved_plan.tasks)),
    }


def _input_bytes(config: dict[str, Any]) -> int:
    """Estimate raw MSI input bytes from the imzML and sibling ibd files."""
    factory = config.get("task", {}).get("parameters", {}).get("factory_parameters", {})
    if not isinstance(factory, dict):
        return 0
    project_path = factory.get("project_path")
    image_path = factory.get("image_path")
    if not isinstance(project_path, str) or not isinstance(image_path, str):
        return 0
    project = Path(project_path)
    image = project / image_path
    candidates = (image, image.with_suffix(".ibd"))
    return sum(path.stat().st_size for path in candidates if path.is_file())


def _planned_workload(task: dict[str, Any], task_count: int) -> dict[str, int | None]:
    """Extract train-spectrum and batch counts from a resolved campaign task."""
    resolved = task.get("parameters", {}).get("resolved", {})
    split_path = resolved.get("split_manifest") if isinstance(resolved, dict) else None
    if not isinstance(split_path, str) or not Path(split_path).is_file():
        return {"train_spectra": None, "batches_per_task": None, "campaign_batches": None}
    manifest = yaml.safe_load(Path(split_path).read_text(encoding="utf-8"))
    assignments = manifest.get("assignments", {}) if isinstance(manifest, dict) else {}
    train_spectra = len(assignments.get("train", []))
    phases = task.get("parameters", {}).get("training", {}).get("phases", [])
    batches_per_task = sum(
        int(phase.get("epochs", 1))
        * math.ceil(train_spectra / int(phase.get("batch_size", 1)))
        for phase in phases
    )
    return {
        "train_spectra": train_spectra,
        "batches_per_task": batches_per_task,
        "campaign_batches": task_count * batches_per_task,
    }


def run_campaign_benchmark(
    reference_yaml: Path | str,
    alternative_yaml: Path | str,
    output_markdown: Path | str,
    repeats: int = 3,
    projected_input_gib: float | None = None,
    projected_spectra: int | None = None,
    device: str | None = None,
) -> str:
    """Benchmark two campaigns and write an auditable Markdown comparison.

    The bounded probe runs two real training batches of the first unique grid
    cell. The measured cost is compared between YAMLs; it does not substitute
    for a complete campaign because reader I/O and validation can vary by node.

    :param reference_yaml: Baseline experiment configuration.
    :type reference_yaml: pathlib.Path | str
    :param alternative_yaml: Alternative experiment configuration.
    :type alternative_yaml: pathlib.Path | str
    :param output_markdown: Destination Markdown report.
    :type output_markdown: pathlib.Path | str
    :param repeats: Independent timings per repeatable stage, at least two.
    :type repeats: int
    :param projected_input_gib: Optional input-size target for linear I/O scaling.
    :type projected_input_gib: float | None
    :param projected_spectra: Optional train-spectrum target for linear batch scaling.
    :type projected_spectra: int | None
    :param device: Optional explicit PyTorch device, for example ``cuda``.
    :type device: str | None
    :return: Rendered Markdown report.
    :rtype: str
    :raises ValueError: If repeats or projected input size are invalid.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least two to calculate a confidence interval.")
    if projected_input_gib is not None and projected_input_gib <= 0:
        raise ValueError("projected_input_gib must be positive when provided.")
    if projected_spectra is not None and projected_spectra < 1:
        raise ValueError("projected_spectra must be a positive integer when provided.")
    if projected_spectra is not None and projected_input_gib is not None:
        raise ValueError("Specify only one of projected_spectra or projected_input_gib.")
    output = Path(output_markdown).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = BenchmarkProgress(total_measurements=6 * repeats)
    try:
        with tempfile.TemporaryDirectory(prefix="msi-campaign-benchmark-") as temporary:
            root = Path(temporary)
            reference_measurements, reference = _profile_campaign(
                "reference", Path(reference_yaml).resolve(), repeats, device, root, progress
            )
            alternative_measurements, alternative = _profile_campaign(
                "alternative", Path(alternative_yaml).resolve(), repeats, device, root, progress
            )
    finally:
        progress.close()

    reference_probe = reference["probe"]
    alternative_probe = alternative["probe"]
    ratio = alternative_probe.mean_seconds / max(reference_probe.mean_seconds, 1e-12)
    input_scale = 1.0
    if projected_input_gib is not None:
        input_scale = projected_input_gib * 1024**3 / max(reference["input_bytes"], 1)
    if projected_spectra is not None:
        train_spectra = reference["workload"]["train_spectra"]
        if train_spectra is None or train_spectra < 1:
            raise ValueError("projected_spectra requires a resolved split manifest with train spectra.")
        input_scale = projected_spectra / train_spectra
    projected_reference = reference_probe.mean_seconds * reference["tasks"] * input_scale
    projected_alternative = alternative_probe.mean_seconds * alternative["tasks"] * input_scale
    report = "\n".join(
        [
            "# Campaign numerical benchmark",
            "",
            "## Reference",
            "",
            _markdown_table(reference_measurements),
            "",
            "## Alternative",
            "",
            _markdown_table(alternative_measurements),
            "",
            "## Comparison",
            "",
            "| Metric | Reference | Alternative | Alternative / reference |",
            "| --- | ---: | ---: | ---: |",
            f"| Campaign tasks | {reference['tasks']} | {alternative['tasks']} | {alternative['tasks'] / max(reference['tasks'], 1):.3f} |",
            f"| Train spectra per task | {reference['workload']['train_spectra']} | {alternative['workload']['train_spectra']} | {float(alternative['workload']['train_spectra'] or 0) / max(float(reference['workload']['train_spectra'] or 0), 1.0):.3f} |",
            f"| Planned training batches | {reference['workload']['campaign_batches']} | {alternative['workload']['campaign_batches']} | {float(alternative['workload']['campaign_batches'] or 0) / max(float(reference['workload']['campaign_batches'] or 0), 1.0):.3f} |",
            f"| Two-batch probe mean (s) | {reference_probe.mean_seconds:.4f} | {alternative_probe.mean_seconds:.4f} | {ratio:.3f} |",
            f"| Projected probe-scaled wall time (s) | {projected_reference:.2f} | {projected_alternative:.2f} | {projected_alternative / max(projected_reference, 1e-12):.3f} |",
            "",
            "## Assumptions",
            "",
            f"- Timings use {repeats} independent observations and two-sided 95% Student-t confidence intervals.",
            "- The training probe executes two actual batches, including forward, backward, optimizer, and validation logic.",
            "- Projected wall time is probe-scaled and therefore conservative; it includes construction and validation overhead.",
            f"- Input-size scale factor: {input_scale:.4f}.",
        ]
    )
    output.write_text(report + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the campaign benchmark command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_yaml", type=Path)
    parser.add_argument("alternative_yaml", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--projected-input-gib", type=float)
    parser.add_argument("--projected-spectra", type=int)
    parser.add_argument("--device", help="Optional explicit PyTorch device, for example cuda.")
    args = parser.parse_args()
    run_campaign_benchmark(
        args.reference_yaml,
        args.alternative_yaml,
        args.output_markdown,
        repeats=args.repeats,
        projected_input_gib=args.projected_input_gib,
        projected_spectra=args.projected_spectra,
        device=args.device,
    )


if __name__ == "__main__":
    main()
