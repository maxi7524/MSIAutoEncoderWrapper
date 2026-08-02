"""Compare sample-wise and batched reader fetching through PyTorch DataLoader."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.data import (
    BatchPreprocessor,
    RawDatasetView,
    RawSpectrumCollator,
    RawSpectrumSample,
    SharedAxisRawBatch,
    TargetBatch,
    TargetSample,
)
from msi_autoencoder_wrapper.readers.readers_manager import ReaderManager


class _Context:
    """Minimal benchmark context required by the production preprocessor."""

    normalization = None

    def __init__(self, reader: Any, binner: Any) -> None:
        self.reader = reader
        self.binner = binner

    def get_data_reader(self, _: str) -> Any:
        """Return the benchmark image reader."""
        return self.reader


class _SampleDataset:
    """Production-compatible dataset retaining per-spectrum reader calls."""

    source = "image"
    normalization = "none"

    def __init__(self, context: _Context) -> None:
        self.active_context = context

    def __len__(self) -> int:
        return self.active_context.reader.GetNumberOfSpectra()

    def get_raw_item(self, index: int) -> RawSpectrumSample:
        axis, values = self.active_context.reader.GetSpectrum(index)
        return RawSpectrumSample(
            sample_id=index,
            mass_values=torch.as_tensor(axis, dtype=torch.float64),
            intensities=torch.as_tensor(values, dtype=torch.float32),
            targets=TargetSample.empty(),
        )

    @staticmethod
    def get_target_schemas() -> dict[str, Any]:
        return {}


class _BatchDataset(_SampleDataset):
    """Dataset exposing native reader batches through ``__getitems__``."""

    def get_raw_batch(self, indices: list[int]) -> Any:
        result = self.active_context.reader.GetSpectrumBatch(indices)
        if result.shared_mass_axis:
            return SharedAxisRawBatch(
                sample_ids=torch.from_numpy(result.sample_ids),
                mass_axis=torch.as_tensor(result.mass_values, dtype=torch.float64),
                intensities=torch.as_tensor(result.intensities, dtype=torch.float32),
                targets=TargetBatch.empty(),
            )
        samples = [
            RawSpectrumSample(
                sample_id=int(sample_id),
                mass_values=torch.tensor(axis, dtype=torch.float64),
                intensities=torch.tensor(values, dtype=torch.float32),
                targets=TargetSample.empty(),
            )
            for sample_id, axis, values in zip(
                result.sample_ids,
                result.mass_values,
                result.intensities,
            )
        ]
        return RawSpectrumCollator()(samples)


def _run_epoch(loader: DataLoader, preprocessor: BatchPreprocessor) -> float:
    """Consume one loader epoch including production binning and transfers."""
    started = time.perf_counter()
    for batch in loader:
        preprocessor(batch)
    if preprocessor.compute_device.type == "cuda":
        torch.cuda.synchronize(preprocessor.compute_device)
    return time.perf_counter() - started


def main() -> None:
    """Benchmark DataLoader throughput for reader, worker, and device choices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imzml", type=Path)
    parser.add_argument("--reader", default="M2aiaReader")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--preprocessing-device", default="cpu")
    parser.add_argument("--compute-device", default=None)
    parser.add_argument("--bin-step", type=float, default=0.01)
    args = parser.parse_args()

    ReaderManager.discover_strategies()
    BinnerManager.discover_strategies()
    compute_device = args.compute_device or args.preprocessing_device
    print("mode,workers,median_seconds,spectra_per_second")
    for mode, dataset_type in (("sample", _SampleDataset), ("batch", _BatchDataset)):
        for workers in args.workers:
            reader = ReaderManager.get_reader(args.reader, file_path=args.imzml)
            binner = BinnerManager.get_binner(
                "LinearBinning",
                bin_step=args.bin_step,
                x_min=reader.GetXMin(),
                x_max=reader.GetXMax(),
            )
            dataset = dataset_type(_Context(reader, binner))
            loader = DataLoader(
                RawDatasetView(dataset),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=workers,
                collate_fn=RawSpectrumCollator(),
                pin_memory=str(args.preprocessing_device).startswith("cuda"),
                persistent_workers=workers > 0,
                prefetch_factor=2 if workers > 0 else None,
            )
            preprocessor = BatchPreprocessor(
                dataset,
                args.preprocessing_device,
                compute_device,
            )
            timings = [_run_epoch(loader, preprocessor) for _ in range(args.epochs)]
            elapsed = statistics.median(timings)
            print(f"{mode},{workers},{elapsed:.4f},{len(dataset) / elapsed:.1f}")


if __name__ == "__main__":
    main()
