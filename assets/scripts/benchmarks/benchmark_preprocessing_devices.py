"""Compare packed linear binning on CPU and CUDA for representative MSI sizes."""

from __future__ import annotations

import argparse
import time

import torch

from msi_autoencoder_wrapper.binners.binners_strategies.linear_binner import (
    LinearBinning,
)
from msi_autoencoder_wrapper.data import RawSpectrumBatch, TargetBatch


def build_batch(batch_size: int, points: int) -> RawSpectrumBatch:
    """Create a deterministic packed raw benchmark batch."""
    generator = torch.Generator().manual_seed(0)
    mass_values = 1000.0 + 600.0 * torch.rand(
        batch_size * points,
        generator=generator,
        dtype=torch.float64,
    )
    intensities = torch.rand(batch_size * points, generator=generator)
    lengths = torch.full((batch_size,), points, dtype=torch.long)
    return RawSpectrumBatch(
        sample_ids=torch.arange(batch_size),
        mass_values=mass_values,
        intensities=intensities,
        offsets=torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]),
        sample_indices=torch.repeat_interleave(torch.arange(batch_size), lengths),
        targets=TargetBatch.empty(),
    )


def measure(
    binner: LinearBinning,
    batch: RawSpectrumBatch,
    device: torch.device,
    repetitions: int,
) -> float:
    """Return median-style average milliseconds after two warm-up executions."""
    prepared = batch.to(device)
    for _ in range(2):
        binner.transform_batch(prepared)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repetitions):
        binner.transform_batch(prepared)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / repetitions


def main() -> None:
    """Run CPU and available CUDA preprocessing measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    batch = build_batch(args.batch_size, args.points)
    binner = LinearBinning(bin_step=0.01, x_min=1000.0, x_max=1600.0)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        elapsed = measure(binner, batch, device, args.repetitions)
        print(f"{device.type}: {elapsed:.3f} ms/batch")


if __name__ == "__main__":
    main()
