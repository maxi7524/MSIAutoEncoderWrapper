"""Compare scalar and bulk reader access using identical spectrum indices."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader


def measure(function, repetitions: int) -> list[float]:
    """Return wall-clock durations for repeated callable execution."""
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        durations.append(time.perf_counter() - started)
    return durations


def main() -> None:
    """Run a direct, reproducible reader throughput comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    reader = PyImzMLReader(args.image)
    indices = list(range(min(args.batch_size, reader.GetNumberOfSpectra())))
    scalar = measure(lambda: [reader.GetSpectrum(index) for index in indices], args.repetitions)
    bulk = measure(lambda: reader.GetSpectra(indices), args.repetitions)
    scalar_median = statistics.median(scalar)
    bulk_median = statistics.median(bulk)
    speedup = scalar_median / bulk_median if bulk_median else float("inf")
    print(f"spectra={len(indices)} repetitions={args.repetitions}")
    print(f"scalar_median_seconds={scalar_median:.6f}")
    print(f"bulk_median_seconds={bulk_median:.6f}")
    print(f"bulk_speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
