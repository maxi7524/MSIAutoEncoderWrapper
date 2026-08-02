"""Compare scalar and bulk raw-sample loading for a PixelDataset."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from msi_autoencoder_wrapper.models.datasets.strategies.pixel_dataset import PixelDataset
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader


class _Context:
    def __init__(self, reader: PyImzMLReader) -> None:
        self.reader = reader

    def get_data_reader(self, source: str) -> PyImzMLReader:
        return self.reader


def duration(function, repetitions: int) -> float:
    """Return the median wall-clock duration of a callable."""
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return statistics.median(values)


def main() -> None:
    """Benchmark dataset conversion independently from model computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    reader = PyImzMLReader(args.image)
    dataset = PixelDataset(active_context=_Context(reader), normalization="none")
    indices = list(range(min(args.batch_size, len(dataset))))
    scalar = duration(lambda: [dataset.get_raw_item(index) for index in indices], args.repetitions)
    bulk = duration(lambda: dataset.get_raw_items(indices), args.repetitions)
    speedup = scalar / bulk if bulk else float("inf")
    print(f"spectra={len(indices)} repetitions={args.repetitions}")
    print(f"scalar_median_seconds={scalar:.6f}")
    print(f"bulk_median_seconds={bulk:.6f}")
    print(f"bulk_speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
