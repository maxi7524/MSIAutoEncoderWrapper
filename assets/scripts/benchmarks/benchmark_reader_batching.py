"""Compare per-spectrum and reader-native batch I/O for one imzML dataset."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from msi_autoencoder_wrapper.readers.readers_manager import ReaderManager


def _measure(operation: Any, repeats: int) -> tuple[float, Any]:
    """Return median wall time and the most recent operation result."""
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
    return statistics.median(timings), result


def main() -> None:
    """Run reader I/O comparisons and print a compact throughput table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imzml", type=Path)
    parser.add_argument(
        "--readers",
        nargs="+",
        default=["M2aiaReader", "PyImzMLReader"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ReaderManager.discover_strategies()
    rng = np.random.default_rng(args.seed)
    print("reader,batch_size,mode,median_ms,spectra_per_second,native_batch")
    for reader_name in args.readers:
        reader = ReaderManager.get_reader(reader_name, file_path=args.imzml)
        count = reader.GetNumberOfSpectra()
        for batch_size in args.batch_sizes:
            indices = rng.choice(count, size=min(batch_size, count), replace=False)
            single_time, _ = _measure(
                lambda: [reader.GetSpectrum(int(index)) for index in indices],
                args.repeats,
            )
            batch_time, _ = _measure(
                lambda: reader.GetSpectrumBatch(indices),
                args.repeats,
            )
            for mode, elapsed in (("single", single_time), ("batch", batch_time)):
                throughput = len(indices) / elapsed
                print(
                    f"{reader_name},{len(indices)},{mode},"
                    f"{elapsed * 1000.0:.3f},{throughput:.1f},"
                    f"{reader.capabilities.native_batch_read}"
                )


if __name__ == "__main__":
    main()
