"""Compare numerical cost, memory, and stability of two experiment YAMLs.

Example:
    uv run --extra cu118 python \\
    assets/scripts/benchmarks/benchmark_campaigns.py \\
      reference.yaml alternative.yaml reports/campaign-benchmark.md \\
      --repeats 3 --device cuda --projected-input-gib 50

    uv run --extra cu118 python \\
        assets/scripts/benchmarks/benchmark_campaigns.py \\
        reference.yaml alternative.yaml benchmark.md \\
        --repeats 3 --device cuda --projected-spectra 500000
"""

from msi_autoencoder_wrapper.analysis.benchmarking.campaign_benchmark import main


if __name__ == "__main__":
    main()
