[](https://github.com/DKFZ-ABI/automsi/tree/main)


uv run python assets/scripts/benchmarks/benchmark_reader_batching.py data/tutorial_workspace/datasets/example_1/example_1.imzML
uv run python assets/scripts/benchmarks/benchmark_dataloader_batching.py data/tutorial_workspace/imgs/example.imzML --workers 0 1 2 


uv run python assets/scripts/benchmarks/benchmark_reader_batching.py data/tutorial_workspace/datasets/kidney-pilot-v2/kidney-pilot-v2.imzML
uv run python assets/scripts/benchmarks/benchmark_dataloader_batching.py workspace/datasets/kidney-pilot-v2/kidney-pilot-v2.imzML --workers 0 1 2 