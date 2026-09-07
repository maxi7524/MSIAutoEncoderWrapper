# Campaign numerical benchmark

## Reference

| Step ID | Samples | Mean time (s) | 95% CI (s) | P95 time (s) | RSS allocation | CUDA peak allocation | Complexity |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| reference.load_validate_expand | 3 | 0.0266 | [0.0191, 0.0342] | 0.0301 | 556.00 KiB | 0.00 B | O(number_of_grid_cells × repetitions) |
| reference.resolve_components | 3 | 44.9066 | [-6.3575, 96.1708] | 68.7336 | 462.98 MiB | 0.00 B | O(selected spectra + annotation index + split construction) |
| reference.two_batch_train_probe | 3 | 41.6530 | [35.1411, 48.1648] | 44.6645 | 343.52 MiB | 136.92 MiB | O(5·C_VJP(B,M,D)) via 5 reverse-mode VJP passes |

## Alternative

| Step ID | Samples | Mean time (s) | 95% CI (s) | P95 time (s) | RSS allocation | CUDA peak allocation | Complexity |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| alternative.load_validate_expand | 3 | 0.0306 | [0.0295, 0.0318] | 0.0309 | 0.00 B | 0.00 B | O(number_of_grid_cells × repetitions) |
| alternative.resolve_components | 3 | 32.6301 | [31.4006, 33.8595] | 33.1606 | 73.23 MiB | 0.00 B | O(selected spectra + annotation index + split construction) |
| alternative.two_batch_train_probe | 3 | 47.8165 | [28.8812, 66.7518] | 56.5976 | 246.59 MiB | 154.93 MiB | O(3·C_VJP(B,M,D) + 4·C_JVP(B,M,D)) |

## Comparison

| Metric | Reference | Alternative | Alternative / reference |
| --- | ---: | ---: | ---: |
| Campaign tasks | 60 | 75 | 1.250 |
| Train spectra per task | 33650 | 33650 | 1.000 |
| Planned training batches | 473400 | 591750 | 1.250 |
| Two-batch probe mean (s) | 41.6530 | 47.8165 | 1.148 |
| Projected probe-scaled wall time (s) | 37134.90 | 53287.34 | 1.435 |

## Assumptions

- Timings use 3 independent observations and two-sided 95% Student-t confidence intervals.
- The training probe executes two actual batches, including forward, backward, optimizer, and validation logic.
- Projected wall time is probe-scaled and therefore conservative; it includes construction and validation overhead.
- Input-size scale factor: 14.8588.
