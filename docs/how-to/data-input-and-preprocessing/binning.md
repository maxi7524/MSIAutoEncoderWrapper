# Configure spectral binning

Binning converts variable mass coordinates into a fixed feature axis. Inverse
binning selects mass positions from reconstructed dense output.

## Purpose and available operations

### Forward and inverse responsibilities

`LinearBinning` sums or averages intensities in equally spaced bins. Inverse
strategies project selected points or peak regions onto a shared reconstruction
axis, which may be irregular.

### Available inverse strategies

The built-ins are `QuantileInverseBinner`, `TopPeaksInverseBinner`, and
`TopPeaksNeighbourhoodInverseBinner`.

## Detailed instructions

### Configure linear binning

```python
wrapper.context_manager.set_binner(
    "LinearBinning",
    bin_step=0.1,
    x_min=None,
    x_max=None,
    aggregation="sum",
)
```

- `bin_step` is the positive width of each m/z bin;
- `x_min` and `x_max` define the covered range;
- omitted bounds are read from the active reader;
- `active_context` is injected by the context manager.

Inspect the axis with `GetXAxis()`, `GetXMin()`, `GetXMax()`, and
`GetXAxisDepth()`.

### Configure inverse selection

```python
wrapper.context_manager.set_inverse_binner(
    "TopPeaksInverseBinner",
    max_peaks=None,
    reconstruction_mass_axis=reader.GetXAxis(),
)
```

`max_peaks=None` retains every detected local maximum. The explicitly supplied
reconstruction axis takes precedence over the shared reader axis and may be
irregular.

Other strategies expose their own registry-visible constructor parameters. Use:

```python
wrapper.context_manager.get_available_inverse_binners(
    print_return=True,
    return_value=False,
)
```

### Verify the transformation

```python
binner = wrapper.active_context.binner
dense_batch = binner(raw_batch)
inverse_batch = wrapper.active_context.inverse_binner(dense_batch)
```

Dense output length must equal `GetXAxisDepth()`. Inverse output contains finite
mass coordinates and non-negative selected intensities.
