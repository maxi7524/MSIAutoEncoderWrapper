# Configure spectral binning

Binning converts variable mass coordinates into a fixed feature axis. Inverse
binning selects mass positions from reconstructed dense output.

## Purpose and available operations

### Forward and inverse responsibilities

`LinearBinning` sums intensities into equally spaced bins. Inverse strategies do
not reconstruct lost raw peaks; they select representative bins according to a
threshold, count, peak region, or retained mass budget.

### Available inverse strategies

The built-ins are `TopPeaksInverseBinner`, `ThresholdInverseBinner`,
`PeakRegionInverseBinner`, and `CumulativeMassInverseBinner`.

## Detailed instructions

### Configure linear binning

```python
wrapper.context_manager.set_binner(
    "LinearBinning",
    bin_step=0.1,
    x_min=None,
    x_max=None,
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
    "CumulativeMassInverseBinner",
    retained_fraction=0.95,
    min_bins=1,
    max_bins=200,
    mass_strategy="intensity",
    mass_options={},
)
```

`retained_fraction` belongs to `(0, 1]`. `mass_strategy` accepts `intensity`,
`normalized_intensity`, `squared_intensity`, or a callable when constructing
the class directly. `min_bins` and `max_bins` constrain the selection budget.

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
dense = binner(xs=mz, ys=intensity)
selected_mz, selected_intensity = wrapper.active_context.inverse_binner(dense)
```

Dense output length must equal `GetXAxisDepth()`. Inverse output contains finite
mass coordinates and non-negative selected intensities.
