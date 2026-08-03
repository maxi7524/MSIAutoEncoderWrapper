# Configure MSI readers

Readers expose spectra, coordinates, mass ranges, metadata, and optional native
batch reads for one image.

## Purpose and available operations

### Reader selection

`PyImzMLReader` reads imzML/ibd pairs through pyimzML. The m2aia strategies use
m2aia when its native image and batch behavior is required. All readers satisfy
`MSIBaseReader`, including the intentionally capitalized m2aia-compatible API.

### Shared reader contract

The contract includes `GetSpectrum()`, `GetSpectrumBatch()`,
`GetSpectrumPosition()`, `GetNumberOfSpectra()`, mass-axis bounds, and metadata.

## Detailed instructions

### List and select readers

```python
available = wrapper.context_manager.get_available_readers(
    print_return=False,
    return_value=True,
)
reader = wrapper.context_manager.set_reader(
    "PyImzMLReader",
    "data/tutorial_workspace/datasets/example_1/example_1.imzML",
    auto_load_annotations=True,
)
```

`set_reader()` accepts a registry key, reader class, or initialized reader.
Constructor keyword arguments follow the selected strategy. Automatic
annotation loading is documented in [annotations](annotations.md).

### Read spectra and spatial information

```python
count = reader.GetNumberOfSpectra()
mz, intensity = reader.GetSpectrum(0)
x, y, z = reader.GetSpectrumPosition(0)
metadata = reader.GetMetaData()
batch = reader.GetSpectrumBatch([0, 1, 2])
```

Spectrum IDs are zero-based. Coordinates follow the reader's physical source;
the wrapper's coordinate-order setting controls how spatial slicing interprets
axes.

### Validate reader input

The imzML file and sibling ibd file must exist. Spectrum mass and intensity
arrays must have matching lengths and finite values before preprocessing.
Reader-specific dependency failures are not converted into empty spectra.
