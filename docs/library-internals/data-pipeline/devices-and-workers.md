# Devices and workers

Storage workers, preprocessing, and model computation have separate device
responsibilities.

## General abstraction

### Device roles

Readers and DataLoader workers perform storage I/O on CPU. The main process
moves packed batches to `preprocessing_device`. Dense batches move to
`compute_device` only after preprocessing.

### Transfer policy

CPU batches may be pinned before asynchronous CUDA transfer. CUDA allocations
do not occur inside reader workers.

## Detailed implementation

### Resolve configuration

Training accepts top-level and phase-level `preprocessing_device` and
`compute_device`. Phase values win. `dataloader.num_workers` overrides any
device-derived worker default.

### Move batch records

Every batch record implements `.to()` so identifiers, spectra, axes, targets,
views, and normalization traces remain colocated. `.pin_memory()` operates on
CPU tensors before CUDA transfer.

### Reject unavailable CUDA

The trainer validates CUDA availability separately for preprocessing and
compute. A requested unavailable device raises before phase execution.
