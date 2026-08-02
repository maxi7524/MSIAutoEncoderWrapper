# Cohort execution

Cohort execution snapshots several configured local contexts into one immutable
multi-image value without merging their runtime ledgers.

## General abstraction

### Member snapshot

`CohortMember` retains reader, binner, annotation reader, optional latent
reader, portable context configuration, and optional model reference. Local
component identity remains tied to the member image.

### Model reference

`ModelReference` identifies a workspace `image/model` artifact or external path
and may retain a fingerprint. It delays model loading until execution needs the
member autoencoder.

## Detailed implementation

### Construct immutable values

[`CohortContext`](../../../src/msi_autoencoder_wrapper/core/mixins/cohort/context.py)
is a frozen dataclass. Membership changes call `with_members()` and replace the
manager's stored value. Duplicate image keys are rejected.

### Apply model policy

Policy `common` stores one reference on the cohort. Policy `per_member` stores a
reference on every member and validates completeness. Artifact fingerprinting
detects a referenced model folder changed after the cohort snapshot.

### Build cohort datasets

[`CohortDataset`](../../../src/msi_autoencoder_wrapper/models/datasets/strategies/cohort_dataset.py)
wraps each member in a minimal local-context adapter and constructs a
`PixelDataset`. Cumulative offsets map global dataset indices to member and
local spectrum indices. Returned identities preserve both values.

### Activate execution scope

Activating a cohort sets workspace execution scope to `cohort` but retains the
local active image. Deactivation returns routing to local scope. This avoids
destroying local state when cohort work is temporary.
