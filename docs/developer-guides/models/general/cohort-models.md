# Support cohort model contexts

Cohort support must consume immutable member contexts and preserve image/local
spectrum identity.

## Cohort scope and decisions

### Dataset strategy

Reuse `CohortDataset` composition when the family consumes the same per-member
sample contract. Add a family dataset only when sampling or cross-member input
semantics differ.

### Model policy

Decide whether cohort execution uses one common artifact, one per member, or a
new family-specific policy. Do not overload autoencoder reference fields for a
different model family.

## Implementation instructions

### Retain member identity

Return sample identities containing `image_key` and local `spectrum_id`. Grouped
splits and spatial outputs depend on this mapping.

### Validate member compatibility

Check feature spaces, target schemas, required readers, and model references for
all members before execution. Report all missing members where possible.

### Persist the policy

Extend cohort schema with a versioned family-specific section and reconstruct
references without loading every artifact during configuration parsing.
