# Metabolite candidate catalogs

This document describes the internal data flow from external metabolite
databases to dataset-specific candidate ions and synthetic pretraining.

## Scope

The document covers provider normalization, local source snapshots, candidate
SQLite persistence, dataset metadata filters, and the wrapper-side synthetic
peak source. Synthetic strategy parameters are documented in
[Run synthetic spectral pretraining](../../how-to/models-and-training/synthetic-pretraining.md).

## Source and catalog layers

The implementation separates complete source data from filtered dataset data:

```text
external provider export
        |
        v
assets/local/metabolite_databases/
        |
        v
CandidateProvider normalization
        |
        v
dataset metadata: polarity, m/z range, adducts
        |
        v
database_annotations/candidates.sqlite
        |
        v
CandidateCatalogPeakSource
        |
        v
synthetic sampling strategy
```

The raw source directory is shared across datasets. Each dataset receives its
own filtered SQLite catalog so that the dataset metadata and provider versions
used for training remain explicit.

## Provider normalization

`LIPIDMAPSCandidateProvider`, `HMDBCandidateProvider`, and
`ChEBICandidateProvider` implement the provider boundary in
`msi_dataset_manager.annotations.candidates.providers`.

Each provider emits `CandidateCompound` records with common fields:

- provider and source version;
- source identifier and name;
- neutral formula;
- monoisotopic mass when supplied by the source;
- SMILES, InChI, and InChIKey when supplied;
- provider chemical classes;
- source-record provenance.

`make_candidate_ions()` expands each compound into configured adducts and
computes theoretical `m/z` using the shared formula/adduct convention. Unsupported
formulas are skipped when an ion cannot be generated for the selected chemistry
implementation.

## Duplicate and distinct candidate identity

The catalog retains compound-level records and ion-level records separately.
Provider records are merged for synthetic-source purposes only when formula,
adduct, and theoretical `m/z` are identical. The merged metadata retains lists
of candidate-ion keys, compound keys, providers, and chemical classes.

Different theoretical `m/z` values remain different candidate components,
even when their source compound names are equal. This prevents distinct ion
positions from being collapsed into one synthetic label.

## Dataset filtering and persistence

`build_candidate_catalog()` first checks the dataset-local SQLite cache. Without
`refresh_cache`, a valid catalog is returned without provider access. When a
catalog is rebuilt, provider records are normalized, ions are filtered by
dataset polarity and `m/z` range, and the result is written atomically.

The catalog manifest records source names and versions, dataset identity, the
effective filters, and the source-cache directory. The raw source manifest is
stored separately in `assets/local/metabolite_databases/sources_manifest.json`.

## Synthetic candidate source

`CandidateCatalogPeakSource` in
`msi_autoencoder_wrapper.data.pretraining.sources` queries candidate ions and
maps their theoretical masses through the active binner's
`map_mass_values_to_bins()` method. It does not implement an independent binning
formula.

The source exposes:

- stable formula/adduct labels;
- one or more binned coordinates per label;
- provider, class, compound, and candidate-ion provenance;
- optional provider-class filtering.

When labelled synthetic targets are enabled, candidate labels are restricted to
the current molecule target vocabulary. Reconstruction-only candidate
generation can use the complete filtered candidate source with target labels
masked.
