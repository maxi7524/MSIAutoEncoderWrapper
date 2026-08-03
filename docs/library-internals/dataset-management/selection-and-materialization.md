# Selection and materialization

Materialization consumes a reviewed selection and transitions each dataset from
discovered metadata to validated local files and canonical annotations.

## General abstraction

### Selection handoff

The selection contains source, effective filters, reviewed records, and
annotation threshold. Materialization never repeats discovery to silently
change the accepted set.

### Dataset lifecycle

Catalog status and local path record discovery and materialization. A complete
pair is reusable; a partial pair is not a valid final state.

## Detailed implementation

### Download sources

[`materialize_selection()`](../../../src/msi_autoencoder_wrapper/dataset_management/operations/download.py)
validates the selection, optionally restricts IDs, calls the adapter, validates
the imzML pair, retrieves annotations, and replaces canonical catalog state.

### Preserve failures

Provider quota, authentication, missing files, inconsistent FDR, and incomplete
spatial annotations stop the dataset transition. Temporary downloads use partial
paths before replacement where supported.
