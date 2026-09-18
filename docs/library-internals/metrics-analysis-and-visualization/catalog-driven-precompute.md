# Catalog-driven precompute

The common precompute runner separates model selection, staged computation, persisted artifacts, and notebook presentation.

## Execution flow

`analysis.precompute` loads domain settings and inventory through a registered strategy. The model catalog resolves aliases against technical identifiers. The runner writes catalog artifacts, validates stage dependencies, then executes stages in declared order with one shared resource manager.

For `autoencoder.heads_general_analysis`, the order is campaign training dynamics, shared inference, local reconstruction, global reconstruction, and latent geometry. The contractive strategy adapts its existing named routines in its declared order.

## Resource and artifact contracts

The resource manager shares decoded partitions and retains inactive checkpoints on CPU while one model is active on the selected device. Stages exchange only declared artifacts. Result tables remain in the existing `part_*_results` directories; the `precompute` directory contains only strategy control artifacts.

The model catalog identity is `(source, grid_id, repetition)`. Raw labels are not an identity because multiple campaigns may reuse them. Campaign sources are inventory providers; they do not by themselves decide which models an analysis may use.

## Presentation contract

The catalog persists display labels and visual encodings in `visualization_contract.csv`. Notebook helpers filter result tables by `model_id` and apply the persisted labels. This prevents notebooks from recreating campaign-specific condition logic.

For modification requirements, see [analysis development](../../developer-guides/analysis/catalog-driven-precompute.md).
