# Extend catalog-driven precompute

Add a strategy when an existing notebook collection requires a defined sequence of reusable analysis stages.

## Implement a strategy

Place strategies below `analysis/precompute/strategies/<model_type>/`. A strategy declares its name, domain settings loader, inventory provider, and ordered `AnalysisPlugin` stages. A plugin imports its analytical calculation module and declares required and produced artifacts.

Calculations remain in their domain modules, such as `analysis/autoencoder/reconstruction` or `analysis/autoencoder/heads`. The connector strategy coordinates them; it must not duplicate their numerical code.

## Preserve contracts

Keep the following invariants:

- resolve selected models by technical identifiers, not raw labels;
- require exactly one technical record per alias and repetition;
- keep campaign coverage based on the complete source inventory;
- write control artifacts under `precompute/` and analysis tables under existing `part_*_results` directories;
- obtain shared splits and models through the resource manager;
- keep notebooks limited to CSV loading, aggregation, and rendering;
- add a focused test for ambiguous selectors, stage order, and any changed artifact contract.

Run the validation command in the [analysis how-to guide](../../how-to/analysis/running-catalog-driven-precompute.md) with `--dry-run`, then execute the affected notebooks from a clean result directory.
