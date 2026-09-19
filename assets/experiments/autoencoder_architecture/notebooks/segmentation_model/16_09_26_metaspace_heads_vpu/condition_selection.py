"""Presentation-time selection of the final three conditions."""

from __future__ import annotations

import json

import pandas as pd


BASELINE_LABEL = "balanced_bce (ClassBalancedMultiLabelBCELoss)"
BCE_LABELS = {
    "bce_per_class_positive_penalty (PositiveWeightedMultiLabelBCELoss)",
    "bce_global_positive_penalty (PositiveWeightedMultiLabelBCELoss)",
}
VPU_LABEL = "vpu_alpha_1_beta_1 (VariationalPULoss)"


def _is_contrastive(objective_json: str) -> bool:
    objective = json.loads(objective_json)
    return any(float(spec.get("weight", 0.0)) != 0.0
               for spec in objective.get("contrastive", {}).values())


def select_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Keep baseline, selected BCE variants, and both VPU alpha=1 beta=1 variants."""
    selected = inventory[
        (inventory["label"].eq(BASELINE_LABEL))
        | inventory["label"].isin(BCE_LABELS)
        | (inventory["label"].eq(VPU_LABEL))
    ].copy()
    selected["display_label"] = selected["label"]
    vpu = selected["label"].eq(VPU_LABEL)
    selected.loc[vpu, "display_label"] = selected.loc[vpu, "objective_json"].map(
        lambda value: f"{VPU_LABEL} [{'contrastive' if _is_contrastive(value) else 'no contrastive'}]"
    )
    selected["label"] = selected["display_label"]
    return selected


def select_frame(frame: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Filter cached rows and replace ambiguous labels by condition-specific labels."""
    # ``select_inventory`` is intentionally idempotent because notebooks pass the
    # already-selected inventory to every cached table loader.
    selected = inventory.copy() if "display_label" in inventory.columns else select_inventory(inventory)
    result = frame.copy()
    if "model_id" in result:
        result = result[result["model_id"].isin(selected["model_id"])]
        labels = selected.drop_duplicates("model_id").set_index("model_id")["display_label"]
        result["label"] = result["model_id"].map(labels)
    for model_column, label_column in (("left_model", "left"), ("right_model", "right")):
        if model_column in result:
            result = result[result[model_column].isin(selected["model_id"])]
            labels = selected.drop_duplicates("model_id").set_index("model_id")["display_label"]
            result[label_column] = result[model_column].map(labels)
    if {"source", "condition"}.issubset(result.columns):
        mapping = selected.drop_duplicates(["source", "condition"]).set_index(
            ["source", "condition"]
        )["display_label"]
        keys = pd.MultiIndex.from_frame(result[["source", "condition"]])
        result["label"] = mapping.reindex(keys).to_numpy()
    return result
