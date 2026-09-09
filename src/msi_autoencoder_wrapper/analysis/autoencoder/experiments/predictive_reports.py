"""Paired comparisons and validation-only selection for predictive campaigns."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from ....utils.logger import get_custom_logger
from ..latent.predictive_geometry import representation_similarity

logger = get_custom_logger(__name__)

PAIR_KEYS = ["repetition", "initialization_seed", "training_seed", "data_contract", "backbone_contract", "training_contract"]


def experimental_units(frame: pd.DataFrame, inventory: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Average duplicate task observations within each condition and paired seed.

    :param frame: Long-form metric records with model_id and value.
    :param inventory: Audited model identities and actual derived seeds.
    :param groups: Additional measurement keys, such as split/scope/metric.
    :return: One row per experimental unit; task multiplicity and range remain visible.
    :rtype: pandas.DataFrame
    :raises ValueError: If model identities or seeds are unavailable.
    """
    columns = ["model_id", "source", "role", *PAIR_KEYS]
    observations = frame.drop(columns=[key for key in columns if key != "model_id" and key in frame])
    joined = observations.merge(inventory[columns], on="model_id", how="left", validate="many_to_one")
    if joined[PAIR_KEYS].isna().any().any():
        raise ValueError("Actual paired seeds and complete contracts are required for comparisons.")
    keys = ["source", "role", "condition", "label", *PAIR_KEYS, *groups]
    # No repeated task may masquerade as an extra seed. Duplicate spread exposes
    # nondeterministic training or differing execution environments.
    return joined.groupby(keys, dropna=False).value.agg(
        value="mean", task_count="size", duplicate_min="min", duplicate_max="max").reset_index()


def paired_comparisons(units: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all conditions only where seed, data, backbone and training match.

    :param units: Deduplicated output of :func:`experimental_units`.
    :param groups: Matching measurement keys (split, population, scope, metric).
    :return: Individual paired differences and descriptive Student-t intervals.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    differences, summaries = [], []
    conditions = units[["source", "condition", "label"]].drop_duplicates().to_dict("records")
    for left, right in combinations(conditions, 2):
        a = units[(units.source == left["source"]) & (units.condition == left["condition"])]
        b = units[(units.source == right["source"]) & (units.condition == right["condition"])]
        paired = a.merge(b, on=[*PAIR_KEYS, *groups], suffixes=("_left", "_right"), validate="one_to_one")
        identity = {"left": left["label"], "right": right["label"], "left_source": left["source"], "right_source": right["source"],
                    "left_condition": left["condition"], "right_condition": right["condition"]}
        if paired.empty:
            summaries.append({**identity, "pairs": 0, "mean_difference": np.nan,
                              "ci_low": np.nan, "ci_high": np.nan, "reason": "No identical seed/data/backbone/training contracts"})
            continue
        paired["difference"] = paired.value_left - paired.value_right
        for _, row in paired.iterrows():
            differences.append({**identity, **{key: row[key] for key in [*PAIR_KEYS, *groups]},
                                "left_value": row.value_left, "right_value": row.value_right, "value": row.difference})
        for group_key, group in paired.groupby(groups, dropna=False):
            group_key = group_key if isinstance(group_key, tuple) else (group_key,)
            values = group.difference.dropna().to_numpy()
            n = len(values)
            mean = values.mean() if n else np.nan
            half_width = t.ppf(.975, n - 1) * values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            summaries.append({**identity, **dict(zip(groups, group_key)), "pairs": n,
                              "mean_difference": mean, "median_difference": np.median(values) if n else np.nan,
                              "positive_pairs": int((values > 0).sum()), "ci_low": mean - half_width,
                              "ci_high": mean + half_width, "reason": "Exploratory unadjusted interval across seeds"})
    return pd.DataFrame(differences), pd.DataFrame(summaries)


def condition_order(inventory: pd.DataFrame) -> list[str]:
    """Return one stable left-to-right condition order for every figure and table.

    Figures are read against each other, so the horizontal position of a condition
    must not depend on which subset a particular notebook happens to plot. Baseline
    roles come first as the reference, then loss families alphabetically, then the
    weighting variants inside a family.

    :param inventory: Audited model inventory carrying role, family and label.
    :type inventory: pandas.DataFrame
    :return: Condition labels in the shared display order.
    :rtype: list[str]
    """
    ordered = inventory.drop_duplicates("label").copy()
    ordered["_baseline"] = (ordered.role != "baseline").astype(int)
    return ordered.sort_values(["_baseline", "family", "weight_mode", "label"]).label.tolist()


def condition_summary(units: pd.DataFrame, groups: list[str], *, value: str = "value") -> pd.DataFrame:
    """Summarize experimental units per condition without discarding the units.

    :param units: Deduplicated output of :func:`experimental_units`.
    :param groups: Measurement keys retained alongside the condition identity.
    :param value: Numeric column summarized.
    :return: Mean, standard deviation, extremes and seed count per condition.
    :rtype: pandas.DataFrame
    """
    keys = ["source", "role", "condition", "label", *groups]
    return units.groupby(keys, dropna=False)[value].agg(
        mean="mean", std="std", minimum="min", maximum="max", seeds="count").reset_index()


def resolve_shortlist(settings: dict, order: list[str], *, decision: pd.DataFrame | None = None,
                      size: int = 4) -> list[str]:
    """Resolve which conditions the detailed per-class analyses compare.

    Figures that overlay every condition become unreadable long before nine of them,
    and the detailed stage exists to characterize a decision that the selection stage
    has already made. Three sources are consulted in order: the explicit list in the
    analysis settings, then the validation ranking of a supplied decision table, then
    the shared campaign order. The explicit list wins because the shortlist is a
    scientific choice, and it is presentational, so editing it never invalidates the
    stored inference.

    :param settings: Resolved analysis settings; ``shortlist`` may name conditions.
    :type settings: dict
    :param order: All audited condition labels, normally :func:`condition_order`.
    :type order: list[str]
    :param decision: Optional output of :func:`decision_table` supplying the fallback
        validation ranking.
    :type decision: pandas.DataFrame | None
    :param size: Number of conditions kept by either fallback.
    :type size: int
    :return: Condition labels, all of which exist in the campaign.
    :rtype: list[str]
    :raises ValueError: If a configured label matches no audited condition.
    """
    configured = list(settings.get("shortlist") or [])
    if configured:
        missing = [label for label in configured if label not in set(order)]
        if missing:
            raise ValueError(f"Configured shortlist labels are not present in the campaign: {missing}")
        return configured
    if decision is not None and decision.eligible.any():
        ranked = decision[decision.eligible].sort_values("validation_rank").label.tolist()
        logger.info("No configured shortlist; using the %s best validation-ranked conditions.", size)
        return ranked[:size]
    logger.warning("No configured shortlist and no decision table; using the first %s campaign conditions.", size)
    return list(order)[:size]


def baseline_contrasts(contrasts: pd.DataFrame, inventory: pd.DataFrame, *,
                       reference_role: str = "baseline") -> pd.DataFrame:
    """Keep only comparisons against the reference condition and orient every sign.

    :func:`paired_comparisons` enumerates unordered condition pairs, so half of the
    comparisons against a baseline come out with the baseline on the left and half
    with it on the right. Reading such a table requires checking the sign convention
    of every row separately, which is exactly the kind of silent error a figure
    hides. This restricts the table to baseline comparisons and rewrites each row so
    the value is always *candidate minus reference*.

    :param contrasts: Paired comparison summaries.
    :type contrasts: pandas.DataFrame
    :param inventory: Audited inventory carrying the ``role`` of each source.
    :type inventory: pandas.DataFrame
    :param reference_role: Inventory role treated as the reference.
    :type reference_role: str
    :return: One row per candidate condition and measurement, with ``label`` naming
        the candidate, ``reference`` the baseline, and the difference and interval
        bounds oriented as candidate minus reference.
    :rtype: pandas.DataFrame
    """
    if contrasts.empty:
        return contrasts.copy()
    references = set(inventory.loc[inventory.role == reference_role, "source"])
    rows = []
    for row in contrasts.to_dict("records"):
        left_is_reference = row["left_source"] in references
        right_is_reference = row["right_source"] in references
        if left_is_reference == right_is_reference:
            continue
        ## Flip the sign, the interval bounds and the pair counts together
        flip = left_is_reference
        record = dict(row)
        record["label"] = row["right"] if flip else row["left"]
        record["reference"] = row["left"] if flip else row["right"]
        if flip:
            record["mean_difference"] = -row["mean_difference"]
            record["median_difference"] = -row.get("median_difference", np.nan)
            record["ci_low"], record["ci_high"] = -row["ci_high"], -row["ci_low"]
            record["positive_pairs"] = row["pairs"] - row.get("positive_pairs", 0)
        rows.append(record)
    result = pd.DataFrame(rows)
    logger.info("Oriented %s baseline contrasts as candidate minus reference.", len(result))
    return result


def masking_contrasts(contrasts: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Extract the matched weighted-BCE versus PN-BCE comparisons.

    :param contrasts: All paired comparison summaries.
    :param inventory: Audited family and weighting metadata.
    :return: Subset with explicit interpretation and source conditions.
    :rtype: pandas.DataFrame
    """
    if contrasts.empty:
        return contrasts.copy()
    identity = inventory.drop_duplicates(["source", "condition"]).set_index(["source", "condition"])
    selected = []
    for row in contrasts.to_dict("records"):
        left = identity.loc[(row["left_source"], row["left_condition"])]
        right = identity.loc[(row["right_source"], row["right_condition"])]
        families = {left.family, right.family}
        if (families == {"PositiveWeightedMultiLabelBCELoss", "SignalMaskedBCELoss"}
                and left.weight_mode == right.weight_mode and left.head_weight == right.head_weight):
            selected.append({**row, "weight_mode": left.weight_mode,
                             "interpretation": "U masking plus changed P/N-derived weights; sign follows left minus right"})
    return pd.DataFrame(selected)


def decision_table(prediction: pd.DataFrame, reconstruction: pd.DataFrame, inventory: pd.DataFrame, *, expected_seeds: int = 5) -> pd.DataFrame:
    """Rank conditions using validation AP and a validation reconstruction Pareto set.

    :param prediction: Aggregate ranking metrics from all splits.
    :param reconstruction: Aggregate decoder metrics from all splits.
    :param inventory: Model inventory; retains missing runs for coverage checks.
    :param expected_seeds: Required independent seed count for shortlist eligibility.
    :return: Validation-only ranking; test columns are confirmation, never selectors.
    :rtype: pandas.DataFrame
    """
    pred = prediction[(prediction.scope == "train_supported") & (prediction.metric == "average_precision")]
    units = experimental_units(pred, inventory, ["split", "population"])
    summary = units.groupby(["source", "condition", "label", "split", "population"]).value.agg(
        mean="mean", std="std", seeds="count").reset_index()
    summary["measure"] = summary["split"] + "_" + summary["population"]
    keys = ["source", "condition", "label"]
    table = summary.pivot(index=keys, columns="measure", values="mean")
    counts = summary.pivot(index=keys, columns="measure", values="seeds")
    std = summary.pivot(index=keys, columns="measure", values="std")
    table = table.join(counts.add_suffix("_seeds")).join(std.add_suffix("_std")).reset_index()
    rec = reconstruction[(reconstruction.metric == "masserstein") & (reconstruction.statistic == "mean") & (reconstruction.split == "validation")]
    rec_units = experimental_units(rec, inventory, ["split"])
    rec_summary = rec_units.groupby(keys).value.mean().rename("validation_masserstein").reset_index()
    table = table.merge(rec_summary, on=keys, how="left", validate="one_to_one")
    required = ["validation_annotation_retrieval", "validation_operational_pn", "validation_masserstein"]
    for name in required:
        if name not in table:
            table[name] = np.nan
    count_column = "validation_annotation_retrieval_seeds"
    if count_column not in table:
        table[count_column] = 0
    table["complete_seeds"] = table[count_column] == expected_seeds
    # Comparisons with different contracts are separate experiments, not a common
    # Pareto front. Block a misleading mixed-cohort decision table.
    active = inventory[inventory.model_id.isin(prediction.model_id)]
    if active[["data_contract", "backbone_contract", "training_contract"]].drop_duplicates().shape[0] != 1:
        table["comparable_contracts"] = False
    else:
        table["comparable_contracts"] = True
    ## Pooled counterpart of the macro selection quantity, for the same conditions
    micro = prediction[(prediction.scope == "train_supported") & (prediction.metric == "micro_average_precision")]
    if not micro.empty:
        micro_units = experimental_units(micro, inventory, ["split", "population"])
        micro_summary = micro_units.groupby([*keys, "split", "population"]).value.mean().reset_index()
        micro_summary["measure"] = "micro_" + micro_summary["split"] + "_" + micro_summary["population"]
        table = table.merge(micro_summary.pivot(index=keys, columns="measure", values="value").reset_index(),
                            on=keys, how="left", validate="one_to_one")
    table["eligible"] = table.complete_seeds & table.comparable_contracts & table[required].notna().all(axis=1)
    table["validation_rank"] = table.validation_annotation_retrieval.where(table.eligible).rank(ascending=False, method="min")
    table["validation_pareto"] = False
    candidates = table[table.eligible]
    for index, row in candidates.iterrows():
        dominated = ((candidates.validation_annotation_retrieval >= row.validation_annotation_retrieval)
                     & (candidates.validation_masserstein <= row.validation_masserstein)
                     & ((candidates.validation_annotation_retrieval > row.validation_annotation_retrieval)
                        | (candidates.validation_masserstein < row.validation_masserstein))).any()
        table.loc[index, "validation_pareto"] = not dominated
    table["next_step"] = np.where(table.eligible & table.validation_pareto,
                                   "Replicate on independent acquisition; test head x contractive interaction",
                                   np.where(table.eligible, "Inspect paired effects and reconstruction tails", "Resolve incomplete seeds or incompatible contracts"))
    return table.sort_values(["validation_rank", "source", "condition"], na_position="last")


def geometry_similarity(settings: dict, inventory: pd.DataFrame) -> pd.DataFrame:
    """Measure CKA across heads and seeds using the identical cached pixel rows.

    :param settings: Resolved settings with completed shared inference.
    :param inventory: Ready model inventory.
    :return: Pairwise CKA with same-condition and same-seed flags.
    :rtype: pandas.DataFrame
    :raises ValueError: If sampled row order differs.
    """
    manifest = json.loads((Path(settings["cache_directory"]) / "latest.json").read_text())
    index = inventory.set_index("model_id")
    rows = []
    for left, right in combinations(manifest["models"], 2):
        a, b = index.loc[left["model_id"]], index.loc[right["model_id"]]
        if a.data_contract != b.data_contract:
            continue
        for split in ("validation", "test"):
            with np.load(Path(left["directory"]) / f"{split}_latent.npz") as first, np.load(Path(right["directory"]) / f"{split}_latent.npz") as second:
                if not np.array_equal(first["rows"], second["rows"]):
                    raise ValueError("Representation comparison requires identical sample row order.")
                for space in ("z", "u"):
                    rows.append({"left_model": left["model_id"], "right_model": right["model_id"],
                                 "left": a.label, "right": b.label, "split": split, "space": space,
                                 "same_condition": a.condition == b.condition,
                                 "same_seed": all(a[key] == b[key] for key in PAIR_KEYS[:3]),
                                 "metric": "linear_cka", "value": representation_similarity(first[space], second[space])})
    return pd.DataFrame(rows)


def save_tables(directory: Path | str, **tables: pd.DataFrame) -> None:
    """Persist canonical numerical report tables as CSV.

    :param directory: Notebook-specific results directory.
    :param tables: Named DataFrames; names become CSV stems.
    :return: None.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(path / f"{name}.csv", index=False)
    logger.info("Saved %s analytical tables in %s.", len(tables), path)
