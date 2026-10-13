"""Distribution of chemical classes along the m/z axis and bin-level class ambiguity.

The question answered here is whether the class of a signal is determined by its
position on the binned m/z axis. If every bin that can contain a member of class
:math:`c` contains members of :math:`c` only, absence of signal in those bins is
evidence of absence of :math:`c` and class labels admit a positive/negative (PN)
formulation. If bins are shared between classes, a signal in the bin does not
identify the class and only annotated positives are reliable, i.e. a
positive/unlabelled (PU) formulation.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from msi_dataset_manager.annotations.candidates.model import calculate_theoretical_mz
from msi_dataset_manager.annotations.chemistry import ChemicalTaxonomy

from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def candidate_universe(
    taxonomy: ChemicalTaxonomy,
    adducts: Sequence[str],
    mz_range: tuple[float, float],
) -> pd.DataFrame:
    """Return theoretical ions of every compound of a taxonomy inside an m/z range.

    :param taxonomy: Taxonomy providing compounds with neutral formulae.
    :type taxonomy: ChemicalTaxonomy
    :param adducts: METASPACE adducts observed in the collection.
    :type adducts: collections.abc.Sequence[str]
    :param mz_range: Inclusive ``(low, high)`` m/z axis.
    :type mz_range: tuple[float, float]
    :return: Table ``compound_id``, ``formula``, ``adduct``, ``mz``.
    :rtype: pandas.DataFrame
    """
    masses: dict[tuple[str, str], float] = {}
    for formula in taxonomy.compounds.formula.unique():
        for adduct in adducts:
            try:
                masses[(formula, adduct)] = calculate_theoretical_mz(formula, adduct)[0]
            except ValueError:
                continue
    rows = [(compound, formula, adduct, masses[(formula, adduct)])
            for compound, formula in zip(taxonomy.compounds.compound_id, taxonomy.compounds.formula)
            for adduct in adducts if (formula, adduct) in masses]
    frame = pd.DataFrame(rows, columns=["compound_id", "formula", "adduct", "mz"])
    return frame[frame.mz.between(*mz_range)].reset_index(drop=True)


def compound_class_depths(taxonomy: ChemicalTaxonomy, compound_ids: Iterable[str]) -> pd.DataFrame:
    """Return every class of the given compounds together with its hierarchy depth.

    A compound with several classes at one depth (DAG membership) contributes one
    row per class; a compound without a class at a depth contributes none there.

    :param taxonomy: Taxonomy.
    :type taxonomy: ChemicalTaxonomy
    :param compound_ids: Compounds of interest.
    :type compound_ids: collections.abc.Iterable[str]
    :return: Table ``compound_id``, ``class_id``, ``depth``.
    :rtype: pandas.DataFrame
    """
    rows = [(compound, class_id, taxonomy.class_depth.get(class_id, -1))
            for compound in pd.unique(pd.Series(list(compound_ids)))
            for class_id in taxonomy.compound_closure(compound)]
    return pd.DataFrame(rows, columns=["compound_id", "class_id", "depth"])


def bin_class_counts(mz: np.ndarray, classes: np.ndarray, bin_width: float, mz_low: float) -> pd.DataFrame:
    """Count class members per m/z bin.

    :param mz: Ion m/z values ``(n,)``.
    :type mz: numpy.ndarray
    :param classes: Class identifier of each value ``(n,)``.
    :type classes: numpy.ndarray
    :param bin_width: Bin width in Da; bins start at ``mz_low``.
    :type bin_width: float
    :param mz_low: Left edge of the first bin.
    :type mz_low: float
    :return: Table ``bin``, ``class_id``, ``count``.
    :rtype: pandas.DataFrame
    """
    bins = np.floor((np.asarray(mz) - mz_low) / bin_width).astype(np.int64)  # (n,)
    frame = pd.DataFrame({"bin": bins, "class_id": classes})
    return frame.groupby(["bin", "class_id"]).size().rename("count").reset_index()


def ambiguity_metrics(counts: pd.DataFrame, purity_threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    """Summarize how well bins identify classes.

    With :math:`n_{bc}` members of class :math:`c` in bin :math:`b`,
    :math:`n_b = \\sum_c n_{bc}` and :math:`N = \\sum_b n_b`:

    * Bayes accuracy :math:`\\sum_b \\max_c n_{bc} / N` — accuracy of the best
      predictor of the class from the bin alone;
    * exclusive member fraction of class :math:`c` — share of its members in bins
      where :math:`n_{bc} = n_b`;
    * pure member fraction — share in bins where :math:`n_{bc}/n_b \\ge` threshold;
    * unambiguous bin fraction — occupied bins containing one class.

    :param counts: Output of :func:`bin_class_counts`.
    :type counts: pandas.DataFrame
    :param purity_threshold: Minimum within-bin class share counted as pure.
    :type purity_threshold: float
    :return: Global metrics and per-class metrics.
    :rtype: tuple[dict[str, float], pandas.DataFrame]
    """
    if counts.empty:
        return {"members": 0}, pd.DataFrame(columns=["class_id", "members", "exclusive_fraction", "pure_fraction",
                                                     "classes_sharing_bins"])
    totals = counts.groupby("bin")["count"].transform("sum")
    share = counts["count"] / totals
    classes_per_bin = counts.groupby("bin").class_id.transform("nunique")
    frame = counts.assign(share=share, exclusive=classes_per_bin == 1, pure=share >= purity_threshold)
    best = counts.groupby("bin")["count"].max().sum()
    total = counts["count"].sum()
    occupied = counts.bin.nunique()
    summary = {
        "members": int(total),
        "classes": int(counts.class_id.nunique()),
        "occupied_bins": int(occupied),
        "bayes_accuracy": float(best / total),
        "unambiguous_bin_fraction": float((counts.groupby("bin").class_id.nunique() == 1).sum() / occupied),
        "classes_per_bin_mean": float(counts.groupby("bin").class_id.nunique().mean()),
        "exclusive_member_fraction": float(frame.loc[frame.exclusive, "count"].sum() / total),
        "pure_member_fraction": float(frame.loc[frame.pure, "count"].sum() / total),
    }
    sharing = counts.merge(counts[["bin", "class_id"]].rename(columns={"class_id": "other"}), on="bin")
    sharing = sharing[sharing.class_id != sharing.other].groupby("class_id").other.nunique()
    per_class = frame.groupby("class_id").apply(
        lambda group: pd.Series({"members": int(group["count"].sum()),
                                 "exclusive_fraction": float(group.loc[group.exclusive, "count"].sum() / group["count"].sum()),
                                 "pure_fraction": float(group.loc[group.pure, "count"].sum() / group["count"].sum())}),
        include_groups=False).reset_index()
    per_class["classes_sharing_bins"] = per_class.class_id.map(sharing).fillna(0).astype(int)
    return summary, per_class


def observed_bin_exclusivity(
    observed: pd.DataFrame,
    universe_counts: pd.DataFrame,
    bin_width: float,
    mz_low: float,
) -> pd.DataFrame:
    """Check whether bins of annotated ions are owned by their class in the universe.

    :param observed: Annotated ions with ``ion_id``, ``mz`` and their unique certain
        ``class_id`` at the depth.
    :type observed: pandas.DataFrame
    :param universe_counts: :func:`bin_class_counts` of the candidate universe at the
        same depth and bin width.
    :type universe_counts: pandas.DataFrame
    :param bin_width: Bin width in Da.
    :type bin_width: float
    :param mz_low: Left edge of the first bin.
    :type mz_low: float
    :return: ``observed`` with ``bin``, ``universe_classes_in_bin`` and
        ``own_class_share`` (share of universe members of the bin in the ion's class).
    :rtype: pandas.DataFrame
    """
    frame = observed.assign(bin=np.floor((observed.mz.to_numpy() - mz_low) / bin_width).astype(np.int64))
    per_bin = universe_counts.groupby("bin").agg(universe_classes_in_bin=("class_id", "nunique"), bin_total=("count", "sum"))
    own = universe_counts.rename(columns={"count": "own_count"})
    frame = frame.merge(per_bin, left_on="bin", right_index=True, how="left").merge(own, on=["bin", "class_id"], how="left")
    frame["own_class_share"] = frame.own_count.fillna(0) / frame.bin_total
    return frame.drop(columns=["own_count", "bin_total"])


def class_mz_profile(
    taxonomies: Mapping[str, ChemicalTaxonomy],
    evidence: pd.DataFrame,
    ions: pd.DataFrame,
    *,
    adducts: Sequence[str],
    mz_range: tuple[float, float],
    depths: Sequence[int],
    bin_widths: Sequence[float],
    purity_threshold: float,
    min_class_members: int,
    resolution: str = "formula",
) -> dict[str, pd.DataFrame]:
    """Compute class m/z distributions and bin ambiguity for all taxonomies.

    :param taxonomies: Loaded taxonomies.
    :type taxonomies: collections.abc.Mapping[str, ChemicalTaxonomy]
    :param evidence: Collection class evidence.
    :type evidence: pandas.DataFrame
    :param ions: Collection ion table with ``ion_id`` and ``mz``.
    :type ions: pandas.DataFrame
    :param adducts: Adducts of the collection.
    :type adducts: collections.abc.Sequence[str]
    :param mz_range: Model m/z axis.
    :type mz_range: tuple[float, float]
    :param depths: Hierarchy depths analysed.
    :type depths: collections.abc.Sequence[int]
    :param bin_widths: Bin widths in Da.
    :type bin_widths: collections.abc.Sequence[float]
    :param purity_threshold: Within-bin share counted as pure.
    :type purity_threshold: float
    :param min_class_members: Universe members needed for per-class distribution rows.
    :type min_class_members: int
    :param resolution: Candidate resolution of the observed class labels.
    :type resolution: str
    :return: Tables ``universe_summary``, ``universe_classes``, ``observed_ions``,
        ``class_mz_quantiles`` and ``mz_histogram``.
    :rtype: dict[str, pandas.DataFrame]
    """
    summaries, class_rows, observed_rows, quantiles, histograms = [], [], [], [], []
    certain = evidence[(evidence.resolution == resolution) & evidence.certain.astype(bool)]
    ion_mz = ions.set_index("ion_id").mz
    for name, taxonomy in taxonomies.items():
        universe = candidate_universe(taxonomy, adducts, mz_range)
        names = taxonomy.classes.set_index("class_id").name
        closure = compound_class_depths(taxonomy, universe.compound_id)
        for depth in depths:
            ## Universe members of classes at this depth
            members = universe.merge(closure[closure.depth == depth].drop(columns="depth"), on="compound_id")
            if members.empty:
                continue
            ## Observed ions with a unique certain class at this depth
            level = certain[(certain.taxonomy == name) & (certain.class_id.map(taxonomy.class_depth) == depth)]
            level = level[level.groupby("ion_id").class_id.transform("size") == 1]
            observed = pd.DataFrame({"ion_id": level.ion_id.to_numpy(), "class_id": level.class_id.to_numpy(),
                                     "mz": level.ion_id.map(ion_mz).to_numpy()})
            observed = observed[observed.mz.between(*mz_range)]
            ## Class m/z distribution (bin independent)
            sizes = members.groupby("class_id").size()
            for class_id, group in members[members.class_id.isin(sizes[sizes >= min_class_members].index)].groupby("class_id"):
                values = group.mz.to_numpy()
                quantiles.append({"taxonomy": name, "depth": depth, "class_id": class_id, "name": names.get(class_id, class_id),
                                  "members": len(values), "observed_ions": int((observed.class_id == class_id).sum()),
                                  **{f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}})
            edges = np.arange(mz_range[0], mz_range[1] + 5.0, 5.0)
            for class_id, group in members.groupby("class_id"):
                counts, _ = np.histogram(group.mz, bins=edges)
                histograms.extend({"taxonomy": name, "depth": depth, "class_id": class_id, "population": "universe",
                                   "mz_left": float(edges[i]), "count": int(c)} for i, c in enumerate(counts) if c)
            for class_id, group in observed.groupby("class_id"):
                counts, _ = np.histogram(group.mz, bins=edges)
                histograms.extend({"taxonomy": name, "depth": depth, "class_id": class_id, "population": "observed",
                                   "mz_left": float(edges[i]), "count": int(c)} for i, c in enumerate(counts) if c)
            ## Bin-level ambiguity per width
            for width in bin_widths:
                counts = bin_class_counts(members.mz.to_numpy(), members.class_id.to_numpy(), float(width), mz_range[0])
                summary, per_class = ambiguity_metrics(counts, purity_threshold)
                observed_summary, _ = ambiguity_metrics(
                    bin_class_counts(observed.mz.to_numpy(), observed.class_id.to_numpy(), float(width), mz_range[0]),
                    purity_threshold)
                summaries.append({"taxonomy": name, "depth": depth, "bin_width": float(width), "population": "universe", **summary})
                summaries.append({"taxonomy": name, "depth": depth, "bin_width": float(width), "population": "observed", **observed_summary})
                class_rows.append(per_class.assign(taxonomy=name, depth=depth, bin_width=float(width),
                                                   name=per_class.class_id.map(names)))
                if not observed.empty:
                    observed_rows.append(observed_bin_exclusivity(observed, counts, float(width), mz_range[0])
                                         .assign(taxonomy=name, depth=depth, bin_width=float(width)))
            logger.info("Class m/z profile %s depth %s: %s universe members, %s observed ions.",
                        name, depth, len(members), len(observed))
    return {
        "universe_summary": pd.DataFrame(summaries),
        "universe_classes": pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame(),
        "observed_ions": pd.concat(observed_rows, ignore_index=True) if observed_rows else pd.DataFrame(),
        "class_mz_quantiles": pd.DataFrame(quantiles),
        "mz_histogram": pd.DataFrame(histograms),
    }
