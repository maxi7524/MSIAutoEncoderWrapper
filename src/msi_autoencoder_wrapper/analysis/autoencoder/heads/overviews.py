"""Composed annotation-head panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .views import plot_class_maps


def plot_class_overviews(
    maps_by_class: Mapping[int, Mapping[str, Mapping[str, object]]],
    class_labels: Mapping[int, str],
    class_indices: Sequence[int],
    theme,
):
    """Return one ground-truth/probability panel per requested class."""
    return {
        int(class_index): plot_class_maps(
            maps_by_class[int(class_index)],
            class_labels.get(int(class_index), str(class_index)),
            theme,
        )
        for class_index in class_indices
    }
