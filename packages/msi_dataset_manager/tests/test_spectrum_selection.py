"""Tests for explicit unannotated-spectrum selection semantics."""

from msi_dataset_manager.operations.spectrum_selection import select_merge_spectrum_ids


def test_none_keeps_all_while_zero_keeps_no_unannotated_spectra() -> None:
    """``None`` and zero must remain observably different API values."""
    arguments = {"candidate_ids": [0, 1, 2], "annotated_ids": [1]}

    assert select_merge_spectrum_ids(**arguments) == [1, 0, 2]
    assert select_merge_spectrum_ids(**arguments, unannotated_amount=0) == [1]
