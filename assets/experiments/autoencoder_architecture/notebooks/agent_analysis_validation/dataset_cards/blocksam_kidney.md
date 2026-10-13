# BlockSAM kidney DESI-MSI

## Source

[Zenodo 18273497, version 1](https://zenodo.org/records/18273497), DOI 10.5281/zenodo.18273497. Local real-data file: `data/kidney_segmentation_reference/blocksam_kidney/raw/kidney_desi_msi.RData`, SHA-256 `2f049600e6eb41ae010bca57abed95df76fe111dc19bdddd6f704af0759628b2`. The NULL_DIFFERENCE control is not used.

## Observed source structure

`xall`: 56827 pixels × 613 filtered features. `filtered_mz`: length 613. `pixelInfoAll`: 56827 source records containing name, className, patNum, scanNum, x, y, z, tic and ctic. The source records specify Normal/RCC; sample names include normal tissue and multiple RCC subtypes. Preserve sample and patient identity separately.

Fixed first sample: NL_0705A190c, 716 pixels. It is a normal-only sample, useful for adapter checks but not for evaluating cancer/normal separation by itself. A meaningful diagnostic segmentation comparison requires a prespecified acquisition or paired set with both classes and an appropriate spatial task; do not manufacture mixed-image adjacency.

Source intensity preprocessing records include TIC normalization and peak filtering. No inverse transformation or unreported full spectrum is inferred. `rdata` returns the source pixelInfo S4 slots as attributes; harmless missing-class-constructor warnings remain visible.

## Identity and compatibility

Key: `(blocksam_kidney, pixelInfoAll.name, original_xall_row_index_zero_based)`. Do not renumber rows after sample extraction. x/y and patNum are retained. Physical coordinate units, spacing and z-section policy require validation before physical-neighbour analysis.

Representation: `source_filtered_tic_features`; source feature filtering differs from a full-spectrum pretrained CNN input. Input compatibility is unapproved. Normal/RCC labels are not detailed anatomical region masks. Patient overlap with Cardinal/Poseidon remains unverified.

Role: `external_test_only`, inherited by caches/masks; no model training or other data-dependent fit/selection.
