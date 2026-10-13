# Poseidon ccRCC

## Source and biological reference

Source: [BOARD release gxz9kjj2r9, version 1](https://board.unimib.it/datasets/gxz9kjj2r9/1), DOI 10.17632/gxz9kjj2r9.1. The expert illustration comes from [poseidon-maldi, pinned commit 673f8bb2164933644c1693b242d2f0c22ac2820a](https://github.com/Fradenti/poseidon-maldi/tree/673f8bb2164933644c1693b242d2f0c22ac2820a). Label definitions follow [Supplement S.1](https://arxiv.org/html/2509.02482v1): cortex, capsule, tumour G2/G3, tumour G2, haemorrhage and lymphocytic infiltration.

The reference contains one human ccRCC section. Patient/specimen overlap with the other collections is unverified. Do not count 4123 pixels as independent biological replicates.

## Observed source structure

Seven RDS: coordinates 4123×2; lipids 4123×54; glycans 4123×42; peptides 4123×95; two neighbour lists of length 4123. Numeric arrays were decoded. Original coordinates are distinct, range 0–79. Coordinate row names retain older filtered indices while assay rows are renumbered 1–4123. The pinned author script `040_Extract_results_Import_data.R` explicitly pairs by position and removes assay row names. The loader records this positional contract and preserves both row-name lists; it does not perform an incorrect label-based join.

The extra repository `pixels.RDS` is coordinate-only and equals original XY plus one. `reverseX` is not a uniform `79-X`; do not use it as a verified H&E registration.

## Canonical identity and mask

Source key: `(poseidon_ccrcc, ccrcc_section_1, serialized_row_index_zero_based)`. Features retain source numeric precision, sorted by m/z with the original feature permutation recorded. Units are source grid pixels; physical spacing is unconfirmed.

`reference/manual_annotation` stores a provisional digitization of visible expert contours in **H&E image coordinates**, shape 1304×2032. H&E index is `y*2032+x`; it is not an MSI source index. The mask is disjoint; unknown includes unmarked tissue, conflicting traces and a three-pixel contour uncertainty band. All six labels have an explicit legend. MSI projection requires independently accepted registration; source raw-file names alone do not establish it.

## Input and usage restrictions

Representation: `selected_transformed_features`, not a full binned raw spectrum. Do not zero-fill unreported peaks, silently reverse transformations, or add glycan/peptide inputs to a lipid-only encoder. Encoder input compatibility and numeric MSI anatomical reference remain unavailable. The H&E mask is provisional, not accepted expert ground truth.

Role: `external_test_only`, inherited by masks/caches. Permitted: frozen inference after compatibility validation, fixed per-image clustering and reporting. Prohibited: model/pretraining/head fitting, preprocessing/vocabulary fit, threshold selection, early stopping and model selection. Repeatedly using results to choose configurations converts the dataset into validation and requires a separate final test.
