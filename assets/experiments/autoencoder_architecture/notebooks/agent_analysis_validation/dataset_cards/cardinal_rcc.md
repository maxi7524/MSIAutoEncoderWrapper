# Cardinal RCC

## Source

[CardinalWorkflows](https://github.com/Cardinalverse/CardinalWorkflows/tree/3dfb4b9ab7c96c3fad7ddee2122b0a842314781e), pinned commit 3dfb4b9ab7c96c3fad7ddee2122b0a842314781e. Local raw file: `data/kidney_segmentation_reference/cardinal_rcc/raw/rcc.rda`, SHA-256 `5314fe009366b3a26c26da18c09271a23f06f9bc7c56beb303a933d74648b2e1`.

## Observed source structure

The original `rcc` S4 object contains `imageData.data.iData`: int32 intensity matrix 10200 features × 16000 pixels. `featureData.data.mz` supplies the ordered mass axis. `pixelData.data` contains x, y, sample and diagnosis. Eight sample names: MH0204_33, UH0505_12, UH0710_33, UH9610_15, UH9812_03, UH9905_18, UH9911_05 and UH9912_01.

Diagnosis values are normal, cancer or missing. Missing is unknown, not normal/background. The selected first sample MH0204_33 contains 1560 pixels, 811 with diagnosis. It was selected by a fixed acquisition name, not a model score. Labels are diagnostic tissue classes, not cortex/capsule/grade masks.

Full generic R conversion fails on an unused legacy positionArray dimension object. The adapter selectively converts the required parsed public slots without modifying rdata or discarding source data. Other serialized objects (`rcc.resample`, `rcc.small`) are not silently substituted.

## Identity and compatibility

Key: `(cardinal_rcc, acquisition_name, original_rcc_column_index_zero_based)`. Preserve original X/Y; do not merge coordinate namespaces across acquisitions. Coordinates use source grid units; physical spacing is unconfirmed. Intensity representation is `source_binned_intensity`; trained-model mass range, binning, polarity and TIC contract must be explicitly checked before inference.

Role: `external_test_only`; cached arrays and masks inherit it. No fitting, training or model selection. Patient independence from other public kidney releases remains unverified.
