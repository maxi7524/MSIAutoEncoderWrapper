# Handoff: synthetic pretraining analysis (parts 0_05–6)

## Campaign state

Campaign `metaspace-pretrain-repaired-20260923-01` completed with 310/310 models and no failed tasks. The analysis covers two spectral axes, real-only baselines, and synthetic-pretraining variants evaluated at the pretrained, frozen-head and unfrozen-head stages.

The 53 notebooks for parts 0_05–6 have executed outputs. Their remarks and conclusions were checked against the result tables. Parts 0_03 and 0_04 report on the baseline models while retaining all 310 models in their result tables.

## Re-running the precompute

The shared cache stores the campaign plan, pixel populations, synthetic reference artifacts and model inference results. Cache contracts validate existing artifacts before reuse. Analyses whose input contracts have not changed are skipped.

```bash
cd /home/max/repositories/MSIAutoEncoderWrapper
uv run --extra cu118 python -m msi_autoencoder_wrapper.analysis.precompute \
  --settings assets/experiments/autoencoder_architecture/notebooks/segmentation_model/20_09_metaspace_pretrain/analysis_settings.yaml \
  2>&1 | tee assets/experiments/autoencoder_architecture/notebooks/segmentation_model/20_09_metaspace_pretrain/shared_precompute_parts_0_6.log
```

To inspect cache status without starting computation:

```bash
uv run python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_cache \
  --settings assets/experiments/autoencoder_architecture/notebooks/segmentation_model/20_09_metaspace_pretrain/analysis_settings.yaml
```

## Runtime and throughput

The complete run took approximately 6 hours for inference and 2–3 hours for the analyses. Inference took about 60 seconds per model on axis 200–900 and 93 seconds per model on axis 100–3000. Profiling showed CPU-bound reconstruction metrics, memmap copies, per-bin aggregates and ion-intensity calculations; GPU use was about 2% while one CPU core was fully occupied.

`stage_segmentation` takes about 53 seconds per representative model and stage, for roughly one hour across the campaign. `stage_latent` and `stage_representation_change` each fit a ridge probe twice. The latent battery runs in parallel using the configured `workers` value.

## Main findings

- Pretrained and frozen-head stages worsen nearly all key metrics. The synthetic head does not reach p = 0.5 on real data (F1 = 0), while a linear probe on the same representation performs about as well as the baseline. This points to the head as the bottleneck.
- The unfrozen stage is mostly inconclusive. Spectral angle is worse for every variant with single-bin spectra except P1. Held-out W1 improves on axis 100–3000 for P1, P2 joint and P5 joint.
- Rare-class and spectral-overlap quotas have no measurable effect after full fine-tuning. Rare synthetic targets are not rare in the real data: they have 726–13,344 training positives.
- Fine-tuning curves do not show faster convergence from pretrained starts: none of 40 paired slope differences is significant. Longer fine-tuning of P1 and joint P2–P5 remains an open test.

## Analysis constraints

Only 97/508 classes on axis 200–900 and 103/508 on axis 100–3000 are eligible for the generator. The rare set contains 25/26 classes. The overlap set contains 8 classes in 4 spectral-bin collision pairs; none of the pairs has at least five held-out pixels, so the held-out image analysis has low power.

Overlap means that classes are annotated at the same spectral bin in the same training pixel. Spatial co-occurrence is a separate secondary question.

## Code and settings

The analysis implementation is in `src/msi_autoencoder_wrapper/analysis/autoencoder/experiments/`:

| Module | Responsibility |
| --- | --- |
| `pretraining_campaign.py` | Settings, campaign inventory, reference tasks and lineage verification |
| `pretraining_campaign_cache.py` | Cache contracts, validation, legacy adoption and status reporting |
| `pretraining_campaign_populations.py` | Population decoding and validation |
| `pretraining_campaign_synthetic.py` | Synthetic reference regeneration and rare/overlap class sets |
| `pretraining_campaign_inference.py` | Per-model inference, result markers and representative selection |
| `pretraining_campaign_precompute.py` | Analysis registry, input contracts and skipping current results |
| `pretraining_campaign_comparison.py` | Paired comparisons, contrasts, ranking and probe helpers |
| `pretraining_campaign_reports.py` | Part 0 and part 1 analyses |
| `pretraining_campaign_stage_reports.py` | Analyses for pretrained and fine-tuned stages |

Settings are in `analysis_settings.yaml`. They define variants, stages, comparison contrasts, targeted pairs, probe and fine-tuning settings, worker count and batch size.
