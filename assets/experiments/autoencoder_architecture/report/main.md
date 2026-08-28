---
format:
  html:
    toc: true
    toc-depth: 5
    number-sections: true
  pdf:
    toc: true
    number-sections: true
    keep-tex: true
---

# Raport z analiz architektury autoenkodera

## Wprowadzenie

{{< include sections/introduction/introduction.md >}}

## Podstawy teoretyczne

### Autoenkoder i rekonstrukcja widma

{{< include sections/theory/part_id_01_autoencoder_reconstruction.md >}}

### Binning i inverse binning

{{< include sections/theory/part_id_02_binning.md >}}

### Geometria przestrzeni ukrytej

{{< include sections/theory/part_id_03_latent_geometry.md >}}

### Uczenie predykcyjne i klasy wieloetykietowe

{{< include sections/theory/part_id_04_predictive_learning.md >}}

### Autoenkoder kontraktywny

{{< include sections/theory/part_id_05_contractive.md >}}

### Uczenie kontrastywne

{{< include sections/theory/part_id_06_contrastive.md >}}

### Uczenie positive-unlabelled i nnPU

{{< include sections/theory/part_id_07_nnpu.md >}}

### Porównywanie ablacjami

{{< include sections/theory/part_id_08_ablation_comparisons.md >}}

## Analizy wyników

### Binning oraz podstawowa architektura 

{{< include sections/analysis/binning_and_base_AE_architecture/methodology.md >}}

{{< include sections/analysis/binning_and_base_AE_architecture/analysis.md >}}

### Kontraktywność

### Kontrastywność

### Rekonstrukcja widm

{{< include sections/analysis/reconstruction/methodology.md >}}

{{< include sections/analysis/reconstruction/analysis.md >}}

### Porównanie rekonstrukcji 

### Predykcyjność reprezentacji

### Przestrzeń ukryta

### nnPU

## Synteza i decyzje

## Załączniki

### Artykuły

[Indeks artykułów](articles/README.md)

### Notebooki

- [Binning bez modelu — 13_08_26](../notebooks/13_08_26_architecture_and_binning/binning_analysis_no_model.ipynb)
- [Rekonstrukcja i architektury — 13_08_26](../notebooks/13_08_26_architecture_and_binning/reconstruction_architecures_analysis.ipynb)
- [Zbiór notebooków predykcyjnych — 23_08_26](../notebooks/23_08_26_architecture_predictive/)

### Konfiguracje i materiały źródłowe

- [Konfiguracje przebiegów](../experiment_runs_configs/)
- [Raporty, wykresy i wyniki przeniesione bez edycji](source_material/)
