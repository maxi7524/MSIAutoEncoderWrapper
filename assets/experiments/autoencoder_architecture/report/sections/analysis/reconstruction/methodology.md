#### Cel analizy

Porównanie architektur autoenkodera dla rekonstrukcji widm oraz ustalenie,
czy wybrana architektura zachowuje użyteczne własności przy różnych szerokościach
binów.

#### Dane i reprezentacja wejściowa

Analiza używa danych z tkanki nerki. Konfiguracja kampanii wybiera warstwowo
losowy podzbiór `10%` pikseli z ziarnem `42`, wykonuje liniowy binning w zakresie
`[200, 900]` i normalizację TIC.

#### Porównywane warianty

Porównywane są trzy architektury o wymiarze przestrzeni ukrytej równym `10`:

- `mlp-ae-512-latent-10`;
- `mlp-ae-512-256-latent-10`;
- `conv1d-ae-32-16-8-latent-10`.

Każda architektura jest uruchamiana dla szerokości binu `0.45`, `0.50`, `0.55`
i `1.00` oraz dla pięciu powtórzeń.

#### Protokół treningowy

Aktualna konfiguracja kampanii definiuje podział `0.8 / 0.1 / 0.1`, batch size
`64`, optymalizator AdamW (`lr = 0.001`, `weight_decay = 0.0001`), maksymalnie
`10` epok, cierpliwość wczesnego zatrzymania `10` oraz odległość Massersteina
jako jedyny człon rekonstrukcyjny.

> Uwaga: starszy raport opisuje niektóre przebiegi jako 50-epokowe. Przed
> interpretacją wyników należy potwierdzić dla każdego artefaktu, z której
> konfiguracji pochodzi.

#### Kryteria oceny

Analiza ocenia błąd rekonstrukcji, zachowanie kształtu widm, złożoność modelu
oraz czas treningu. Wyniki i ich interpretacja należą wyłącznie do
`analysis.md`.

#### Artefakty

- [Konfiguracja kampanii](../../../../experiment_runs_configs/13_08_26_architecture_and_binning/architecture_binning_experiment.yaml)
- [Notebook rekonstrukcji i architektur](../../../../notebooks/13_08_26_architecture_and_binning/reconstruction_architecures_analysis.ipynb)
- [Niezmieniony raport źródłowy](../../../source_material/13_08_26_architecture_and_binning/report_modele/reconstruction_methodology.md)
