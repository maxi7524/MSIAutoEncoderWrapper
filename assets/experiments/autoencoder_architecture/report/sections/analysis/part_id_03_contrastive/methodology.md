#### Cel analizy

Celem jest określenie, czy człon InfoNCE kształtuje reprezentację w sposób odmienny od modelu bazowego oraz czy wybór pików podlegających permutacji ma znaczenie dla rekonstrukcji, predykcji i geometrii reprezentacji. Analiza rozdziela efekt samego członu kontrastywnego od efektu kary kontraktywnej i ważenia negatywów.

#### Projekt eksperymentu

Wszystkie warianty używają tej samej architektury `conv1d-ae-32-16-8-latent-10`, tych samych danych, podziału, rekonstrukcji Massersteina i głowy `balanced_bce`. Każdy wariant ma pięć powtórzeń. Różnice między wariantami wynikają wyłącznie z konfiguracji funkcji celu.

##### Główne porównania kontrastywne

| wariant | grid | dodatkowy człon | pytanie |
|---|---|---|---|
| `bce` | `grid_0000` | brak | punkt odniesienia |
| `peak_random` | `grid_0002` | InfoNCE z `permutation_random` | jaki jest efekt kontrastywności bez ochrony anotowanych pików? |
| `peak_label_invariant` | `grid_0003` | InfoNCE z `permutation_label_invariant` | jaki jest efekt kontrastywności, gdy anotowane piki są chronione? |

Porównania `grid_0000` z `grid_0002` i `grid_0003` izolują obecność InfoNCE. Porównanie `grid_0002` z `grid_0003` izoluje regułę wyboru pików dla tej samej wagi InfoNCE.

##### Porównania uzupełniające

| wariant | grid | dodatkowe czynniki | zastosowanie |
|---|---|---|---|
| `contractive` | `grid_0001` | kara kontraktywna | kontrola wpływu kary kontraktywnej |
| `contractive + label_invariant` | `grid_0004` | kara kontraktywna i InfoNCE | sprawdzenie interakcji z karą kontraktywną |
| `contractive + label_invariant + jaccard` | `grid_0005` | jak wyżej oraz ważenie negatywów Jaccarda | sprawdzenie wpływu ważenia negatywów |

Wariantów `grid_0004` i `grid_0005` nie wolno porównywać bezpośrednio z `grid_0000` jako czystego efektu kontrastywności, ponieważ zawierają również karę kontraktywną. Efekt kary kontraktywnej ocenia porównanie `grid_0003` z `grid_0004`, a efekt ważenia negatywów porównanie `grid_0004` z `grid_0005`.

#### Dane i warunki stałe

Analiza używa znormalizowanych metodą TIC widm nerki, po liniowym binningu w zakresie $[200, 900]$ przy `bin_step = 0.55`. Kampania wykorzystuje warstwowy podzbiór `10%` pikseli z ziarnem `42` oraz grupowy podział według `dataset_id` w proporcji `0.8 / 0.1 / 0.1`. Do porównań między wariantami należy używać tych samych identyfikatorów pikseli z odpowiedniego splitu.

#### Trening i konstrukcja widoków

Każdy model jest trenowany przez maksymalnie `15` epok z batch size `64`, optymalizatorem AdamW (`lr = 0.001`, `weight_decay = 0.0001`), `patience = 10` i `gradient_clip_norm = 5.0`. Rekonstrukcja Massersteina ma wagę `1.0`, głowa klasyfikacyjna `0.2`, a InfoNCE `0.1` przy temperaturze `0.07`.

W InfoNCE dla każdego widma konstruowany jest drugi widok przez permutację trzech pików z banku preobliczonych permutacji. `permutation_random` wybiera piki niezależnie od anotacji. `permutation_label_invariant` chroni bin odpowiadający anotacji, z `annotation_bin_radius = 0`. Oba warianty zachowują normalizację wejścia. W wariancie `grid_0005` pary o nakładających się anotacjach pozostają negatywami, ale ich wkład do mianownika InfoNCE ma wagę `0.25`.

#### Protokół oceny

##### Rekonstrukcja i predykcja

Dla każdego powtórzenia należy raportować metryki rekonstrukcji oraz metryki predykcji dla tego samego splitu testowego. Wyniki należy prezentować jako wartości dla pięciu powtórzeń, z miarą położenia i rozrzutu. Wnioski o efekcie InfoNCE można wyciągać tylko z porównań, w których zmienia się jeden wskazany czynnik.

##### Reprezentacja ukryta

Geometrię enkodera należy oceniać na kanonizowanej reprezentacji $u = (z - \beta) / \gamma$, a nie na surowym $z$, ponieważ parametry afiniczne końcowej `LayerNorm` są uczone osobno dla każdego modelu. Dla identycznie uporządkowanych pikseli testowych należy porównywać odległość Procrustesa, linear CKA i nakładanie sąsiedztw $k$NN, uzupełnione miarami wykorzystania wymiarów i asymetrii chmury.

##### Przestrzeń projektora

Przestrzeń $g(z)$ należy analizować osobno i wyłącznie dla `grid_0002` do `grid_0005`, ponieważ tylko te warianty trenują projektor przez InfoNCE. Projektor w `grid_0000` i `grid_0001` nie uczestniczy w funkcji celu, więc jego wyjście nie jest poprawnym punktem odniesienia.

##### Niepewność między powtórzeniami

Porównania między wariantami wykonuje się parami dla tego samego indeksu powtórzenia. Wielkość sygnału należy odnieść do rozrzutu między wszystkimi parami powtórzeń wewnątrz każdego wariantu. Wynik mieszczący się w tym rozrzucie nie stanowi wykrywalnego efektu przy pięciu powtórzeniach.

#### Zakres odroczony

Miary alignment i uniformity oraz krzywa selektywności kontrastywnej wymagają użycia dokładnie tej samej, stanowej augmentacji permutacji pików, której używa `InfoNCELoss`. Nie należy ich liczyć na zastępczej augmentacji. Do czasu udostępnienia tej samej implementacji pozostają poza zakresem tej analizy.

#### Artefakty

- [Konfiguracja kampanii](../../../../experiment_runs_configs/23_08_26_architecture_predictive/architecture_predictive_experiment.yaml)
- [Notebook geometrii wariantów BCE i kontrastywnych](../../../../notebooks/23_08_26_architecture_predictive/part_7_latent_bce_vs_all.ipynb)
- [Teoria kontrastywności](../../theory/part_id_06_contrastive.md)
