# Wybór headów: predictive expanded

Ten katalog zawiera analizę kampanii [`predictive_heads_initial.yaml`](../../experiment_runs_configs/07_09_26_predictive_expanded/predictive_heads_initial.yaml) względem wskazanego [`bce_baseline_experiment.yaml`](../../experiment_runs_configs/05_09_26_contractive_expaned/bce_baseline_experiment.yaml). Celem jest wybór headów do kolejnych eksperymentów na podstawie predykcji, generalizacji, rekonstrukcji i geometrii reprezentacji.

**Kampania:** `predictive-heads-initial-20260908-01`, 60 ukończonych modeli oraz 5 modeli historycznego baseline’u. Pliki `part_0_*` i ich dotychczasowe wyniki pozostają analizą populacji i reguły dowodu, poprzedzającą wybór headów. Część 8 łączy ich klasy z wynikami predykcji. Zakres wykonanych testów i komendy znajdują się w [VALIDATION.md](VALIDATION.md).

## Uruchomienie po pobraniu wyników

### Pliki wejściowe

Pobierz komplet workspace lub poniższe części z zachowaniem struktury:

| Artefakt | Lokalizacja / zastosowanie |
|---|---|
| Manifesty zadań | `configs/entropy-runs/<run>/plan/status/task_*.yaml` lub `configs/execution/<campaign>/status/task_*.yaml`; identyfikacja warunków, statusów i rzeczywistych seedów |
| Konfiguracje modeli | `models/kidney/<model>/config/config.json`; zapisane przypisania train/validation/test, architektura i preprocessing |
| Wagi | `models/kidney/<model>/config/weights.pt`; zapisany wynik treningu, zwykle po restore_best |
| Historie | `models/kidney/<model>/config/history.json`; dynamika treningu i diagnostyka checkpointu |
| Widma | `datasets/kidney/kidney.imzML` **oraz** `kidney.ibd`; same wagi i historia nie wystarczą do analizy per piksel |
| Adnotacje | `datasets/kidney/kidney.sqlite` i pozostałe pliki wskazane przez zapisany annotation_reader; dokładna kopia danych użytych w treningu |

Nie zmieniaj nazw modeli ani nie łącz zawartości różnych runów w jednym katalogu statusów. Czytnik obsługuje nazwy z `runtime.model_name` oraz `<run>__<task_id>`. Nie korzysta ze zdalnego `result.model_path`, jeżeli potrafi znaleźć lokalny model w model store.

### Jedna konfiguracja dla wszystkich notebooków

Edytuj [`analysis_settings.yaml`](analysis_settings.yaml):

1. Ustaw `workspace` i `model_store`, jeżeli dane są w innym miejscu. Ścieżki względne są liczone od głównego katalogu repozytorium.
2. `predictive_initial.status_directory` wskazuje pobrany `predictive-heads-initial-20260908-01/plan/status`. Przy analizie innej kampanii zmień tę ścieżkę; `null` włącza wykrywanie po nazwie eksperymentu i odrzuca niejednoznaczny wybór.
3. Źródło `historical_bce` domyślnie wskazuje lokalny run `bce-baseline-20260905-01`. Sprawdź ten katalog po pobraniu danych na innym komputerze.
4. Ustaw `device: cuda`, jeśli używasz środowiska z odpowiednią instalacją PyTorch/CUDA. `cpu` działa bez GPU. Nie zmieniaj zależności w trakcie kampanii.
5. Pozostaw `pixel_fraction: 1.0` dla analizy końcowej. Mniejsza wartość służy sprawdzeniu uruchomienia, zmienia częstości klas i zakres wnioskowania. `geometry_sample_size: 512` ogranicza tylko obliczenia parowe geometrii.

Ścieżki zapisane na węźle treningowym są przenoszone do lokalnego workspace w **tymczasowej kopii** konfiguracji. Oryginalny config i wagi nie są modyfikowane. Dla niestandardowych lokalizacji użyj `path_remap`, np. mapowania starego prefiksu katalogu na nowy.

Można wskazać inną konfigurację przez zmienną `MSI_PREDICTIVE_SETTINGS`; przydatne do osobnego cache dla próbnego przebiegu. Domyślny plik i wszystkie notebooki powinny być uruchamiane z tego samego checkoutu biblioteki.

### Kolejność

Uruchom Jupyter z głównego katalogu repozytorium, w skonfigurowanym środowisku projektu:

```bash
.venv/bin/jupyter lab assets/experiments/autoencoder_architecture/notebooks/07_09_26_predictive_expanded
```

| Część | Pytanie | Główne wyniki |
|---|---|---|
| [1. Audyt](part_1_campaign_audit.ipynb) | Czy porównujemy właściwe, ukończone modele? | grid, duplikaty, źródła, inventory, historie |
| [2. Wspólna inferencja](part_2_shared_inference.ipynb) | Czy modele oceniono na identycznych danych? | trwały cache, klasy, próbki, provenance |
| [3. Predykcja i generalizacja](part_3_prediction_generalization.ipynb) | Które heady dobrze szeregują etykiety, także poza train? | AP/AUC per klasa i częstość, luka na wspólnych klasach, score P/N/U |
| [4. Heady i negatywy](part_4_head_negative_comparisons.ipynb) | Jak duże i stabilne są różnice? | różnice sparowane, przedziały, mapa interwencji |
| [5. Rekonstrukcja](part_5_reconstruction.ipynb) | Czy lepsza predykcja kosztuje jakość rekonstrukcji? | błędy per widmo i m/z, q90/q99, przykłady, kompromis AP–Masserstein |
| [6. Geometria](part_6_latent_geometry.ipynb) | Jak zmienia się latent i dostępność informacji dla wspólnego odczytu? | rank, widmo wariancji, normy, odległości, kNN, CKA, ridge probe |
| [7. Wybór headów](part_7_head_selection.ipynb) | Co przechodzi do kolejnego eksperymentu? | ranking walidacyjny, front Pareto i ograniczenia kompletności |
| [8. Grupy klas](part_8_class_stratification.ipynb) | Dla których klas predykcja się poprawia? | rare/common/undetectable z części 0 i train, wsparcie treningowe, rozdzielność dowodu, zakresy m/z, pokrycie i różnice względem baseline’u |

`Run All` w części 1 działa bez nowej kampanii i pokaże jej brak. Część 2 wymaga dostępności obu wymaganych źródeł oraz co najmniej jednego kompletnego modelu. Nie czeka na trenujące zadania. Zapisze wyniki dostępnych modeli; warunki bez pięciu seedów nie wejdą do shortlisty.

Po części 2 części 3–8 można uruchamiać niezależnie. Nie powtarzają inferencji. Część 8 korzysta dodatkowo z istniejących `class_prevalence.csv`, `class_regimes.csv` i `class_separation.csv` z części 0. Po pobraniu kolejnych checkpointów uruchom ponownie 1 i 2: ukończone obliczenia o zgodnych hashach zostaną wykorzystane ponownie. Po zmianie kodu, wersji bibliotek lub ustawień wykonaj ponownie 2; stary cache zostanie odrzucony.

## Jak czytać wynik

Najpierw sprawdź inventory i pokrycie seedów. Potem oceń **walidacyjne AP w obu populacjach**, rozkład per klasa i lukę train–validation. W części 4 sprawdź wielkość efektu w każdej parze, nie tylko znak średniej. W części 5 zwróć uwagę na q99 i pogorszenie konkretnych widm, nawet gdy średnia pozostaje podobna.

Geometria pomaga wyjaśnić różnice: zmiana efektywnego wymiaru, norm lub sąsiedztw nie ma z góry korzystnego kierunku. Jeśli główny head poprawia wynik, ale wspólny ridge probe nie, to jest przesłanka do zbadania roli odczytu; nie jest to dowód, że encoder się nie zmienił. CKA bliskie 1 także nie gwarantuje identycznej funkcjonalności.

Część 7 porządkuje warunki według **validation annotation-retrieval AP**. Front Pareto uwzględnia dodatkowo walidacyjny Masserstein. Kolumny testowe są sprawdzeniem przeniesienia wyniku i nie wpływają na wybór. Nie ma automatycznej arbitralnej granicy „dopuszczalnego pogorszenia” rekonstrukcji: należy ją uzasadnić zastosowaniem przed kolejnym eksperymentem.

Interpretację formalną, ograniczenia i definicje zawiera [METHODOLOGY.md](METHODOLOGY.md).

## Co faktycznie zawiera ten run

W aktualnym YAML jest **12 wpisów × 5 seedów = 60 zadań**, ale tylko **8 unikalnych warunków**. Zduplikowane są PN-BCE unweighted, PN-BCE global, PN-BCE per-class oraz CE. Zduplikowane zadania zachowujemy do audytu powtarzalności, a w porównaniach uśredniamy w obrębie tego samego warunku i seedu. Nie tworzą dziesięciu niezależnych powtórzeń.

| Rodzina | Unikalne warunki | Co można porównać |
|---|---:|---|
| PositiveWeighted BCE | 2 | global/per-class square-root weighting, U traktowane jako negatywy |
| SignalMasked PN-BCE | 3 | none/global/per-class; U pomijane, N z dowodu widmowego |
| ThreeState CE | 1 | nieważona CE na N/P/U; ranking dodatniego score bez progu decyzyjnego |
| VPU | 1 | `negative_weight=0`, `consistency_weight=0` |
| Symmetric PU-ranking | 1 | `temperature=1`, `pairs_per_sample=32` |
| Historyczny baseline | osobne 5 zadań | ClassBalanced BCE, train inverse prevalence, max weight 20, balanced class mean |

**CE „bez progu” oznacza tutaj brak progu decyzyjnego w ocenie predykcji.** Sam podany YAML zawiera próg dowodu widmowego 0.0119 i radius=1 do konstruowania stanów N/P/U. Analiza tego nie usuwa ani nie przedstawia jako CE trenowanej bez reguły dowodu.

Brakuje VPU ± dodatkowa kara N, PU-ranking ± N oraz nnPU. Nie można wywnioskować przyrostu po dodaniu negatywów do PU z porównania dwóch różnych rodzin. Nie ma też sweepu kontraktywności, kontrastywności ani pretraining. Bieżący eksperyment służy wyborowi headów; interakcje z tymi komponentami wymagają następnego etapu.

## Wyniki i odtwarzalność

Każdy notebook zapisuje `part_<id>_<topic>_results/*.csv` oraz figury PNG. Wyniki pojedynczych zadań i klas pozostają dostępne; agregaty ich nie zastępują. `sample_indices.csv` przechowuje pozycje w materializowanej próbce i w zapisanym podziale. **split_position nie jest automatycznie identyfikatorem piksela w imzML**: odwzorowanie wynika z zapisanych assignments i datasetu.

Cache w `data/kidney_workspace/cache/predictive_heads_analysis` przechowuje tabele, próbki latentów do CKA i metadane, zamiast pełnych logitów każdego modelu. Odczyt widm jest współdzielony. Koszt parowej geometrii wynosi O(S²), gdzie S=512; normy i wariancja korzystają z pełnego analizowanego podziału. Domyślna analiza nie wymaga losowego bootstrapu pikseli.

Zachowaj razem pobrane artefakty, `analysis_settings.yaml`, cache i checkout kodu. Kopia tych samych plików adnotacji jest warunkiem poprawnej interpretacji: zapisane modele zawierają liczbę klas, lecz nie pełny historyczny katalog nazw klas. Analiza zapisuje odtworzony katalog i sprawdza zgodność rozmiarów, ale nie może udowodnić historycznej tożsamości katalogu po podmianie pliku adnotacji.

## Ograniczenia i następny eksperyment

- Pixel split mierzy przenoszenie na inne piksele w tej samej populacji; nie dowodzi generalizacji między akwizycjami ani pacjentami. `group_fields: dataset_id` w konfiguracji grupuje procedurę proporcjonalną i nie stanowi deklaracji holdoutu całych akwizycji.
- AP względem adnotacji nie jest AP względem pełnej prawdy biologicznej. Operacyjne N są określone heurystyką widma i mogą faworyzować heady trenowane tą samą regułą.
- Pięć seedów daje ograniczoną precyzję. Przedziały dla wielu par są eksploracyjne i bez korekcji wielokrotnych porównań.
- Równa waga head loss 0.2 nie wyrównuje skali gradientu różnych funkcji celu. Krótki limit 15 epok i wybór checkpointu według własnego objective mogą premiować szybciej uczące się warianty.
- Historyczny baseline buduje jeden head, nowa kampania buduje oba. Te same numery seedów nie gwarantują identycznego zużycia strumienia RNG ani identycznych parametrów wszystkich komponentów; należy interpretować parowanie jako wspólny plan seedów.

Po shortlistowaniu warto przeprowadzić: **VPU ± N**, analogiczne kontrolowane porównanie PU-ranking, następnie **wybrane heady × ustalona kontraktywność**, osobno kontrastywność i pretraining. Zachowaj tę samą architekturę, dane i budżet treningu oraz zweryfikuj wynik na niezależnej akwizycji. Do tezy o geometrii ułatwiającej optymalizację potrzebny jest eksperyment interwencyjny lub pomiary dynamiki, nie sama korelacja końcowego ranku z AP.
