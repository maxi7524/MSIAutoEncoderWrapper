# Walidacja implementacji

Stan sprawdzony 2026-09-08. Analiza kodu bazuje na lokalnej gałęzi `tests_ae_and_new_binning`, commit `90e1ea6670e7e9063c21f3992a0fc292ee9209b0`, z nowymi plikami opisanymi poniżej. Próba `git ls-remote origin refs/heads/tests_ae_and_new_binning` nie powiodła się: DNS nie rozwiązał `github.com`; gałąź nie została odświeżona.

## Dodane pliki

- Siedem notebooków `part_1_campaign_audit` … `part_7_head_selection`, z pustymi wynikami wykonania.
- `analysis_settings.yaml`, `README.md`, `METHODOLOGY.md` i ten rejestr walidacji.
- `analysis/autoencoder/experiments/predictive_campaign.py`: identyfikacja warunków, migracja ścieżek w kopii configu, audyt manifestów i coverage.
- `analysis/autoencoder/experiments/predictive_precompute.py`: odtworzenie danych, wspólna inferencja, cache, provenance i eksport miar rekonstrukcji.
- `analysis/autoencoder/experiments/predictive_reports.py`: jednostki eksperymentalne, porównania sparowane, CKA i wybór na walidacji.
- `analysis/autoencoder/heads/predictive_comparison.py`: score binarne/CE, dwie populacje oceny, klasy i dopasowane luki generalizacji.
- `analysis/autoencoder/latent/predictive_geometry.py`: rank, normy, odległości, sąsiedztwa i liniowy probe.
- `visualization/predictive.py`: wspólne wykresy korzystające z istniejącego systemu stylów.
- Cztery pliki `tests/analysis/test_predictive_{comparison,geometry,reports,precompute}.py`.

Ścieżki modułów są względne do `src/msi_autoencoder_wrapper`. Nie dodano zależności ani zmian do konfiguracji trenującej kampanii.

## Wykonane sprawdzenia

### Testy jednostkowe i przepływu

```bash
MPLCONFIGDIR=/tmp/msi-matplotlib MPLBACKEND=Agg .venv/bin/python -m pytest -q \
  tests/analysis/test_predictive_comparison.py \
  tests/analysis/test_predictive_geometry.py \
  tests/analysis/test_predictive_reports.py \
  tests/analysis/test_predictive_precompute.py \
  tests/analysis/test_campaign_reader.py \
  tests/analysis/test_entropy_status_reader.py \
  tests/analysis/test_sweep_evaluation.py \
  tests/analysis/test_molecular_head_metrics.py \
  tests/analysis/test_reconstruction_metrics.py \
  tests/analysis/test_sphere_geometry.py
```

Wynik szerszego zestawu: **112 passed**. Obejmuje istniejące testy odczytu kampanii, geometrii i metryk oraz nowe testy:

- zgodności dodatniego score CE z istniejącym softmax P, odporności na duże logity, remisów i niedostępnych wpisów;
- jawnego wykluczania klas bez możliwości oceny rozróżniania oraz dopasowania klas przed liczeniem luki;
- geometrycznych przypadków o znanym ranku, collapse i niezmienniczości CKA;
- trenowania probe wyłącznie na train;
- niedoliczania duplikatów jako seedów, blokowania niewłaściwego parowania i braku wpływu testu na shortlistę;
- rzeczywistego układu plików manifest/config/weights/history, przenoszenia ścieżek w kopii, wykrywania nakładających się splitów i niezgodnego configu;
- inferencji obu kształtów headu, wznawiania cache, naprawy niekompletnego cache oraz odrzucenia zmienionych wag/ustawień;
- wykonania **wszystkich komórek kodu siedmiu notebooków** na deterministycznej miniaturowej kampanii, wraz z obliczeniami, tabelami, wykresami i eksportem CSV.

Test całego przepływu zastępuje wyłącznie odtworzenie dużego datasetu i konstrukcję pobranego modelu małą deterministyczną siecią i danymi. Nie zastępuje testowanych metryk, inferencji, łączenia tabel, selekcji ani wykresów. Po końcowym doprecyzowaniu obsługi znaczników kompletności ponownie uruchomiono cztery nowe moduły testowe.

### Rzeczywisty checkpoint baseline’u

Wykonano `inventory` i `precompute` dla `historical_bce/task_000000` z pobranego runu `bce-baseline-20260905-01`, z ustawieniami próbnymi `pixel_fraction=0.002`, `batch_size=32`, `geometry_sample_size=8`, CPU i cache w `/tmp/msi-predictive-baseline-smoke`.

Przepływ odtworzył reader, adnotacje, binner oraz zapisane assignments z prawdziwego configu. Oceniono **67/33650** widm train, **8/4209** validation i **8/4205** test, dla **508 klas**. Powstały tabele predykcji, rekonstrukcji, geometrii oraz probe. Ta próba potwierdza integrację z realnymi artefaktami; nie jest oceną pełnego baseline’u ani porównaniem nowej kampanii. Końcowe dopracowanie histogramów i ochrony cache zweryfikowano testami deterministycznymi.

### Jupyter i pliki

`part_1_campaign_audit.ipynb` wykonano również przez `nbclient.NotebookClient` w rzeczywistym lokalnym kernelu Jupyter: **3 komórki kodu, bez błędu**. Pierwsza próba wewnątrz sandboxu nie mogła otworzyć lokalnego gniazda; wykonanie poza tym ograniczeniem zakończyło się poprawnie. Wyniki zapisywano wyłącznie do `/tmp`, bez dopisywania ich do dostarczonych notebooków.

Dodatkowo wykonano walidację schematu nbformat, parsowanie AST nowych plików Python i komórek notebooków, kontrolę lokalnych linków Markdown, pustych outputs/execution_count oraz pustych sekcji użytkownika `Notes`/`Person`. Przejrzano przykładowe wygenerowane wykresy widm i rozkładów odległości. `git diff --check` nie zgłosił problemów; nowe, nieśledzone pliki sprawdzono także bezpośrednio pod kątem końcowych spacji.

## Granice walidacji

Nie wykonano pełnej analizy nowej kampanii: jej artefakty nie były jeszcze dostępne. W notebookach nie ma wymyślonych wyników ani automatycznych deklaracji przewagi headów. Nie sprawdzono GPU; środowisko testowe ma PyTorch 2.7.1+cpu.

Nie uruchamiano pełnego zestawu wszystkich testów repozytorium ani budowania dokumentacji Sphinx: zmiany dotyczą izolowanych modułów analizy i notebooków, a sprawdzono ich przepływ i powiązane istniejące testy. Repozytorium nie deklaruje skonfigurowanego formattera, lintera ani statycznego checkera typów; Black i Ruff nie są zainstalowane w lokalnym środowisku. Zastosowano kontrolę składni, schematów, formatowania końców linii i testy wykonania.

Wyniki naukowe pozostają zależne od identyczności pobranych adnotacji, kompletności seedów, zgodności zapisanych kontraktów oraz ograniczeń pixel split opisanych w [METHODOLOGY.md](METHODOLOGY.md).
