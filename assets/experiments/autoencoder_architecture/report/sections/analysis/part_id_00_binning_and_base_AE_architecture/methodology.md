#### Zakres analizy i jednostka porównania

Analiza obejmuje kampanię rekonstrukcyjną z trzema architekturami autoenkodera: MLP z jedną warstwą ukrytą, MLP z dwiema warstwami ukrytymi oraz konwolucyjnym autoenkoderem 1D. Dla każdej architektury uruchomiono cztery szerokości binów $\Delta m/z \in \{0.45, 0.50, 0.55, 1.00\}$ i pięć niezależnych powtórzeń, czyli łącznie 60 zadań treningowych. Analiza składa się z dwóch odrębnych etapów: wyboru architektury na podstawie historii walidacyjnej oraz oceny rzeczywistej rekonstrukcji surowych widm przez wybraną architekturę. Te etapy odpowiadają na różne pytania i ich wyników nie należy utożsamiać.

#### Wybór architektury

##### Historia treningu i checkpoint

Dla każdego zadania zapisana jest historia epok z wartościami `train_loss`, `validation_loss`, `best_loss` oraz czasem wykonania. `best_loss` oznacza najniższą dotychczas zaobserwowaną wartość straty walidacyjnej, a nie stratę z ostatniej epoki. Dla każdego zadania do porównania końcowego pobierana jest ostatnia zapisana wartość `best_loss`, czyli wynik checkpointu wybranego przez trening. W tej kampanii funkcją celu rekonstrukcji jest odległość Massersteina, dlatego niższy `best_loss` oznacza lepszą rekonstrukcję w sensie tej funkcji celu na zbiorze walidacyjnym.

##### Agregacja po powtórzeniach

Wyniki są grupowane według pary `(architecture, binning_step)`. Dla pięciu powtórzeń obliczane są `mean_best_validation_loss`, odchylenie standardowe, najlepsza wartość oraz identyfikator zadania o najlepszej walidacji. Średnia służy do porównania stabilności wariantu, natomiast najlepsze zadanie wskazuje konkretny checkpoint używany później do wykresów widm i map przestrzennych. Różnica między wariantami mniejsza od rozrzutu między powtórzeniami nie jest samodzielnym dowodem przewagi architektury. Wynik walidacyjny jest kryterium wyboru modelu, ale nie opisuje jeszcze wprost zgodności z oryginalnym, niezbinnowanym widmem.

#### Ocena binningu przez rekonstrukcję widm

##### Konstrukcja pojedynczego porównania

Po wyborze architektury konwolucyjnej z każdej szerokości binu analizowane są wszystkie pięć jej powtórzeń. Z zapisanego podziału pobierana jest deterministyczna próba 500 widm ze zbioru testowego, z ziarnem `7`. Dla każdego widma $X$ odtwarzany jest binner użyty przez dany model, obliczane jest jego przedstawienie $\mathrm{B}(X)$ i wykonywana rekonstrukcja $\hat{X} = f(\mathrm{B}(X))$. Wejście modelu oraz widmo referencyjne są normalizowane do TIC zgodnie z konfiguracją treningu. Piki widma referencyjnego i rekonstrukcji są dopasowywane jeden do jednego z tolerancją $0.01$ Da, dlatego każda para referencja-rekonstrukcja daje osobny rekord metryki.

##### Metryka Wassersteina w tabeli

Kolumna `wasserstein` w tabeli i na wykresach binningu pochodzi z analizy dopasowanych punktów spektralnych. Jest to jednowymiarowa miara transportu między rozkładami intensywności po osi $m/z$: przeniesienie intensywności na większą odległość zwiększa wynik, więc wartość niższa oznacza lepiej zlokalizowaną i bliższą rekonstrukcję. Nie jest to ta sama liczba co `best_loss` z treningu, mimo że obie miary odnoszą się do odległości Massersteina. `best_loss` pochodzi z funkcji celu liczonej na zbiorze walidacyjnym w trakcie treningu, a `wasserstein` w tabeli opisuje rekonstrukcje surowych widm testowych po dopasowaniu pików. Rozbieżność tych dwóch wyników należy interpretować jako sygnał, że dobry wynik optymalizacji nie gwarantuje jeszcze dobrej zgodności z pierwotnym widmem.

##### Agregacja i interpretacja rozkładu błędu

Dla każdej szerokości binu rekordy `wasserstein` są łączone po 500 widmach testowych i pięciu powtórzeniach modelu. Z tej połączonej populacji liczone są mediana, pierwszy kwartyl `q25`, trzeci kwartyl `q75` i średnia. Mediana opisuje typowy błąd widma, kwartyle opisują jego zmienność między widmami i powtórzeniami, a średnia jest wrażliwa na trudne przypadki o dużym błędzie. Te statystyki nie są przedziałami ufności dla średniej ani pięcioma niezależnymi wynikami eksperymentu, ponieważ widma w obrębie jednego powtórzenia nie są niezależnymi treningami. Do tabeli dołączono `mean_best_validation_loss` wyłącznie jako osobne kryterium wyboru modelu.

Notebook źródłowy pobiera próbę ze zbioru testowego, chociaż nagłówek tabeli i nazwy części plików graficznych w analizie zawierają słowo „treningowych”. Przed sformułowaniem końcowych wniosków należy ujednolicić to oznaczenie w tekście analizy. Metoda użyta do uzyskania tabeli odpowiada widmom testowym, niewidzianym przez model podczas treningu.

#### Inspekcja widm kluczowych

Dla każdej szerokości binu wybierane jest jedno zadanie o najlepszej walidacji, a następnie rekordy testowe są sortowane według `wasserstein`. Pokazywane są po dwa widma z najmniejszym i największym błędem. Wykres zawiera pełny zakres widma oraz lokalne przybliżenie o szerokości 30 Da, wycentrowane na najwyższym piku referencyjnym. Ten krok służy do kontroli jakościowej: pozwala rozróżnić przesunięcie pików, zmianę ich szerokości, utratę obwiedni i błąd intensywności, których sama wartość agregowanej metryki nie wyjaśnia.

#### Przestrzenny rozkład błędu

Mapa globalna przypisuje wartość `wasserstein` pojedynczego modelu do współrzędnych pikseli w odpowiednim podziale danych; piksele spoza analizowanej próby pozostają puste. Wspólna skala kolorów umożliwia wzrokowe porównanie obszarów i splitów. Mapa jest diagnostyką heterogeniczności przestrzennej jednego wybranego checkpointu, a nie rozkładem po pięciu powtórzeniach ani statystycznym testem generalizacji. Ewentualny wniosek o stabilności przestrzennej powinien być potwierdzony rozkładem błędów dla wszystkich powtórzeń.

#### Artefakty

- [Konfiguracja kampanii](../../../../experiment_runs_configs/13_08_26_architecture_and_binning/architecture_binning_experiment.yaml)
- [Notebook analizy rekonstrukcji](../../../../notebooks/13_08_26_architecture_and_binning/reconstruction_architecures_analysis.ipynb)
- [Implementacja analizy rekonstrukcji](../../../../../../../src/msi_autoencoder_wrapper/analysis/autoencoder/binning/model_reconstruction_analysis.py)
- [Implementacja historii treningu](../../../../../../../src/msi_autoencoder_wrapper/analysis/autoencoder/reconstruction/training_dynamics_analysis.py)
- [Teoria autoenkodera i rekonstrukcji](../../theory/part_id_01_autoencoder_reconstruction.md)
- [Teoria binningu](../../theory/part_id_02_binning.md)
