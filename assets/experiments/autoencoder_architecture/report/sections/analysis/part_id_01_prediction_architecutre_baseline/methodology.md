#### Zakres analizy i jednostka porównania

Część obejmuje trzy niezależne odczyty baseline'u predykcyjnego `grid_0000`: rekonstrukcję widm, jakość wieloetykietowej predykcji oraz geometrię reprezentacji ukrytej. Nie wszystkie odczyty mają tę samą jednostkę porównania. Rekonstrukcja i geometria zestawiają pojedyncze checkpointy `rep_00` modelu AE-only z kampanii `kidney-architecture-binning` oraz modelu BCE z kampanii `kidney-architecture-predictive`; ponieważ kampanie mają różne splity, używane są wyłącznie identyfikatory pikseli wspólne dla obu modeli. Predykcja jest liczona dla pięciu powtórzeń `grid_0000` na pełnym splicie testowym kampanii predykcyjnej. Wyniki geometrii z tego rozdziału są więc porównaniem dwóch konkretnych checkpointów.

#### Rekonstrukcja widm

##### Krzywe treningowe i walidacyjne

Dla każdego przebiegu i każdej epoki z `history.json` pobierana jest wartość `masserstein` dla treningu i walidacji. Powstaje jedna obserwacja na parę `(kampania, powtórzenie, epoka)`. Obie kampanie używają tego samego kryterium Massersteina, ale checkpoint jest wybierany po minimalnej całkowitej stracie walidacyjnej. W AE-only całkowita strata jest równa Massersteinowi; w `grid_0000` zawiera również składnik BCE. Wartość rekonstrukcji w wybranym checkpoincie oznacza zatem błąd Massersteina w momencie optymalnym dla całego celu, a nie osobne minimum błędu rekonstrukcji.

Masserstein jest liczony osobno dla każdego widma na uporządkowanej osi $m/z$. Po doprowadzeniu obu widm do reprezentacji TIC liczona jest skumulowana różnica intensywności, a następnie suma jej wartości bezwzględnych ważona szerokościami kolejnych binów. Miara odpowiada masie, którą trzeba przesunąć wzdłuż osi $m/z$, aby rekonstrukcja zgadzała się z wejściem. Mniejsza wartość oznacza lepsze zgodne położenie i rozkład intensywności; nie jest to zwykła różnica punkt po punkcie.

Historyczny wariant AE-only miał błąd implementacyjny: wyjście dekodera nie było normalizowane TIC. Z tego powodu jego liczby Massersteina i porównanie średnich między kampaniami nie są ilościowo porównywalne z modelem BCE. Krzywe służą tu do zidentyfikowania problemu i jego charakteru, a nie do wniosku o przewadze jednego modelu nad drugim.

##### Widma i mapa błędu

Dla wspólnych pikseli porównywane są wejście i rekonstrukcja w pełnym zakresie oraz w lokalnym oknie wokół najsilniejszego binu danego widma. Widok pełnego zakresu ujawnia przesunięcia masy i szerokie różnice obwiedni, a widok lokalny pozwala ocenić położenie, szerokość i wysokość pojedynczych pików. Obrazy nie są średnią po pikselach; przedstawiają konkretne, sparowane widma.

Mapa błędu zawiera jeden koszt Massersteina na piksel, policzony tą samą implementacją co kryterium treningowe z `reduction = none`, a następnie przypisany do natywnej współrzędnej obrazu MSI. Jednolita skala barw dla modeli pozwala lokalizować obszary systematycznie wysokiego błędu. Mapa nie pokazuje przyczyny błędu; wymaga odczytu razem z widmami lokalnymi.

#### Predykcja molekuł

##### Dane liczone przez głowę

Dla każdego piksela głowa zwraca logity dla 508 klas. Po zastosowaniu sigmoidy otrzymujemy niezależne prawdopodobieństwo każdej klasy, a predykcja progowa jest dodatnia, gdy prawdopodobieństwo jest co najmniej `0.5`. Metryki korzystają z maski dostępności etykiet na poziomie pary `(piksel, klasa)`; brakujące pozycje nie wchodzą do zliczeń.

W tabelach występują dwa zakresy klas. `all classes (508)` zawiera wszystkie wyjścia głowy. `train-active ∩ split-active` zawiera tylko klasy mające co najmniej jeden dodatni przykład zarówno w treningu, jak i w aktualnie ocenianym splicie. Drugi zakres jest właściwy do oceny jakości: klasa nieobecna w treningu nie mogła zostać nauczona, a klasa bez pozytywów w ocenianym splicie nie daje definiowalnego average precision.

##### Metryki rankingowe

`average_precision` jest liczona osobno dla każdej klasy z par `(prawdopodobieństwo, etykieta)` bez progu, a następnie uśredniana po klasach z co najmniej jednym dostępnym pozytywem. Odpowiada polu pod krzywą precision-recall i mierzy, czy pozytywne piksele są ustawiane wyżej niż negatywne. W danych rzadkich jest to główna miara: wartość rośnie, gdy model znajduje dodatnie piksele bez zalewania wyniku fałszywymi pozytywami.

`roc_auc` jest również liczona per klasa bez progu, a następnie uśredniana po klasach mających przynajmniej jeden pozytyw i jeden negatyw. Interpretacja to prawdopodobieństwo, że losowy pozytyw otrzyma wyższy wynik niż losowy negatyw. Miara jest użyteczna jako uzupełnienie average precision, ale przy dużej nierównowadze klas może wyglądać dobrze mimo słabego odzyskiwania rzadkich pozytywów.

##### Metryki przy progu 0.5

`macro_f1` jest średnią F1 policzoną osobno dla klas, dlatego każda klasa ma ten sam wpływ niezależnie od liczności. `micro_precision`, `micro_recall` i `micro_f1` powstają po zsumowaniu wszystkich dostępnych komórek `(piksel, klasa)`, dlatego są silniej kształtowane przez częste klasy. Wysoki micro wynik i niski macro wynik oznacza zwykle, że model działa głównie dla częstych klas.

`hamming_loss` to odsetek błędnych decyzji binarnych po progu `0.5` wśród wszystkich dostępnych komórek. `hamming_loss_baseline_positive_rate` to odsetek rzeczywistych pozytywów, czyli dokładny hamming loss klasyfikatora zawsze zwracającego zero. Hamming loss modelu jest użyteczny tylko w zestawieniu z tym baseline'em; sam niski wynik może wynikać z dominacji etykiet ujemnych.

##### Generalizacja i agregacja

Ocena testowa baseline'u jest wykonywana dla wszystkich pięciu powtórzeń `grid_0000` na tym samym splicie testowym. Wyniki należy czytać przez średnią, odchylenie standardowe i wartości poszczególnych powtórzeń. Porównanie train, validation i test w analizie generalizacji zostało policzone dla `rep_00`; jest diagnostyką luki generalizacyjnej, a nie estymacją jej niepewności między seedami.

#### Reprezentacja ukryta

##### Kanonizacja i odległość kątowa

Każdy kod postaci $z$ jest przekształcany do $u = (z - \beta) / \gamma$, gdzie $\gamma$ i $\beta$ są parametrami końcowej `LayerNorm` danego modelu. Usuwa to uczoną transformację afiniczną, która różni się między modelami. Dla latent dim `D = 10` każdy poprawnie kanonizowany wiersz ma sumę współrzędnych równą zero i normę $\sqrt{10}$, więc punkty leżą na sferze o ośmiu stopniach swobody. Wszystkie miary geometryczne używają $u$, nie surowego $z$.

Dla dwóch kodów odległość jest liczona jako kąt $\theta = \arccos((u_i^\top u_j) / D)$ w stopniach. Na sferze ta definicja jest niezależna od skali wektorów. Mały kąt oznacza podobne położenie, duży kąt oznacza odległe położenie.

##### Rozkład kątów i struktura chmury

Dla każdego modelu i splitu losowanych jest do 20 000 par różnych pikseli. Dla każdej pary liczony jest cosinus kąta $(u_i^\top u_j) / D$, a raport zawiera jego rozkład, średnią i odchylenie standardowe. Punktem odniesienia jest jednorodny rozkład punktów na sferze o wymiarze `D - 2 = 8`, dla którego odchylenie standardowe cosinusa wynosi $1 / \sqrt{D - 1} = 1/3$. Średni cosinus bliski zero i odchylenie bliskie `1/3` oznaczają brak wyraźnej globalnej struktury; dodatnia średnia oznacza skupienie punktów, a odchylenie większe od baseline'u może oznaczać polaryzację lub wiele klastrów, nie większą losowość.

##### Wykorzystanie wymiarów

Z populacyjnej kowariancji `Cov(u)` (`ddof = 0`) wyznaczane jest malejące spektrum wartości własnych. `trace` jest całkowitą wariancją, `effective rank = exp(-Σ p_i log p_i)` wykorzystuje entropię znormalizowanych wartości własnych, a `participation ratio = (Σ λ_i)^2 / Σ λ_i^2` jest drugą miarą efektywnej liczby wymiarów. Większe wartości obu rang oznaczają bardziej równomierne rozłożenie wariancji na wymiarach; małe wartości oznaczają koncentrację na kilku kierunkach. Jedna wartość własna bliska zero jest oczekiwana z powodu warunku sumy współrzędnych równej zero i stanowi kontrolę pipeline'u, nie wynik biologiczny.

`TwoNN intrinsic dimension` jest lokalnym, nieliniowym estymatorem. Dla każdego punktu wyznaczane są kąty do najbliższego i drugiego najbliższego sąsiada, a estymata ma postać $N / \sum_i \log(r_{2,i} / r_{1,i})$. Wartość bliższa ośmiu oznacza wykorzystanie większej części dozwolonej sfery, a mniejsza koncentrację na podrozmaitości. Metoda jest wrażliwa na liczność; wynik dla 42 pikseli testowych ma charakter orientacyjny i nie może być zestawiany liczbowo z wynikiem dla 2662 pikseli treningowych.

##### Zgodność geometrii AE-only i BCE

Porównanie jest sparowane: wiersz o tym samym indeksie oznacza ten sam piksel w obu modelach. `Procrustes distance` centruje i skaluje obie chmury, dopasowuje najlepszy obrót ortogonalny i zwraca pierwiastek z disparity w zakresie od 0 do 1. Zero oznacza tę samą strukturę po obrocie; większa wartość oznacza zmianę globalnego kształtu.

`linear CKA` porównuje scentralizowane macierze podobieństwa obu chmur. Wartość bliska 1 oznacza podobną globalną strukturę liniową, ale nie jest metryką i może pozostawać wysoka mimo widocznej różnicy chmur. Należy ją czytać razem z odległością Procrustesa, nie jako jej zastępstwo.

`k-NN overlap` dla `k = 10` jest średnim odsetkiem wspólnych sąsiadów dla tego samego piksela w obu przestrzeniach. Jeden oznacza identyczne lokalne otoczenia, zero brak wspólnych sąsiadów. `trustworthiness` mierzy, czy sąsiedzi w przestrzeni BCE byli także blisko w AE-only, a `continuity` ten sam warunek w odwrotną stronę; obie miary są w zakresie od 0 do 1. Wysokie trustworthiness i continuity przy niskim overlap oznacza zachowanie szerokiego porządku rang odległości przy lokalnym przetasowaniu sąsiadów. Dla małego wspólnego testu `N = 42` metryki są niestabilne, więc podstawą interpretacji jest treningowy wspólny zbiór `N = 2662`.

##### Czułość enkodera

Dla każdego testowego widma tworzona jest losowa jednostkowa perturbacja $\delta$ i liczony jest kod dla $x + \varepsilon \lVert x \rVert \delta$ dla $\varepsilon \in \{0, 0.01, 0.03, 0.1, 0.3, 1.0\}$. Dla każdej skali raportowany jest średni kąt między $u(x)$ i $u(x + \varepsilon \lVert x \rVert \delta)$. Mniejszy kąt oznacza lokalnie bardziej stabilny enkoder. Jest to miara odpowiedzi na ogólne zaburzenie wejścia, a nie dowód niezmienniczości wobec konkretnej operacji biologicznej lub eksperymentalnej.

##### Relacja z anotacjami

Dla do 20 000 losowych par tych samych pikseli liczony jest kąt w latencie oraz odległość Jaccarda etykiet $1 - \lvert Y_i \cap Y_j \rvert / \lvert Y_i \cup Y_j \rvert$. Raportowany współczynnik Spearmana koreluje rangi obu wielkości, a wartość `p` pochodzi z testu permutacyjnego. Dodatnia korelacja oznacza, że bardziej różne zestawy anotacji mają tendencję do większej odległości kątowej. Nie ustanawia kierunku przyczynowego i sama nie dowodzi, że reprezentacja koduje wyłącznie anotacje, ponieważ podobieństwo widm i podobieństwo anotacji mogą wynikać ze wspólnej struktury danych.

#### Artefakty

- [Notebook rekonstrukcji baseline'u](../../../../notebooks/23_08_26_architecture_predictive/part_1_reconstruction_bce_vs_baseline.ipynb)
- [Notebook metryk predykcji](../../../../notebooks/23_08_26_architecture_predictive/part_2_prediction_metrics_bce.ipynb)
- [Notebook geometrii AE-only i BCE](../../../../notebooks/23_08_26_architecture_predictive/part_3_latent_ae_vs_bce.ipynb)
- [Implementacja metryk geometrii](../../../../../../../src/msi_autoencoder_wrapper/analysis/autoencoder/latent/sphere_geometry.py)
