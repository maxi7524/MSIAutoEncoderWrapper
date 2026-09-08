# Metodyka porównania headów

Ten dokument definiuje wykonywane analizy i granice wnioskowania. Instrukcję danych, kolejność uruchamiania i mapę plików zawiera [README.md](README.md).

## Obiekty i populacja

Model realizuje `X → Enc → Z → Dec → X_hat` oraz `Z → head → logits`. X jest widmem po binnerze i normalizacji TIC, Y binarną macierzą obserwowanych adnotacji, A maską dostępności. Wszystkie heady oceniamy na identycznym zestawie próbek i kolejności klas, odtwarzając zapisane assignments. Nie tworzymy nowego podziału z samego seeda.

Pozytywy P to dostępne wpisy z Y=1. Dla pozostałych dostępnych wpisów reguła `SignalEvidencePolicy` oblicza maksimum sygnału w binach jonu, poszerzonych o `bin_radius`, i porównuje je z `max(absolute_threshold, relative_threshold * max(X_i))`. Sygnał **większy** od tej wartości oznacza U; mniejszy lub równy oznacza operacyjne N. Istniejąca adnotacja P ma pierwszeństwo. Niedostępne wpisy mają stan -1 i są wykluczane.

Domyślnie wszystkie modele oceniamy tą samą regułą: absolute=0, relative=0.0119, radius=1. Nie zmieniamy reguły dla kolejnych headów, nawet jeśli ich trening nie używał dowodu. Nie dostrajamy jej do wyniku testowego.

## Ranking bez progu

### Ciągły score

Dla binarnego headu używamy surowego logitu. Dla CE o kolejności N/P/U używamy:

\[
s_{ic}=l_{ic,P}-\log\left(\exp(l_{ic,N})+\exp(l_{ic,U})\right).
\]

Jest to logit dodatniego prawdopodobieństwa softmax, monotoniczny względem `p_P`. Obliczenie przez logsumexp zachowuje porządek przy dużych logitach i nie wprowadza remisów spowodowanych nasyceniem sigmoid. Nie sumujemy P+U i nie stosujemy progu 0.5. Transformacja probabilistyczna używana do diagnostyki P/N/U pochodzi z istniejącego `probabilities_from_logits`.

AP liczymy standardową implementacją scikit-learn jako sumę przyrostów recall razy precision, bez interpolacji trapezowej. ROC AUC korzysta z tych samych ciągłych score. Dokumentacja dopuszcza nieprogowane wartości decyzyjne: [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). Lokalna walidacja implementacji używa zainstalowanego scikit-learn 1.9.0; dokładne wersje zapisywane są w metadanych cache.

### Dwie populacje oceny

| Populacja | Pozytywy | Porównawcze zera | Znaczenie |
|---|---|---|---|
| `annotation_retrieval` | P | wszystkie dostępne Y=0, czyli N i U | odtwarzanie i szeregowanie **obserwowanych adnotacji** |
| `operational_pn` | P | wyłącznie operacyjne N | rozróżnianie P od negatywów określonych regułą sygnału |

Pierwsza populacja może karać biologicznie poprawną predykcję niezaobserwowanego jonu. Druga pomija trudną część U i może częściowo mierzyć odtwarzanie reguły użytej w treningu PN/CE. Przewaga tylko w jednej populacji wymaga wyjaśnienia; nie wolno przenosić jej automatycznie na pełną prawdę molekularną.

### Klasy i agregacja

Do porównawczego macro AP/AUC dopuszczamy klasy mające co najmniej jeden P w analizowanym train oraz co najmniej jeden P i jedno porównawcze zero w ocenianym podziale/populacji. AP dodatniej-only klasy byłoby równe 1 bez jakiegokolwiek rozróżniania; takie klasy zachowujemy w tabeli z `eligible=False` i NaN metryk. Zachowujemy też klasy bez pozytywów i klasy nieobecne w train, wraz z denominatorami.

Średnia macro nadaje równą wagę każdej kwalifikującej się klasie. `ap_above_prevalence` to AP pomniejszone o częstość P w danej populacji — punkt odniesienia dla stałego score, **nie** estymator oczekiwania AP skończonego losowego rankingu. Nie jest głównym kryterium wyboru.

Grupy częstości powstają wyłącznie z train: rare <10 P, medium 10–99, frequent ≥100. To jawne kategorie opisowe, bez strojenia granic pod rezultat. W wyniku przechowujemy liczbę klas dla każdej metryki. Wszystkie modele korzystają z tych samych supportów, jeżeli data contract jest zgodny.

Lukę train–validation i train–test liczymy po dopasowaniu tych samych kwalifikujących się klas, a nie przez odejmowanie dwóch średnich o różnych supportach. Różne częstości etykiet między podziałami nadal mogą wpływać na AP. Dlatego sprawdzamy jednocześnie AP, ROC AUC i liczbę P.

## Interwencje, powtórzenia i niepewność

Identyfikator warunku jest hashem całego criterion mapping. Audyt obejmuje dodatkowo architekturę. Aktualne 60 zaplanowanych zadań daje 8 unikalnych warunków × 5 niezależnych planów seedów, z powtórnie zaplanowanymi czterema warunkami. Duplikaty nie są odrzucane z danych: ich minimum, maksimum i liczba zostają zapisane, a do porównań trafia średnia w obrębie warunku i rzeczywistego seedu.

Pary wymagają zgodności repetition, derived model-initialization seed, derived training seed, preprocessing/dataset contract, encoder/decoder contract i treningu poza head loss. Kontrakt danych obejmuje zapisane assignments i konfigurację readera/adnotacji po przeniesieniu ścieżek. Kontrakt treningu zachowuje fazy, optimizer, budżet, checkpoint i inne kryteria. Nie twierdzimy, że równe seedy gwarantują identyczną realizację wszystkich strumieni losowych.

Dla par Δ_r=a_r−b_r zapisujemy każdy wynik, średnią, medianę i liczbę dodatnich różnic. Opisowy 95% przedział to:

\[
\bar\Delta \pm t_{0.975,n-1}\,s_\Delta/\sqrt n.
\]

Przy jednej parze granice pozostają NaN. Przy braku zgodnych par powstaje rekord z przyczyną, bez estymaty. Przedział zakłada sensowność przybliżenia rozkładu efektów seedowych i jest mało precyzyjny dla n=5. Nie wykonujemy testów na milionach pikseli, które sztucznie zwiększyłyby liczebność. Nie ma bootstrapu pikselowego ani deklaracji istotności po wielu porównaniach.

PN-BCE vs odpowiadające weighted BCE bada pakiet zmiany maski i wag obliczanych z innej populacji P/N. Historyczne ClassBalanced BCE jest odrębnym baseline’em o innym ważeniu i redukcji. VPU bez N może pokazać przewagę rodziny nad baseline’em, lecz obecny run nie identyfikuje efektu dodania N do VPU. Analogiczne ograniczenie dotyczy PU-ranking.

## Rekonstrukcja

Masserstein obliczamy istniejącym `masserstein_distances`, który wywołuje `SpectrumMasserstein` także używany przez trening, z zapisanymi parametrami modelu i `reduction='none'`. Pozostałe miary pochodzą z `reconstruction_metrics`: MSE, MAE, cosine similarity, spectral angle i błąd TIC.

Dla każdego widma przechowujemy błąd i charakterystyki wejścia: maksimum, liczbę niezerowych binów, liczbę dostępnych adnotacji. Dla każdego runu/podziału zachowujemy mean, median, q90 i q99. Średnia rekonstrukcji i rozkład seedów odpowiadają innym pytaniom niż ogon błędów pikselowych.

Per m/z zachowujemy średni błąd bezwzględny, średni błąd ze znakiem i średnie wejście. To lokalizacja błędu w **przestrzeni binned**, a nie test odzyskiwania surowych pików przed binningiem. Po normalizacji TIC oryginalna całkowita intensywność akwizycji nie jest dostępna z X i nie jest tu pozornie rekonstruowana.

Przykłady obejmują trzy z góry ustalone wspólne pozycje oraz trzy widma o największym Masserstein danego modelu. Zbiór pozycji jest zapisywany w tabeli wraz z flagą wyboru. Przykłady najgorsze diagnozują awarie; nie służą do oszacowania częstości ich występowania.

## Geometria

### Przestrzenie Z i U

Z jest rzeczywistym wyjściem encodera dla headu i dekodera. U=(Z−β)/γ usuwa wyuczoną transformację afiniczną końcowej LayerNorm, zgodnie z istniejącą implementacją `canonicalize`. Zakładamy architekturę CNNEncoder z końcową LayerNorm w `bottleneck_layer`; nie stosujemy tego wzoru do dowolnego latent space. Zerowe γ powoduje błąd zamiast dzielenia przez arbitralny epsilon.

Dla U suma współrzędnych powinna być bliska zeru, a promień bliski sqrt(D); epsilon LayerNorm może powodować odstępstwo od idealnego promienia. Kierunek stałej sumy powoduje strukturalne ograniczenie ranku, więc sam brak pełnych D wymiarów nie jest dowodem collapse. Z może mieć inną geometrię z powodu γ i β. Obie przestrzenie trzeba oglądać równolegle.

### Wariancja i lokalność

Covariance używa ddof=0. Dla wartości własnych λ_j≥0 i p_j=λ_j/Σλ obliczamy effective rank exp(−Σp log p), participation ratio (Σλ)²/Σλ² oraz trace. Idealny collapse ma trace=0 i otrzymuje osobną flagę; nie znika z zestawienia przez wyjątek. Niewielki trace i duże cosine similarity mogą ujawniać koncentrację mimo niezerowego formalnego ranku.

Wszystkie normy i wariancja używają pełnego analizowanego podziału. Geometria parowa używa wspólnej próbki S≤512, wybranej bez zwracania przez NumPy `default_rng(sample_seed)`. Zrealizowane pozycje są zapisane. Euklidesowe odległości i cosines obliczamy dla każdej nieuporządkowanej pary. Do CSV zapisujemy liczebności histogramów o 128 przedziałach, ich krawędzie oraz liczbę par skończonych i wszystkich. Histogram korzysta ze wszystkich par; dokładne wartości można odtworzyć z cache latentów, zamiast zapisywać dziesiątki milionów powtórzonych identyfikatorów. Dla różnych runów krawędzie mogą być różne, więc pokazujemy gęstość, a nie surowe count. Medianę liczymy z pełnych odległości przed agregacją; cosine z wektorem zerowym jest NaN, nie arbitralnie równy 0 lub 1.

kNN wyklucza samą próbkę i używa k=min(10,S−1). Przy remisach obowiązuje stabilna kolejność indeksów. Jaccard to |Y_i∩Y_j|/|Y_i∪Y_j| na zbiorach dostępnych dodatnich adnotacji; pustą unię kodujemy jako 0. To diagnostyka organizacji adnotacji zależna od ich częstości, bez modelu zerowego ani testu istotności. Dołączona projekcja SVD/PCA jest wyłącznie ilustracją; osie i znaki osobno dopasowanych projekcji nie są wspólnymi współrzędnymi między modelami.

CKA korzysta z istniejącego `linear_cka`, centrując każdą macierz po próbkach. Jest niezmiennicze wobec rotacji i globalnej skali, więc nie wykrywa każdej funkcjonalnie istotnej zmiany. Wynik dla stałej reprezentacji jest NaN. Porównania within-condition across-seed stanowią opisowy kontekst dla między-headowych różnic; pary CKA dzielą modele i nie są niezależnymi replikami.

### Wspólny liniowy probe

Do każdego zamrożonego Z dopasowujemy ridge regression adnotacji. Średnie i odchylenia współrzędnych obliczamy wyłącznie na train; stałe współrzędne mają scale=1. Rozwiązujemy:

\[
B=(\tilde Z^T\tilde Z/N+\alpha I)^{-1}\tilde Z^T(Y-\bar Y)/N,
\quad \hat Y=\tilde Z_{eval}B+\bar Y,
\]

z ustalonym α=0.01. Używamy `solve`, nie jawnego odwracania macierzy. Nie dopasowujemy α do testu ani do kolejnego headu. Score jest ciągły i nieograniczony, oceniany AP/AUC bez progu; nie jest prawdopodobieństwem. Probe traktuje niezaobserwowane adnotacje jako zero celu regresji i mierzy odczytywalność adnotacji, nie identyfikowalność ukrytych pozytywów.

Lepszy probe może wskazać bardziej dostępny liniowo sygnał. Różnica wobec nieliniowego headu nie izoluje jednej przyczyny: różnią się pojemność, cel i regularizacja. Nie wyciągamy wniosku o łatwości optymalizacji z samej końcowej geometrii.

## Decyzja i dalszy test

Główną miarę wyboru ustalamy przed oglądaniem testu: walidacyjne macro AP w annotation_retrieval. Operacyjne P/N, rzadkie klasy, luka generalizacji, rekonstrukcja i probe są diagnostykami. Front Pareto maksymalizuje tę samą AP i minimalizuje średni walidacyjny Masserstein. Warunki z niepełnym zestawem pięciu seedów lub niezgodnymi kontraktami nie są automatycznie kwalifikowane.

Waga headu jest stała liczbowo, ale różne losses mają inną skalę i gradienty. Wynik po 15 epokach może odzwierciedlać szybkość uczenia, nie asymptotyczną jakość. Historie pozwalają ocenić zbieżność i niestabilność; nie zawierają pełnych historycznych latentów ani Jacobianów. Nie dopisujemy nieobserwowanej dynamiki geometrii.

Zaproponowany kolejny etap to interwencje na negatywach w obrębie rodziny, a następnie skrzyżowanie wybranych headów z ustaloną kontraktywnością. Kontrastywność i pretraining pozostają osobnymi czynnikami. Generalizację należy potwierdzić na niezależnej akwizycji; bieżący podział pikselowy i pięć seedów tego nie zastępują.
## Class-stratified prediction (part 8)

Part 8 joins predictions to the existing part-0 class catalogue by `class_name`, with a one-to-one identity check. Positional joins are rejected. The historical `rare`, `common` and `undetectable` labels are retained exactly. They describe the part-0 population at its recorded evidence threshold (approximately 0.005309), while the trained campaign uses 0.0119 with radius 1. These populations and thresholds must not be silently substituted for one another.

The independent `train_regime` uses only training counts at the campaign evidence rule. For each class, let P, N and U be its numbers of available positive, negative and uncertain entries. Its prevalence is P/(P+N+U), and its negative fraction among unannotated entries is N/(N+U). Following the part-0 classification order, a negative fraction of at least 0.95 yields `undetectable`; otherwise prevalence below 0.01 yields `rare`, and the remaining classes are `common`. These names denote operational groups, not biological conclusions about detectability. Undefined fractions remain undefined, and a class without available entries is `unavailable`.

Three further descriptive partitions are retained: training-positive support (0, 1–9, 10–99, at least 100), historical evidence AUC (at most 0.5, above 0.5 through 0.7, above 0.7, undefined), and half-open 100-Da m/z windows. The support partition differs deliberately from the prevalence-based `rare` definition. Evidence-AUC boundaries and mass windows are descriptive diagnostic choices; they are not optimized against validation or test performance. Historical evidence AUC and regimes use the full part-0 population, so they are descriptive strata only. They do not enter head selection or model fitting.

Within each model, split, population and stratum, AP, ROC AUC and AP minus observed prevalence are unweighted means over eligible classes. A class is eligible only when it has training positives and both positives and negatives in the evaluated population. Tables retain total and eligible class counts, evaluated P/N counts, and every per-class observation. Empty eligible groups yield NaN rather than zero. Thus the very-low-support group can have different evaluable class counts across splits; direct train–held-out differences require the matched-class procedure of part 3.

Repeated tasks are averaged within condition and actual seed before comparisons. Baseline contrasts use the same seed and data/backbone/training contracts as the main analysis. The 95% t intervals over five paired seeds are exploratory, without multiplicity adjustment. Part 8 exports both historical-to-training regime transitions and all individual paired differences. No stratum is promoted to a separate confirmatory hypothesis after observing its result.
