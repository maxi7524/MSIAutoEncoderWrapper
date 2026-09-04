#### Zakres analizy i porównanie bazowe

Analiza izoluje wpływ kary kontraktywnej przez porównanie `grid_0000` (`bce`) z `grid_0001` (`+contractive`). Oba warianty używają tej samej architektury CNN, binningu $0.55$, danych, podziału danych, optymalizatora, liczby epok i harmonogramu pięciu powtórzeń. Dla tego samego numeru powtórzenia stosowany jest ten sam harmonogram ziaren inicjalizacji i treningu. Jedyną planowaną różnicą jest obecność składnika kontraktywnego, dlatego porównanie należy wykonywać parami o tym samym `rep_id`, a nie mieszać wszystkich modeli w jedną pulę.

W obu wariantach funkcja celu zawiera rekonstrukcję Massersteinem o wadze $1.0$ oraz zbalansowane wieloetykietowe BCE głowy predykcyjnej o wadze $0.2$. W `grid_0001` dodano karę kontraktywną o wadze $\lambda_{\mathrm{CAE}} = 0.001$. Wartości rekonstrukcji i predykcji pozostają obowiązkowymi wynikami kontrolnymi: mniejsza czułość enkodera nie jest poprawą, jeżeli została uzyskana kosztem rekonstrukcji albo jakości predykcji.

#### Kara użyta podczas treningu

Niech $x \in \mathbb{R}^M$ oznacza widmo po binningu i normalizacji TIC, a $z = \mathrm{enc}(x) \in \mathbb{R}^{10}$ oznacza wyjście enkodera zapisane jako `latent_space`. Dla pojedynczego widma kara ma postać $L_{\mathrm{CAE}}(x) = \lVert J_z(x) \rVert_F^2$, gdzie $J_z(x) = \partial z / \partial x$. W kroku treningowym raportowana jest średnia tej wielkości po widmach w batchu, a całkowita strata ma postać $L = L_{\mathrm{Masserstein}} + 0.2L_{\mathrm{BCE}} + 0.001L_{\mathrm{CAE}}$.

Jawne utworzenie Jakobianu o kształcie $(B,10,M)$ byłoby niepotrzebnie kosztowne. Implementacja wykorzystuje estymator Hutchinsona. Dla pięciu niezależnych wektorów Rademachera $v_k \in \{-1,+1\}^{10}$ liczone są iloczyny wektor-Jakobian $J_z(x)^\top v_k$ i średnia $\frac{1}{5}\sum_{k=1}^{5}\lVert J_z(x)^\top v_k\rVert_2^2$. Ponieważ $\mathbb{E}[v_kv_k^\top]=I$, estymator jest nieobciążony względem $\lVert J_z(x)\rVert_F^2$. W praktyce oznacza to pięć dodatkowych przebiegów wstecz na batch, bez materializowania pełnego Jakobianu.

Kara jest lokalna: jest liczona w punktach rzeczywiście obecnych w batchu i nie interpoluje między różnymi widmami. Minimalizuje średnią czułość na małe zaburzenia we wszystkich kierunkach wejścia, a nie gwarantuje odporności na najgorszy kierunek ani na konkretną fizyczną transformację widma.

#### Wpływ LayerNorm i poprawna przestrzeń oceny

Końcowa `LayerNorm` enkodera ma uczone parametry $\gamma$ i $\beta$. Do analizy geometrii kod jest kanonizowany jako $u = (z - \beta)/\gamma$. Tylko $u$ opisuje położenie na sferze wynikającej z normalizacji, niezależnie od afinicznej skali modelu. Surowe $z$ nie może być podstawową miarą kontrakcji, ponieważ model może zmniejszać normę $\gamma$ i kompensować tę zmianę w pierwszych warstwach dekodera oraz głowy, bez równoważnej zmiany geometrii $u$.

Z tego powodu $\lVert J_z\rVert_F^2$ jest prawidłowym składnikiem aktualnie trenowanej funkcji celu, ale nie jest samodzielną miarą powodzenia eksperymentu. W raporcie podstawową miarą efektu jest odpowiedź kanonizowanego kodu $u$ na perturbację wejścia. Normy $\gamma$, normy pierwszych warstw dekodera i głowy oraz rozkład odchylenia $\sigma(a(x))$ przed końcową `LayerNorm` są diagnostyką możliwych dróg obniżania kary, a nie dowodem odporności.

#### Pomiar czułości enkodera

Z każdego splitu, `train`, `validation` i `test`, losowana jest bez zwracania deterministyczna próba do 1500 widm z ziarnem `42`. Dla każdego z pięciu powtórzeń obu wariantów i każdego widma losowany jest kierunek $\delta$ o normie jeden. Tworzone jest zaburzone wejście $x_{\varepsilon} = x + \varepsilon\lVert x\rVert_2\delta$ dla $\varepsilon \in \{0, 0.01, 0.03, 0.1, 0.3, 1.0\}$. Po zakodowaniu $x$ i $x_{\varepsilon}$ oraz kanonizacji ich kodów liczony jest kąt $\theta_{\varepsilon}(x) = \arccos(u(x)^\top u(x_\varepsilon)/10)$ w stopniach.

Najpierw średnia $\theta_\varepsilon$ jest liczona po widmach wewnątrz jednego modelu. Otrzymujemy więc jedną krzywą na parę `(wariant, rep_id, split)`. Dopiero potem zestawiane są pięć krzywych powtórzeń: średnia opisuje typowy efekt, a rozrzut między powtórzeniami opisuje jego stabilność treningową. Mały kąt oznacza mniejszą zmianę kierunku reprezentacji po relatywnie tej samej perturbacji wejścia. Wynik przy $\varepsilon=0$ jest kontrolą numeryczną i powinien wynosić zero.

Wykres względnej czułości dzieli średni kąt perturbacyjny przez odchylenie standardowe kątów pomiędzy do 20 000 parami nieidentycznych, nieperturbowanych widm z tego samego modelu i splitu. Wartość mniejsza od jeden oznacza, że perturbacja przesuwa kod słabiej niż naturalne zróżnicowanie widm w danej reprezentacji. Ta normalizacja jest konieczna, ponieważ mały kąt bezwzględny może wynikać zarówno z lokalnego wygładzenia, jak i z globalnego zapadnięcia chmury kodów.

#### Kontrole przeciw pozornej kontrakcji

`cloud_asymmetry = \lVert\bar{u}\rVert_2^2/10`, gdzie $\bar{u}$ jest średnim kodem kanonizowanym w danym modelu i podziale danych, mierzy globalne skupienie chmury. Wartość bliska zero oznacza rozłożenie kodów po sferze, a wartość bliska jeden oznacza, że różne widma otrzymują niemal ten sam kod. Ta miara musi być czytana razem z krzywą czułości: spadek kąta perturbacyjnego przy jednoczesnym wzroście `cloud_asymmetry` nie świadczy sam w sobie o użytecznej stabilności.

Rozkład $\sigma(a(x))$ przed `LayerNorm` sprawdza, czy kara jest częściowo realizowana przez zwiększanie kontrastu przed normalizacją. Normy $\gamma$, pierwszej warstwy dekodera i pierwszej warstwy głowy sprawdzają, czy model zmniejsza skalę $z$ i kompensuje ją w modułach następujących. Dodatkowa analiza ścieżki perturbacji dla 50 widm testowych przy $\varepsilon=1.0$ porównuje kolejno normy $\delta a$, $P\delta a$, $\delta u$, $\delta z$ oraz zmianę wyjścia pierwszej warstwy dekodera, każdą podzieloną przez $\lVert\delta x\rVert_2$. Pozwala to ustalić, w którym miejscu potoku pojawia się obserwowane wygładzenie.

#### Kryterium wniosku

Wniosek o korzystnym działaniu kontraktywności wymaga łącznie: stabilnego spadku czułości kątowej po pięciu powtórzeniach, braku zapadnięcia chmury według `cloud_asymmetry`, zachowania rekonstrukcji oraz braku istotnego pogorszenia metryk predykcji. Wyniki na treningu, walidacji i teście należy raportować osobno. Test jest miarą generalizacji, a trening pomaga rozpoznać, czy efekt został wyuczony wyłącznie lokalnie dla danych widzianych.

#### Rozszerzenia do kolejnych ablacji

##### Przeszukanie wagi $\lambda_{\mathrm{CAE}}$

Aktualna implementacja pozwala zmienić wagę `regularization.contractive.weight` bez zmian w kodzie. Minimalny kolejny eksperyment powinien porównać $\lambda_{\mathrm{CAE}} \in \{0, 10^{-5}, 10^{-4}, 10^{-3}\}$ przy pięciu powtórzeniach na wariant. Dla każdej wartości należy raportować pełny zestaw kontroli z tej metodologii, a nie wybierać konfiguracji wyłącznie po najmniejszej czułości. Jest to najtańszy test, czy obserwowane zapadnięcie reprezentacji wynika z nadmiernej wagi obecnej kary.

##### Kara w przestrzeni kanonizowanej

Obecna kara działa na $z$ i pozostawia drogę kompensacji przez $\gamma$. Następna implementacja może karać $\lVert\partial u/\partial x\rVert_F^2$, gdzie $u=(z-\beta)/\gamma$, albo unieruchomić afiniczne parametry końcowej `LayerNorm`. Pierwszy wariant jest bardziej bezpośredni, lecz wymaga rozszerzenia `ContractiveLoss`, aby różniczkował kanonizację wewnątrz grafu autograd. Drugi wymaga zmiany architektury, ponieważ obecny enkoder tworzy `LayerNorm` z uczonymi parametrami. W obu przypadkach obowiązkowe są testy zgodności z obecną kanonizacją i ponowne porównanie jakości rekonstrukcji.

##### Najgorszy kierunek zamiast średniej po kierunkach

W poprzedniej wersji raportu zaproponowano zastąpienie normy Frobeniusa normą spektralną Jakobianu. Taka kara minimalizowałaby $\lVert J_u(x)\rVert_2^2$, czyli wrażliwość na najgorszy lokalny kierunek, zamiast średniej po wszystkich kierunkach. Obecna implementacja nie obsługuje tej strategii: potrzebna jest estymacja metodą iteracji potęgowej, wykorzystująca naprzemiennie JVP i VJP, oraz test porównujący wynik na małym przykładzie z jawnym Jakobianem. To osobna ablacja, a nie równoważna zamiana aktualnego estymatora Hutchinsona.

##### Kontrakcja zdefiniowana przez fizyczną perturbację

Drugi odnaleziony pomysł to kara niewrażliwa na zmianę intensywności, a skupiona na przesunięciach po osi $m/z$. Powinien być sformułowany jako kierunkowa kara na konkretnych transformacjach, na przykład $\mathbb{E}_{x,\Delta}[\angle(u(x),u(T_\Delta x))^2]$, gdzie $T_\Delta$ symuluje małe przesunięcie kalibracyjne. Można analogicznie zdefiniować transformację skalującą intensywności. Przed wdrożeniem trzeba jawnie zdecydować, które przesunięcia są artefaktem pomiarowym, a które niosą informację chemiczną. Ten wariant wymaga nowego kryterium lub kontrolowanej augmentacji treningowej; nie jest realizowany przez obecną karę Jakobianową.

##### Ograniczenie zapadnięcia reprezentacji

Jeżeli skan wagi nadal prowadzi do wysokiego `cloud_asymmetry`, można dodać niezależny człon zachowujący rozproszenie kodów, na przykład karę za zbyt mały ślad $\operatorname{Cov}(u)$ lub za zbyt dużą normę $\bar{u}$. Taki człon musi być traktowany jako nowa hipoteza eksperymentalna, a nie jako poprawka wizualna. Jego konfiguracja powinna zawierać osobną wagę, wartość docelową oraz ablację bez tego członu.

#### Artefakty

- [Konfiguracja kampanii predykcyjnej](../../../../experiment_runs_configs/23_08_26_architecture_predictive/architecture_predictive_experiment.yaml)
- [Notebook weryfikacji kontraktywności](../../../../notebooks/23_08_26_architecture_predictive/part_4_contractive_verification.ipynb)
- [Implementacja aktualnej kary kontraktywnej](../../../../../../../src/msi_autoencoder_wrapper/training/criterions/autoencoder/regularization/contractive_loss.py)
- [Implementacja metryk geometrii](../../../../../../../src/msi_autoencoder_wrapper/analysis/autoencoder/latent/sphere_geometry.py)
- [Część teoretyczna o kontraktywności](../../theory/part_id_05_contractive.md)
