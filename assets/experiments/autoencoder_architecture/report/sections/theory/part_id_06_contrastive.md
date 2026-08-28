#### Kontrastywność 

Żeby zapewnić że różne widma z tymi samymi anotacjami, będą w podobnym miejscu w przestrzeni stosuje kare kontrastywną. 

Stosuję symetryczny InfoNCE (NT-Xent) w wariancie z permutację obwiedni pików. Dla batcha o rozmiarze $B$ buduję dla każdego widma *podobne widmo* $\tilde{x}$, a strata liczona jest na wyjściu projektora:

$$L_{\mathrm{NCE}} = -\frac{1}{2B}\sum_{i=1}^{2B}
\log \frac{\exp\big(\mathrm{sim}(g_i, g_{\pi(i)})/\tau\big)}
{\sum_{j \neq i} w_{ij}\,\exp\big(\mathrm{sim}(g_i, g_j)/\tau\big)}$$

gdzie $\mathrm{sim}$ to podobieństwo cosinusowe znormalizowanych projekcji, $\pi(i)$ to indeks pary pozytywnej, a $w_{ij}$ to waga negatywu (domyślnie $1$).

##### Konstrukcja pary pozytywnej

Żeby sprawdzić czy niezmienniczość klas wpływa na model porównuję dwa warianty doboru pików podlegających permutacji:

- `permutation_random` - permutacja bez względu na anotacje. Jest to kontrola do sprawdzenia czy niezmienniczość wpływa na wyniki. 
- `permutation_label_invariant` -  permutowane są tylko piki nieanotowane.


##### Implementacja architektury - strata nie działa bezpośrednio na przestrzeni ukrytej

Bazując na konstrukcji z SimCLR strata liczona jest na $g(z)$, a nie na $z$. Projektor `Linear(L, L) → LayerNorm → ReLU → Linear(L, 64)` oddziela geometrię wymuszaną przez InfoNCE od przestrzeni używanej przez rekonstrukcję i głowę.

Zastosowana funkcja **wpływa pośrednio na $z$**. Możemy się zatem spodziewać słabego efektu geometrycznego na przestrzeni ukrytej przy jednoczesnym silnym efekcie na $g(z)$. 

> Uwaga:
> Jest to istotne, ponieważ stosujac to **bezpośrednio na $z$** konstruowalibyśmy przestrzeń **niezmienniczą**, ze względu na permutacje widm. Rekonstrukcja była by wtedy możliwa (w teorii) tylko dla annotowanych widm. 

Weryfikacja wymaga policzenia tych samych miar geometrycznych osobno na $z$ i na $g(z)$.

> Uwaga:
> Projektor zawiera `ReLU` bezpośrednio po `LayerNorm`, czyli działa na wektorze o zerowej
> sumie. Przy $\beta \approx 0$ oznacza to, że dla każdej próbki zerowana jest istotna część
> współrzędnych. Warto zweryfikować empirycznie frakcję zer na wyjściu tej warstwy.

##### Przewidywany wpływ na geometrię

InfoNCE rozkłada się na dwa przeciwstawne człony (Wang i Isola, 2020): przyciąganie par pozytywnych oraz odpychanie wszystkich pozostałych. Prowadzi to do dwóch testowalnych przewidywań:

1. **Niezmienniczość.** Kąt między $u(x)$ a $u(\tilde{x})$ dla pary pozytywnej powinien być
   mniejszy w modelach kontrastywnych. Dla wariantu `label_invariant` przewiduję dodatkowo
   asymetrię: przesunięcie przy permutacji pików anotowanych powinno być wyraźnie większe niż
   przy permutacji nieanotowanych.
2. **Ryzyko zapadania wymiarów.** Człon odpychający jest znanym źródłem *dimensional collapse*
   — koncentracji wariancji w niewielkiej liczbie kierunków. Monitoruję to przez widmo
   wartości własnych, rangę efektywną i współczynnik partycypacji.

***

#### Kontrastywność

Strata liczona jest na $g(z)$, a nie na $z$, więc jej wpływ na przestrzeń ukrytą jest pośredni i przenoszony wyłącznie przez gradienty płynące wstecz przez projektor. Wszystkie miary geometryczne liczę zatem osobno na $u$ i na $g(z)$. Słaby efekt na $u$ przy silnym na $g(z)$ jest wynikiem zgodnym z architekturą, a nie oznaką, że strata nie działa.

Testem właściwym dla hipotezy z konstrukcji pary pozytywnej jest **wskaźnik selektywności**:

$$\frac{\mathbb{E}\,\angle\big(u(x),\, u(\tilde{x}_{\text{permutacja anotowanych}})\big)}
{\mathbb{E}\,\angle\big(u(x),\, u(\tilde{x}_{\text{permutacja nieanotowanych}})\big)}$$

Mierzy on wprost to, o co pytam, czyli czy enkoder polega na pikach anotowanych. Miary typu Procrustes i CKA odpowiadają jedynie na pytanie, czy reprezentacja jest inna, co jest warunkiem koniecznym, ale nie wystarczającym.

Uzupełniająco raportuję alignment i uniformity, ponieważ InfoNCE rozkłada się dokładnie na te dwa człony, oraz rangę efektywną, ponieważ człon odpychający jest znanym źródłem zapadania wymiarów.

Dla ważenia Jaccardem testem właściwym jest RSA między odległością kątową a odległością Jaccarda, ponieważ strata dosłownie wymusza monotoniczny związek między tymi wielkościami.

> Uwaga:
> Raportuję frakcję par o $w_{ij} < 1$. Wektory anotacji są rzadkie, więc znaczna część par
> ma $\mathrm{Jaccard} = 0$ i nie podlega ważeniu. Jest to warunek konieczny, żeby mechanizm
> mógł się w ogóle ujawnić.

***
