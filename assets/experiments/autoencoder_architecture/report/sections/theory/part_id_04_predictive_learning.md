#### Problem z dużą ilością klas 
U nas mamy 508 klas, ich końcowo i tak będzie znacznie więcej. W wiekszości będą one na zerze (#TODO - wstawić później gdzie to w analizie się pojawia). 

Problem polega na tym, że interpretowane są one jako **negatywy**. Chciałbym wprowadzić interpretacje, że są one **nieoznaczone**, wtedy ustalamy rozkłady prawdopodobieństwa 

##### Ważenie negatywów podobieństwem Jaccarda

**Wstęp**

Pierwszym rozwiązaniem  zmodyfikowanie kontrastywności, poprzez wprowadzenie ważenia tych prawdopodobieństw za pomoca jaccarda.  

Standardowy InfoNCE traktuje każdy inny element batcha jako negatyw. W MSI dwa piksele pochodzące z tej samej struktury tkanki mają niemal identyczny zestaw anotacji, więc odpychanie ich od siebie jest fałszywym negatywem.

**Wyprowadzenie**

Obiekt, na którym liczę podobieństwo a wektory anotacji. Każdemu pikselowi $i$ odpowiada binarny wektor $y_i \in \{0,1\}^C$, $C = 508$, gdzie $y_{i,c} = 1$ oznacza, że molekuła $c$ została anotowana w tym pikselu przez METASPACE.

Podobieństwo dwóch pikseli definiuję jako współczynnik Jaccarda tych wektorów:

$$\mathrm{Jaccard}(y_i, y_j)
= \frac{\lvert \{c : y_{i,c} = 1 \wedge y_{j,c} = 1\} \rvert}
       {\lvert \{c : y_{i,c} = 1 \vee y_{j,c} = 1\} \rvert}
= \frac{y_i^\top y_j}{\lVert y_i \rVert_1 + \lVert y_j \rVert_1 - y_i^\top y_j} \in [0, 1]$$

jest liczba molekuł anotowanych w obu pikselach, podzielona przez liczbę molekuł anotowanych w co najmniej jednym z nich.

> Uwaga:
> Mianownik nigdy nie jest zerowy, ponieważ bierzemy tutaj pod uwagę peaki tylko z annotacją. Gwarantuje to, że współczynnik jest zatem dobrze określony dla każdej pary.


Następnie wykorzystując współczynnik Jaccarda defniuje wage jako:

$$w_{ij} = 1 - (1 - w_{\min}) \cdot \mathrm{Jaccard}(y_i, y_j), \qquad w_{\min} = 0.25$$

Poniżej zamieszczam przykłady liczbowe: 

| sytuacja | $\mathrm{Jaccard}$ | $w_{ij}$ |
|---|---|---|
| brak wspólnych anotacji | $0$ | $1.00$ — pełne odpychanie |
| połowa wspólnych | $0.5$ | $0.625$ |
| identyczne zestawy anotacji | $1$ | $0.25$ — odpychanie stłumione czterokrotnie |



Waga dodaje do mianownika InfoNCE, mnożąc składnik odpowiadający parze $(i, j)$:

$$L_{\mathrm{NCE}} = -\frac{1}{2B}\sum_{i}
\log \frac{\exp\big(\mathrm{sim}(g_i, g_{\pi(i)})/\tau\big)}
{\underbrace{\sum_{j \neq i,\; j \neq \pi(i)} w_{ij}\,\exp\big(\mathrm{sim}(g_i, g_j)/\tau\big)}_{\text{tu działa waga}}
+ \exp\big(\mathrm{sim}(g_i, g_{\pi(i)})/\tau\big)}$$

Para pozytywna nie jest ważona, mechanizm dotyczy wyłącznie siły odpychania negatywów.

Ponieważ $w_{ij} > 0$ dla każdej pary, żaden negatyw nie jest usuwany z mianownika.

#### Predykcja

Rzadkość etykiet wymusza wybór metryk odpornych na próg. Klasy o zerowej liczbie pozytywów w danym splicie nie niosą informacji, a `pos_weight` do $20$ przesuwa optymalny próg wyraźnie poniżej $0.5$, więc metryki progowe mierzą kalibrację, a nie zdolność dyskryminacyjną.

Przyjmuję zatem:

1. **Metryki bezprogowe jako główne**: average precision i AUROC. Metryki progowe raportuję
   pomocniczo, przy progu dobranym na walidacji, osobno dla każdej klasy.
2. **Baseline all-zero** jako punkt odniesienia dla hamming loss oraz prewalencję klasy jako
   punkt odniesienia dla average precision.
3. **Uśrednianie makro wyłącznie po klasach z co najmniej jednym pozytywem** w danym splicie,
   z jawnym podaniem tej liczby. Uśrednianie po wszystkich $C = 508$ klasach zaniża wynik
   proporcjonalnie do liczby klas nieobecnych i uniemożliwia porównanie splitów.
4. **Rozbicie wyników według liczebności klasy**, ponieważ obcięcie `max_positive_weight` do
   $20$ niedoważa klasy rzadkie, a ich jakość jest osobnym pytaniem niż jakość klas obfitych.

> Uwaga:
> Baseline all-zero jest trywialny, pokazuje jedynie, że model nie jest zdegenerowany.
> Mocniejszym punktem odniesienia jest regresja logistyczna na surowych $M$ binach, przy tym
> samym podziale i tej samej stracie. Odpowiada ona na pytanie, czy wąskie gardło o rozmiarze
> $L$ wnosi cokolwiek, czy tylko traci informację molekularną.

***
