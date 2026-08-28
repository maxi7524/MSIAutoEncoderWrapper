### Notacja

| symbol | znaczenie |
|---|---|
| $x \in \mathbb{R}^{M}$ | widmo wejściowe, $M = 1273$ |
| $a \in \mathbb{R}^{L}$ | wyjście `Linear(368, L)`, przed normalizacją |
| $u \in \mathbb{R}^{L}$ | wyjście LayerNormu przed transformacją afiniczną |
| $z = \gamma \odot u + \beta$ | reprezentacja w przestrzeni ukrytej (wyjście enkodera) |
| $\mathbf{1}$ | wektor jedynek w $\mathbb{R}^L$ |
| $A_x = \partial a / \partial x$ | Jakobian części enkodera przed normalizacją |
| $J(x) = \partial z / \partial x$ | Jakobian całego enkodera |
| $g(z) \in \mathbb{R}^{64}$ | wyjście projektora, używane wyłącznie przez stratę kontrastywną |
| $P = I - \tfrac{1}{L}\mathbf{1}\mathbf{1}^\top - \tfrac{1}{L}uu^\top$ | rzut na przestrzeń styczną do $M$ |

Dodatkowe założenia: 
- $10^{-8}=\varepsilon \ll \sigma^2$ (będziemy to później pomijać).

***

### Motywacje teoretyczne - trening

#### Charakteryzacja przestrzeni
Ostatecznie będziemy pracować na elipsoidzie ($L-2$)-wymiarowej, zanurzonej w ($L-1$) wymiarowej podprzestrzeni afinicznej.  

$$\boxed{\;M \;\cong\; S^{L-2}\big(\sqrt{L}\big) \subset \mathbb{R}^{L-1}\;}$$

Poniżej wytłumaczenie.

> Uwaga:
> Jest to ważny fakt, który wymusza na nas późniejsze wprowadzenie poprawek w uczeniem kontraktywnym. 

##### Przekształcenia wynikające z architektury 
W architekturze po $\mathrm{CNN}$, jest ustawiona warstwa liniowa `Linear(368, L)`, którą następnie:
1. normalizujemy po warstwie (`LayerNorm`) 
$$\mu = \frac{1}{L}\sum_{i=1}^{L}a_i, \qquad \sigma^2 = \frac{1}{L}\sum_{i=1}^{L}(a_i - \mu)^2, \qquad u_i = \frac{a_i - \mu}{\sigma}$$
2. przenosimy poprzez odwzorowanie liniowe 
$$z = \gamma \odot u + \beta$$

##### Podprzestrzeń afiniczna (normalizacja po warstwie)
Z 1. otrzymujemy, że $u$ znajduje się na hiperpłaszczyźnie  
$$\mathbf{1}^T \cdot u = \underbrace{\sum_{i=1}^L u_i}_{\text{Uwaga 1.1}} = \frac{\sum_{i=1}^L (a_i - \mu)}{\sqrt{\sigma^2 + \varepsilon}} = \frac{\sum_{i=1}^L (a_i) - \overbrace{L \cdot \mu}^{=\sum_{i=1}^L a_i}}{\sqrt{\sigma^2 + \varepsilon}} = 0$$

> Uwaga 1.1:
> Zauważmy że jest to przestrzeń afiniczna, można to potwierdzić poprzez PCA które powinno miec wymiar 9.

Sprawdzając norme $u$: 
$$\lVert u \lVert^2 = \sum_{i=1}^D (u_i)^2 =  \frac{\overbrace{ \sum_{i=1}^D(a_i - \mu)^2}^{= D \sigma^2 }}{\sigma^2 + \varepsilon} = \frac{D \sigma^2}{\sigma^2 + \varepsilon} \xrightarrow{\varepsilon -> 0} \frac{D \sigma^2}{\sigma^2} = D$$
Otrzymujemy ostatecznie, że nasza podrozmaitość jest postaci

$$M = \{\,u \in \mathbb{R}^L \;:\; \mathbf{1}^\top u = 0,\;\; \|u\|^2 = L \,\}$$

co jest definicją sfery $L-2$ wymiarowej zanurzonej w przestrzeni afinicznej ${L-1}$-wymiarowej:

$$\boxed{\;M \;\cong\; S^{L-2}\big(\sqrt{D}\big) \subset \mathbb{R}^{D-1}\;}$$

<!-- ##### Wymiar $M$  

Ja wiem ze to jest intuicyjnie oczywiste, ale nigdy się nie zastanawiałem nad tym jak to się wykazuje 

... -> można te tweirdzenia rozpisac z czego to wynika 

-->

##### Przekształcenie na elipsoide
Nastepnie przenosimy elementy $u \in S^{L-2}$,  przez odwzorowanie afiniczne i otrzymujemy punkt w przestrzeni ukrytej
$$z = \gamma \odot u + \beta$$ 
Dla $\gamma_i \not = 0$ jest to afiniczna bijekcja. 


***

### Motywacje teoretyczne - analiza

Omawiam tutaj metryki oraz sposób porównywania modeli w różnych analizach, które uwzględniają podstawy teoretyczne. 

#### Przestrzeń ukryta

##### Obiekt, na którym liczę miary

**Wszystkie miary geometryczne liczę na $u$, nie na $z$.** Robię to odwracając transformację afiniczną:

$$\boxed{\;u = \frac{z - \beta}{\gamma}\;}$$

gdzie $\gamma, \beta$ odczytuję z `state_dict` warstwy `LayerNorm` . Dla $\gamma_i \neq 0$ jest to odwzorowanie odwrotne do $T$ , więc odzyskuję dokładnie ten wektor, który zwróciła normalizacja.

Robie to ponieważ:
1. $T$ jest homeomorfizmem, nie zmienia własności przestrzeni. Zmienia natomiast obserwowane
   odległości, kąty, normy oraz widmo kowariancji, a więc rangę efektywną i współczynnik
   partycypacji.
2. Bez kanonizacji porównywałbym dodatkowo różnicę wynikającą z odzorowania afinicznego. $\gamma$ różni się systematycznie między ablacjami, ponieważ otrzymuje gradienty z członów różniących się pomiędzy kombinacjiami.

Po kanonizacji wszystkie modele leżą na tej samej podrozmaitości $S^{L-2}(\sqrt{L})$ i możemy je wiarygodnie porównać.

> Uwaga:
>
> Kanonizacja jest konieczna niezależnie od wartości $\gamma$, natomiast skalę zniekształcenia,
> które usuwa, mierzę współczynnikiem uwarunkowania $\kappa = \gamma_{\max}/\gamma_{\min}$.
>
> Wynika to z tego, że $z - z' = \gamma \odot (u - u')$, więc zachodzi
> $\gamma_{\min}\lVert u - u'\rVert \le \lVert z - z'\rVert \le \gamma_{\max}\lVert u - u'\rVert$.
> Stąd porządek dwóch punktów może się odwrócić tylko wtedy, gdy iloraz ich odległości mieści
> się w $[1/\kappa,\, \kappa]$. Dla $\kappa \approx 1$ transformacja jest jednokładnością i nie
> zmienia niczego liczbowo, dla $\kappa \gg 1$ przetasowuje sąsiedztwa.
>
> Wartość $\kappa$ raportuję z dwóch powodów. Po pierwsze pozwala ocenić, czy wielkości
> policzone bez kanonizacji wymagają przeliczenia. Po drugie samo $\lVert\gamma\rVert$ jest
> wynikiem, ponieważ kara kontraktywna spycha $\gamma$ w dół, więc porównanie tej wartości
> między ablacjami jest testem degeneracji skali.

##### Metryka

Raportuję kąt:

$$\cos\theta = \frac{\langle u, u' \rangle}{L}, \qquad \theta \in [0, \pi]$$

Ponieważ wszystkie punkty mają jednakową normę, zachodzi tożsamość

$$\lVert u - u' \rVert^2 = 2L\,(1 - \cos\theta) = 4L\sin^2(\theta/2)$$

Metryka cięciwowa $2\sqrt{L}\sin(\theta/2)$, geodezyjna $\sqrt{L}\,\theta$ i kosinusowa $1 - \cos\theta$ są zatem ściśle rosnącymi funkcjami $\theta$, a więc przekształcają się jedna w drugą przez rosnącą bijekcję.

> Uwaga:
> Nie jest to konsekwencja równoważności norm na $\mathbb{R}^L$. Równoważność norm gwarantuje
> jedynie zgodność topologii i nie mówi nic o uporządkowaniu sąsiadów. Powyższa tożsamość jest
> algebraiczna i obowiązuje wyłącznie przy równości obu norm, czyli na $u$, a nie na $z$.

Wybieram $\theta$, ponieważ jest ograniczony i czytelny w stopniach, jest metryką wewnętrzną rozmaitości, oraz nie zawiera czynnika $\sqrt{L}$, przez co pozostaje porównywalny między modelami o różnym rozmiarze przestrzeni ukrytej.

Konsekwencje są:
- **Niezmiennicze na wybór metryki** (zależą tylko od porządku odległości): kNN-overlap,
  kNN-purity, RSA ze Spearmanem, trustworthiness, continuity.
- **Zależne od wartości** (metrykę trzeba zadeklarować): średnie odległości, silhouette,
  RSA z Pearsonem, CKA z jądrem RBF, uniformity, $k$-means, MDS.

##### Punkt odniesienia: brak struktury

Dla dwóch niezależnych punktów jednostajnych na $S^{d}$ zachodzi $\mathbb{E}[\cos\theta] = 0$ oraz $\operatorname{Var}[\cos\theta] = 1/(d+1)$. W kampanii $d = L - 2 = 8$, zatem

$$\operatorname{sd}[\cos\theta] = \tfrac{1}{3}, \qquad \theta \approx 90° \pm 19.5°$$

Empiryczny rozkład o średniej bliskiej zeru i odchyleniu bliskim $0.33$ jest nieodróżnialny od jednostajnego, czyli oznacza brak struktury. Odchylenie od tej wartości jest pierwszą liczbą raportowaną w tej analizie.

Analogiczny baseline stosuję dla kNN-overlap: dla dwóch niezależnych zbiorów $k$ sąsiadów z $N$ punktów oczekiwane pokrycie wynosi $k/N$.

##### Trzy poziomy porównania

Pytanie "czy przestrzeń ukryta się zmieniła" rozbijam na trzy niezależne analizy.

| poziom | pytanie | narzędzie | niezmienniczość |
|---|---|---|---|
| geometria globalna | czy struktura jest ta sama z dokładnością do transformacji | odległość Procrustesa, CKA liniowe | obrót, odbicie, skalowanie izotropowe |
| sąsiedztwa lokalne | czy te same piksele są blisko siebie | kNN-overlap, trustworthiness, continuity | dowolne przekształcenie zachowujące porządek |
| zawartość informacyjna | czy da się odczytać to samo | probing, RSA względem etykiet | dowolna bijekcja |

Uzupełniająco raportuję charakterystyki rozkładu na sferze: widmo wartości własnych, rangę efektywną, współczynnik partycypacji, wymiar wewnętrzny (TwoNN) oraz asymetrię chmury $\lVert \bar{u} \rVert^2 / L$.

> Uwaga:
> Ograniczenie $\operatorname{tr}\operatorname{Cov}(u) = L - \lVert \bar{u} \rVert^2$ oznacza
> stały budżet wariancji. Widmo wartości własnych jest więc czystą alokacją ustalonej puli,
> co czyni rangę efektywną i współczynnik partycypacji bezpośrednio porównywalnymi między
> ablacjami, bez dodatkowej normalizacji.
>
> Rangę liniową i wymiar wewnętrzny raportuję osobno, ponieważ mierzą różne rzeczy. Warunek
> $\mathbf{1}^\top u = 0$ jest liniowy, więc widzi go PCA i daje rangę $L-1$. Warunek na normę
> jest nieliniowy, więc PCA go nie wykrywa, a ujawnia się dopiero w estymacji wymiaru
> wewnętrznego, gdzie sufitem jest $L-2$.

***
