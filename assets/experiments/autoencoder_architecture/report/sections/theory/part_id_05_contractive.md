#### Kara za kontraktywność

Chciałem żeby encoder był stabilny ze względu na output, to jest żeby małe zaburzenia widma $x$, nie zmieniały znacząco położenia w latencie. 

Żeby to spełnić zastosowałem karę Rifai i in. (ICML 2011) w postaci kwadratu normy Frobeniusa Jakobianu enkodera:

$$L_{\mathrm{contractive}}(x) = \left\lVert \frac{\partial\, \mathrm{enc}(x)}{\partial x} \right\rVert_F^2
= \sum_{i=1}^{L}\sum_{m=1}^{M}\left(\frac{\partial z_i}{\partial x_m}\right)^2$$

Ma ona sprawić, aby encoder spełnial własność *kontrakcji* 

> Uwaga:
>  Norma Frobeniusa jest **średnią wrażliwością po wszystkich kierunkach zaburzenia**. Myślałem, że można by zmodyfikować tą funkcje zastępując norme Frobeniusa, normą spektralną, wtedy minimalizujemy błąd w najgorszym kierunku. 

Kara działa przeciwnie do rekonstrukcji. Rekonstrukcja wymaga, by $z$ zachowywało informację o $x$, czyli żeby enkoder rozróżniał widma. Kontraktywność wymaga, by ich nie rozróżniał. Równowaga ustala się tak, że enkoder pozostaje czuły w kierunkach, w których dane faktycznie się zmieniają, a staje się płaski w pozostałych — czyli wzdłuż rozmaitości danych, a nie w poprzek.

> Uwaga:
> Wymaga to, by enkoder był funkcją pojedynczej próbki. Wszystkie normalizacje w architekturze muszą byc  `LayerNorm`, nie `BatchNorm`, żeby $\partial \mathrm{enc}(x)/\partial x$ **nie zależy** od pozostałych elementów batcha. Wtedy Jakobian jest dobrze określony.

##### Implementacja 

Jawny Jakobian ma rozmiar $(B, L, M)$, czyli przy $B = 64$, $L = 10$, $M = 1273$ jest to zbyt kosztowne. Stosuję estymator Hutchinsona. Dla losowego $v$ o niezależnych współrzędnych $\pm 1$ zachodzi $\mathbb{E}[vv^\top] = I$, a stąd:

$$\mathbb{E}_{v}\big[\lVert J^\top v\rVert_2^2\big]
= \mathbb{E}_{v}\big[v^\top J J^\top v\big]
= \operatorname{tr}\big(J J^\top \mathbb{E}[vv^\top]\big)
= \operatorname{tr}(J J^\top)
= \lVert J\rVert_F^2$$

Każdy człon $J^\top v$ to jeden iloczyn wektor-Jakobian, czyli jeden przebieg wstecz. Używam $5$ prób, więc koszt to $5$ przebiegów zamiast $L = 10$ przy jawnym rachunku.

##### Wpływ normalizacji na interpretację kary

Kara zaproponowana w artykule Rifai i in. działa na enkoderze bez normalizacji, czyli na całej przestrzeni $\mathbb{R}^L$. U nas wyjście enkodera leży na sferze, więc chcę sprawdzić, co ta kara regularyzuje w naszym przypadku.

**Jakobian normalizacji**

Różniczkujac Jacobian, korzystamy z reguły łańcuchowej i otrzymujemy:
$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial a}\cdot \frac{\partial a}{\partial x}$$
Gdzie, trywialne do policzenia są:
- $\frac{\partial a}{\partial x}$ - to jest pochodna po $\mathrm{CNN+Linear}$
- $\frac{\partial z}{\partial u}$ - to jest $\gamma$ 

Skupiamy się zatem na wyliczeniu $\frac{\partial u}{\partial a}$, gdzie $u_i = (a_i - \mu)/\sigma$, pamiętając, że **$\mu$ i $\sigma$ też zależą od $a_j$ oraz pomijajać $\varepsilon$**.

Stosuję regułę ilorazu do $u_i = (a_i - \mu)/\sigma$:

$$\frac{\partial u_i}{\partial a_j}
= \frac{\overbrace{\left(\delta_{ij} - \frac{\partial \mu}{\partial a_j}\right)}^{\text{(1)}}\sigma
\;-\; (a_i - \mu)\,\overbrace{\frac{\partial \sigma}{\partial a_j}}^{\text{(2)}}}{\sigma^2}$$

gdzie $\delta_{ij}$ to delta Kroneckera.

> (1) Pochodna średniej. Z $\mu = \tfrac{1}{L}\sum_k a_k$ tylko jeden składnik zawiera $a_j$:
> $$\frac{\partial \mu}{\partial a_j} = \frac{1}{L}$$

> (2) Pochodna odchylenia. Różniczkuję stronami $\sigma^2 = \tfrac{1}{L}\sum_k (a_k - \mu)^2$,
> korzystając z (1):
> $$2\sigma\,\frac{\partial \sigma}{\partial a_j}
> = \frac{2}{L}\sum_{k=1}^{L}(a_k - \mu)\left(\delta_{kj} - \frac{1}{L}\right)
> = \frac{2}{L}\Big[(a_j - \mu) - \frac{1}{L}\underbrace{\sum_{k=1}^{L}(a_k - \mu)}_{=\,0}\Big]
> = \frac{2}{L}(a_j - \mu)$$
> Suma znika, ponieważ odchylenia od średniej sumują się do zera. Po podzieleniu przez $2\sigma$:
> $$\frac{\partial \sigma}{\partial a_j} = \frac{a_j - \mu}{L\sigma}$$

Podstawiam (1) i (2) do wyjściowego wyrażenia:

$$\frac{\partial u_i}{\partial a_j}
= \frac{\left(\delta_{ij} - \frac{1}{L}\right)\sigma - (a_i - \mu)\cdot\frac{a_j - \mu}{L\sigma}}{\sigma^2}
= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{L}
- \frac{1}{L}\cdot\frac{(a_i - \mu)}{\sigma}\cdot\frac{(a_j - \mu)}{\sigma}\right)$$

W dwóch ostatnich ułamkach wstawiamy z definicji $u$, czyli $u_i = (a_i - \mu)/\sigma$ oraz $u_j = (a_j - \mu)/\sigma$. Ostatecznie:

$$\boxed{\;\frac{\partial u_i}{\partial a_j} = \frac{1}{\sigma}\,P_{ij},
\qquad P_{ij} = \delta_{ij} - \frac{1}{L} - \frac{u_i u_j}{L}\;}$$

Trzy człony odpowiadają trzem źródłom zależności: 
- $\delta_{ij}$ to wpływ bezpośredni,
- $-1/L$ pochodzi od średniej, 
- $-u_iu_j/L$ od odchylenia.

**Działanie $P$ — interpretacja**

Zadziałajmy $P$ na dowolny wektor $v$:

$$(Pv)_i = v_i - \underbrace{\frac{1}{L}\sum_{k} v_k}_{\text{średnia } \bar v}
- \; u_i \cdot \underbrace{\frac{\langle u, v\rangle}{L}}_{\text{składowa wzdłuż } u}$$

Czyli $P$ odejmuje od $v$ dwie rzeczy: jego średnią oraz jego rzut na kierunek $u$.

Sprawdzam, co $P$ kasuje. Dla $v = \mathbf{1}$:

$$(P\mathbf{1})_i = 1 - 1 - \frac{u_i}{L}\underbrace{\textstyle\sum_k u_k}_{=\,0} = 0$$

Dla $v = u$:

$$(Pu)_i = u_i - \underbrace{\bar u}_{=\,0} - \frac{u_i}{L}\underbrace{\lVert u\rVert^2}_{=\,L} = u_i - u_i = 0$$

Natomiast dla $v$ prostopadłego do obu ($\sum_k v_k = 0$ oraz $\langle u, v\rangle = 0$) otrzymuję $Pv = v$.

$P$ jest zatem odwzorowaniem, które zeruje kierunki $\mathbf{1}$ i $u$, a pozostałe zostawia bez zmian — czyli rzutem ortogonalnym na $\{\mathbf{1}, u\}^\perp$, o rzędzie $L - 2$.

> Uwaga:
> Powyższy wniosek jest oczywisty z tego powodu, że jestesmy na sferze, służy bardziej do weryfikacji rachunków 

Kara mierzy zatem wyłącznie ruch **po powierzchni sfery**, to jest obrót kierunku $u$, a nie zmianę amplitudy.

**Pełna postać**

Dokładając transformację afiniczną i część enkodera przed normalizacją:

$$J(x) = \operatorname{diag}(\gamma)\cdot\frac{1}{\sigma}P\cdot A_x
\qquad\Longrightarrow\qquad
\lVert J(x)\rVert_F^2 = \frac{1}{\sigma(x)^2}\,\big\lVert \operatorname{diag}(\gamma)\,P\,A_x \big\rVert_F^2$$

Widzimy tutaj trzy czynniki, odpowiednio:
- czynnik $\sigma$ (odwrotność),
- czynnik $\gamma$ (skala),
- transformacja $PA_x$ (wrażliwość rzutu).

Chcę sprawdzić jak one wpłwają i czy są redukowalne. 

##### Analiza czynników: Czynnik $1/\sigma$ - nieredukowalny
**Redukowalność**

Obecność $1/\sigma^2$ sugeruje, że wystarczy zwiększyć $\sigma$, by zmniejszyć karę. Okazuje sie, że bez względu na skalowanie, ten parametr nie ulega zmianie.

Rozważmy przeskalowanie $a \to \lambda a$. Wtedy $\mu \to \lambda\mu$ oraz $\sigma \to \lambda\sigma$, natomiast $u$ pozostaje bez zmian, bo licznik i mianownik skalują się jednakowo. Zatem $P$ też się nie zmienia, a $A_x \to \lambda A_x$. Podstawiając:

$$\lVert J\rVert_F^2 \;\to\; \frac{1}{\lambda^2\sigma^2}\,\lambda^2\big\lVert \operatorname{diag}(\gamma)PA_x\big\rVert_F^2
= \lVert J\rVert_F^2$$

Przesunięcie $a \to a + t\mathbf{1}$ również nic nie daje: $\sigma$ w ogóle się nie zmienia, a gdyby $t$ zależało od $x$, dodatkowy człon w $A_x$ byłby proporcjonalny do $\mathbf{1}$, więc $P$ by go skasował.

Zatem otrzymujemy, że $J$ jest pochodną **odwzorowania** $x \mapsto z(x)$, **i jest niezmienniczy względem parametryzacji**.

**Interpretacja czynnika $1/\sigma^2$**

Kara rośnie odwrotnie proporcjonalnie do $\sigma^2$, więc widma o małym $\sigma(a(x))$ są karane najmocniej. **Ma to uzasadnienie geometryczne**. Kierunek $u$ powstaje przez podzielenie wektora wycentrowanego przez jego długość. Gdy $\sigma$ maleje, dzielimy przez coraz mniejszą liczbę, więc dowolnie małe zaburzenie $a$ potrafi obrócić $u$ o duży kąt. Wysokie $\sigma$ oznacza natomiast, że kierunek jest wyznaczony stabilnie.

Model minimalizujący karę ma zatem powód, żeby utrzymywać duże $\sigma$, czyli wartości przed normalizacją o **dużym rozrzucie współrzędnych**. **Jest to efekt uboczny, ponieważ deklarowanym celem kary była lokalna płaskość enkodera, a nie kontrast kodu**.

> Uwaga:
> Spodziewam się, że rozkład $\sigma(a(x))$ będzie przesunięty w górę w eksperymentach z karą kontrastywną względem zwykłego bce. W eksperymentach bez kary kontrastywnej, gdzie działa
> wyłącznie InfoNCE i ważenie Jaccardem, nic nie wywiera nacisku na $\sigma$, więc rozkład
> powinien pozostać na poziomie bce. Odchylenie od tego wzorca **oznaczałoby, że $\sigma$ jest sterowane czymś innym niż karą**.

##### Analiza czynników:  Czynnik $\gamma$ - redukowalny
**Redukowalność**

Znowu sprawdzamy czy modyfikując $\sigma$ model może zmniejszyć karę, nie wpływając na pozostałe człony straty. Tutaj okazuje się, że model może to kompensować. 

Przeskalujmy parametry `LayerNorm` przez $\lambda > 0$:

$$\gamma \to \frac{\gamma}{\lambda}, \qquad \beta \to \frac{\beta}{\lambda}$$

Cały kod skaluje się wtedy jednorodnie, $z \to z/\lambda$. Zmienia się więc wejście dekodera, głowy i projektora — i to trzeba skompensować.

Każdy z tych trzech modułów przyjmuje $z$ przez warstwę liniową $Wz + b$, gdzie $W$ i $b$ to jej własne parametry. Zwiększmy w każdej z nich wagę, zostawiając bias:

$$W \to \lambda W \qquad\Longrightarrow\qquad
(\lambda W)\left(\frac{z}{\lambda}\right) + b = Wz + b$$

Wyjście tej warstwy jest identyczne jak przed zmianą, a więc identyczne jest wszystko, co po niej następuje. Rekonstrukcja, logity i projekcje nie zmieniają się, zatem $L_{\mathrm{rec}}$, $L_{\mathrm{head}}$ i $L_{\mathrm{NCE}}$ też nie.

Sprawdźmy teraz karę. W $J = \operatorname{diag}(\gamma)\cdot\frac{1}{\sigma}P\cdot A_x$ czynniki $\sigma$, $P$ i $A_x$ zależą wyłącznie od $a$, którego nie ruszaliśmy. Zmienia się tylko pierwszy:

$$\lVert J\rVert_F \;\to\; \frac{1}{\lambda}\lVert J\rVert_F$$

Zatem otrzymujemy, że $J$ **nie jest niezmienniczy względem skali $\gamma$**. Biorąc $\lambda \to \infty$ model wypycha $L_{\mathrm{contractive}}$ do zera, nie płacąc za to nic w pozostałych członach. Jedynym oporem jest weight decay $10^{-4}$ na powiększonych wagach $W$.

**Interpretacja czynnika $\gamma$**

Geometrycznie $\gamma$ to zestaw półosi elipsoidy, na której leży $z$. Ustala on rozmiar przestrzeni ukrytej, ale nie zmienia rozmieszczenia punktów na sferze $u$. Ściskanie $\gamma$ przybliża wszystkie kody do siebie w sensie odległości euklidesowej, natomiast kąty między nimi pozostają identyczne.

Wynika stąd, że kontrakcja uzyskana przez $\gamma$ jest pozorna. Kara maleje, ale enkoder rozróżnia widma dokładnie tak samo jak wcześniej. Jest to sytuacja odwrotna niż przy czynniku $\sigma$, gdzie nacisk kary przekładał się na realną zmianę uwarunkowania normalizacji.

Degeneracja jest przy tym szersza, niż wynika z samego przeskalowania. Podstawiając $\gamma \to \gamma \odot s$ dla dowolnego dodatniego wektora $s$ i kompensując $W \to W\operatorname{diag}(1/s)$ w warstwach następnych, otrzymujemy ponownie identyczne wyjścia. Nieidentyfikowalne jest zatem całe $\gamma$, czyli $L$ stopni swobody, a nie tylko jego skala.

Ma to konsekwencję dla kształtu $\gamma$. Rozpisując normę po współrzędnych:

$$\lVert J\rVert_F^2 = \frac{1}{\sigma^2}\sum_{i=1}^{L}\gamma_i^2\,\lVert (PA_x)_i\rVert^2$$

gdzie $(PA_x)_i$ to $i$-ty wiersz. Nacisk kary na $\gamma_i$ jest więc proporcjonalny do wrażliwości tej konkretnej współrzędnej. Współrzędne reagujące najsilniej na zmiany widma są ściskane najmocniej, co powinno zwiększać rozrzut wartości $\gamma_i$.

Parametr $\beta$ nie występuje w $J$ w ogóle, ponieważ pochodna stałej jest zerem. Przesunięcie elipsoidy jest dla kary niewidoczne.

> Uwaga:
> Spodziewam się, że w eksperymentach z karą kontraktywną $\lVert\gamma\rVert$ będzie mniejsze
> niż w `bce`, przy jednoczesnym wzroście norm pierwszych warstw dekodera, głowy i projektora.
> Powinien też wzrosnąć współczynnik uwarunkowania $\kappa = \gamma_{\max}/\gamma_{\min}$,
> ponieważ nacisk kary jest różny dla różnych współrzędnych. W eksperymentach bez kary
> kontraktywnej nic nie wywiera nacisku na $\gamma$, więc obie wielkości powinny pozostać
> na poziomie `bce`.
>
> Sam spadek $\lVert\gamma\rVert$ nie rozstrzyga jednak, czy enkoder faktycznie stał się
> stabilniejszy. Rozstrzyga dopiero zestawienie go z krzywą wrażliwości kątowej: jeżeli
> $\lVert\gamma\rVert$ maleje, a krzywa się nie zmienia, kara została zaspokojona ściskaniem
> kodu, nie wygładzaniem odwzorowania.

##### Miara raportowana w analizie

**Ze względu na degenerację przez $\gamma$ nie wyliczam $\lVert J\rVert_F$**. Zamiast tego liczę wrażliwość na $u$, czyli po odrzuceniu transformacji afinicznej:

$$S(x) = \frac{1}{\sqrt{L}}\left\lVert \frac{\partial u}{\partial x}\right\rVert_F
= \frac{\lVert P A_x\rVert_F}{\sigma(x)\sqrt{L}}$$

Dzielnik $\sqrt{L}$ wynika stąd, że przemieszczenie styczne $\mathrm{d}u$ odpowiada przyrostowi kąta $d\theta = \lVert \mathrm{d}u\rVert/\sqrt{L}$, ponieważ promień rozmaitości wynosi $\sqrt{L}$. **Wielkość $S$ mierzy zatem przyrost kąta na jednostkę zaburzenia wejścia i jest wyrażona w tej samej jednostce co metryka przyjęta w analizie.**

W praktyce raportuję wersję empiryczną, nie wymagającą Jakobianu:

$$\varepsilon \;\longmapsto\;
\mathbb{E}_{x, \delta}\Big[\angle\big(u(x),\; u(x + \varepsilon\lVert x\rVert\,\delta)\big)\Big],
\qquad \delta \sim \mathrm{Unif}(S^{M-1})$$

Skalowanie zaburzenia przez $\lVert x\rVert$ czyni $\varepsilon$ wielkością względną, więc krzywa jest odporna zarówno na skalę wejścia, jak i na $\gamma$. Jest to jedyna z rozważanych miar pozwalająca porównać modele trenowane z karą kontraktywną i bez niej, ponieważ modele bez kary nie mają powodu utrzymywać $\gamma$ w tym samym reżimie.


***

#### Kontraktywność

Nie raportuję $\lVert J\rVert_F$, ponieważ jest redukowalna przez skalę $\gamma$. Zamiast tego liczę wrażliwość na $u$, czyli po odrzuceniu transformacji afinicznej:

$$S(x) = \frac{1}{\sqrt{L}}\left\lVert \frac{\partial u}{\partial x}\right\rVert_F
= \frac{\lVert P A_x\rVert_F}{\sigma(x)\sqrt{L}}$$

Dzielnik $\sqrt{L}$ wynika stąd, że przemieszczenie styczne $du$ odpowiada przyrostowi kąta $d\theta = \lVert du\rVert/\sqrt{L}$, ponieważ promień rozmaitości wynosi $\sqrt{L}$. Wielkość $S$ mierzy przyrost kąta na jednostkę zaburzenia wejścia i jest wyrażona w tej samej jednostce co metryka przyjęta powyżej.

W praktyce raportuję wersję empiryczną, nie wymagającą Jakobianu:

$$\varepsilon \;\longmapsto\;
\mathbb{E}_{x, \delta}\Big[\angle\big(u(x),\; u(x + \varepsilon\lVert x\rVert\,\delta)\big)\Big],
\qquad \delta \sim \mathrm{Unif}(S^{M-1})$$

Skalowanie zaburzenia przez $\lVert x\rVert$ czyni $\varepsilon$ wielkością względną, więc krzywa jest odporna zarówno na skalę wejścia, jak i na $\gamma$. Jest to jedyna z rozważanych miar pozwalająca porównać modele trenowane z karą i bez niej, ponieważ modele bez kary nie mają powodu utrzymywać $\gamma$ w tym samym reżimie.

Obok krzywej raportuję rozkład $\sigma(a(x))$ oraz $\lVert\gamma\rVert$ i $\kappa$, czyli wielkości przewidziane w analizie czynników. Zestawienie ich z krzywą rozstrzyga, czy kara została zaspokojona ściskaniem kodu, czy wygładzeniem odwzorowania.

*** 
