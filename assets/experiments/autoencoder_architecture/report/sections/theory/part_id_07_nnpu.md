##### Zmian interpretacji negatywów w głowie predykcyjnej - nnPU 

**Wstęp**

Drugim rozwiązaniem jest zmiana samej straty głowy. Ważenie Jaccardem działa poprzez działanie na stratę kontrastywną, ale nie zmienia interpretacji zera w $y$. Tutaj zmieniam ją wprost: $y_{i,c} = 0$ traktuję jako **nieoznaczone**, nie jako potwierdzony negatyw.

Metoda pochodzi z pracy Kiryo i in. (NIPS 2017), gdzie sformułowana jest dla klasyfikacji binarnej. Stosuję ją niezależnie dla każdej z $C = 508$ klas.

**Oznaczenia**

Poniższe wielkości definiuję dla ustalonej klasy $c$; dla czytelności pomijam indeks $c$ tam, gdzie nie prowadzi to do niejednoznaczności.

| symbol | znaczenie |
|---|---|
| $f_c(x)$ | surowy logit głowy dla klasy $c$ |
| $\ell(t, y)$ | strata poniesiona przy predykcji $t$, gdy prawdą jest $y$ |
| $\pi_c$ | prior klasy — udział pikseli, w których molekuła $c$ **faktycznie** występuje |
| $p_P(x)$ | rozkład widm w pikselach zawierających molekułę $c$ |
| $p_N(x)$ | rozkład widm w pikselach jej niezawierających |
| $p(x)$ | rozkład brzegowy wszystkich widm |
| $\mathbb{E}_{P}[\,\cdot\,]$ | wartość oczekiwana po pikselach **anotowanych** dla $c$ ($y_c = 1$) |
| $\mathbb{E}_{U}[\,\cdot\,]$ | wartość oczekiwana po pikselach **nieoznaczonych** ($y_c = 0$) |
| $\mathbb{E}_{N}[\,\cdot\,]$ | wartość oczekiwana po pikselach faktycznie negatywnych — **nieobserwowalna** |

> Uwaga:
> Kluczowe jest rozróżnienie $\pi_c$ od obserwowanej częstości anotacji. Jeżeli w $2000$ pikseli
> molekuła $c$ jest anotowana w $20$, to obserwowana częstość wynosi $0.01$, ale $\pi_c > 0.01$,
> ponieważ **zakładamy, że część pikseli** faktycznie ją zawierających nie przeszła filtru FDR. Gdyby obie wielkości były równe, problem, nastąpiła by anihilacja problemu, który próbuje tu rozwiązać .

**Wyprowadzenie**

Punktem wyjścia jest ryzyko w klasyfikacji nadzorowanej, gdzie znamy oba zbiory:

$$R_c = \pi_c\,\underbrace{\mathbb{E}_{P}\big[\ell(f_c, 1)\big]}_{\text{koszt na pozytywach}}
\;+\; (1 - \pi_c)\,\underbrace{\mathbb{E}_{N}\big[\ell(f_c, 0)\big]}_{\text{koszt na negatywach}}$$

Pierwszy człon umiemy policzyć, sumuję stratę po pikselach anotowanych. Drugiego nie jesteśmy w stanie, ponieważ nie znamy zbioru $N$. Wynika to z założenia, że piksele o $y_c = 0$ to mieszanina prawdziwych negatywów i pominiętych pozytywów.

Możemy go jednak wyznaczyć pośrednio. Rozkład wszystkich pikseli jest mieszaniną obu klas:

$$p(x) = \pi_c\, p_{P}(x) + (1 - \pi_c)\, p_{N}(x)
\qquad\Longleftrightarrow\qquad
(1 - \pi_c)\,p_N(x) = p(x) - \pi_c\,p_P(x)$$

Całkując obie strony względem $\ell(f_c, 0)$ otrzymujemy tożsamość między wartościami oczekiwanymi:

$$(1 - \pi_c)\,\mathbb{E}_{N}\big[\ell(f_c, 0)\big]
= \underbrace{\mathbb{E}_{U}\big[\ell(f_c, 0)\big]}_{\text{mierzalne}}
- \;\pi_c \underbrace{\mathbb{E}_{P}\big[\ell(f_c, 0)\big]}_{\text{mierzalne}}$$

Prawą stronę możemy interpretować jako **wszystkie** nieoznaczone piksele były negatywami, następnie **korygujemy** nadmiar, który definiujemy poprzez $\pi_c$, który mówi ile takich oznaczeń wykryć nie powinniśmy. Poprawkę szacuję na pikselach anotowanych, ponieważ one pochodzą z $p_P$.

Podstawiając, otrzymujemy ryzyko PU złożone wyłącznie z wielkości obserwowalnych:

$$R_c = \pi_c\, \mathbb{E}_{P}\big[\ell(f_c, 1)\big]
+ \underbrace{\mathbb{E}_{U}\big[\ell(f_c, 0)\big] - \pi_c\, \mathbb{E}_{P}\big[\ell(f_c, 0)\big]}_{\text{estymator } (1-\pi_c)\mathbb{E}_{N}[\ell(f_c,0)]}$$

Jest to estymator nieobciążony, ale estymator może być ujemny. Zachodzi to wtedy, gdy dopasuje się do pozytywów treningowych, wtedy $\mathbb{E}_{P}[\ell(f_c, 0)]$ rośnie i odejmowany człon, przeważy. Minimalizacja prowadzi do rozbieżności

Rozwiązaniem jest wymuszenie nieujemności przez obcięcie:

$$\boxed{\;R_c = \pi_c\, \mathbb{E}_{P}\big[\ell(f_c, 1)\big]
+ \max\!\Big(0,\; \mathbb{E}_{U}\big[\ell(f_c, 0)\big] - \pi_c\, \mathbb{E}_{P}\big[\ell(f_c, 0)\big]\Big)\;}$$

Człon $\max(0, \cdot)$ jest jedyną różnicą względem estymatora nieobciążonego. Obcięcie wprowadza obciążenie, ale zanika ono wykładniczo wraz z liczbą próbek, a estymator pozostaje zgodny.

> Uwaga:
> Obcięcie liczone jest na mini-batchu, nie na całym zbiorze. Jest to górne ograniczenie. Wynika to z tego, że funkcja $\max$ jest funkcją wypukłą i zachodzi
> $\max\{0, \frac{1}{N}\sum_i z_i\} \le \frac{1}{N}\sum_i \max\{0, z_i\}$.

**Estymacja priorów**

$\pi_c$ wyznaczam jako obserwowaną częstość anotacji klasy $c$ w splicie treningowym, przemnożoną przez stały mnożnik $1.5$ i obciętą do $[10^{-4},\, 0.99]$.

> Uwaga:
> Mnożnik jest założeniem o stopniu niedoanotowania zbioru, nie wielkością mierzoną. 
> 
> W eksperymentach Kiryo i in. zaniżenie $\pi$ szkodziło bardziej niż zawyżenie, a najlepsze wyniki testowe uzyskiwano przy wartościach nieco powyżej prawdziwego prioru. Mnożnik $> 1$ bazuje na tej obseracji, jest wybrany bez żadnej podstawy teoretycznej. 

**Interpretacja mechanizmu**

Zwykłe BCE podczas optymalizacji przenosi każdy logit na pozycji nieoznaczonej w stronę negatywu, nnPU wstrzymuje dokładnie $\pi_c$ tego działa, zakładając, że taki ułamek nieoznaczonych to ukryte pozytywy. 

Dla klas rzadkich, gdzie $\pi_c$jest małe, obie straty zachowują się niemal tak samo, różnica narasta wraz z różnorodnością klas.

> Uwaga:
> Obu strat nie sumuję. Nakładałyby powodowało byto nakładanie **przeciwnych** gradientów na te same logity: jedna podnosiłaby je o wielkość wynikającą z $\pi_c$, druga obniżała o wielkość wynikającą z BCE.
> 
> Nie testowałoby to żadnej z dwóch hipotez o znaczeniu brakujących anotacji. Są to więc alternatywne ramiona ablacji, nie składniki.

**Relacja do ważenia Jaccardem**

Obie metody adresują ten sam problem w różnych miejscach architektury:

| | miejsce działania | co zmienia |
|---|---|---|
| ważenie Jaccardem | mianownik $L_{\mathrm{NCE}}$ | siłę odpychania podobnych pikseli |
| nnPU | $L_{\mathrm{head}}$ | interpretację zera w $y$ |

Są zatem niezależne i mogą wystąpić w jednym modelu, badam to aktualnie osobno. 

***
