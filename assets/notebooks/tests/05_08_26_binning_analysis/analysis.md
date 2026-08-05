# Analiza binnerów

## Podstawy teoretyczne 
Tutaj rozpisze formalne uzasadnienie dlaczego pewne metody powinny działac, w dalszej części są przeprowadzone testy numeryczne. 

### Grid 

#### Da
Jednostajnie rozłożona siatka

#### ppm
Dla tolerancji $p$ ppm definiujemy jako dopuszczalne odchylenie
$$\Delta (\mathrm{m/z}) = \frac{p}{10^6}\cdot (\mathrm{m/z})$$
Zatem wraz ze wzrostem wartości $\mathrm{m/z}$, grid staje się **rzadszy**.  

> Uwaga:
> Wynika to z założenia że stosunek $\frac{\Delta m/z}{m/z} \approx \mathrm{const}$
 
<!-- > Jest to po prostu definicja **błędu względnego**, to jest dla różnicy na osi $\Delta \mathrm{m/z}$ otrzymujmy $p = 10^6 \cdot \frac{\Delta m/z}{m/z}$, gdzie tutaj $\Delta m/z$ oznacza błąd pomiarowy podawany przy urzadzeniu, powyżej $\Delta m/z$ oznacza jaka gęstość przypada na dany fragment $\mathrm{m/z}$ -->

> Uwaga: 
> Siatka jest tutaj rozłożona geometrycznie, więc stosując transformate logarytmiczną otrzymamy znowu liniową przestrzeń w AE 


#### Uwagi
- większość zdjęć MSI jest w ppm, powinno się to przeskalować względem **najrzadszej** gęstości, ponieważ inaczej widma będą prezrywane przez zera i algorytmy do wykrywania peaków nie zadziałają


## Przeprowadzone eksperymenty

### Ogólne

#### Oznaczenia
W analizie będą sie pojawiać oznaczenia 
- $X = \{(m_i, I_i)\}_{i=1}^n$ - oryginalne spektrum w **oryginalnej** przestrzeni $m/z$
- $\mathrm{B}(X) = \{(c_j, J_j)\}_{j=1}^d$ - spektrum po `binning'u` na ustalony wspólny grid o dokładności $\Delta m$ 
- $\mathrm{INB}(X) := \mathrm{INB}(\mathrm{B}(X)) = \{(\hat m_k, \hat I_k)\}_{k=1}^q$ - spektrum po `inverse binner'rze`

#### Zliczanie 


> Jest to ważne, ponieważ wszelkie zrzutowanie na wspólną przestrzeń, oraz późniejsze zliczanie wektorowe zabiera nam informacje o błędzie

> Implementacja tego zliczania znajduje się w `src/msi_autoencoder_wrapper/metrics/spectral_points.py`  - `match_spectral_points` 



### Analiza binnerów 
Sprawdzenie jak przekształcenie wpływa na dokładność. 

Interesowało nas 

## Analiza

### ppm vs Da
Poniewaz zliczanie błędów za pomocą ppm dało te same wyniki co Da, uznałem że nie ma to znaczenia, jeżeli chodzi o błąd i brałem pod uwagę tylko Da

> Uwaga
> Nie zmieniałem działania linear binningu, ALE jeżeli pomiar błędu się nie zmienił, to znaczy to, że nie zyskujemy dodatkowej dokładności. 