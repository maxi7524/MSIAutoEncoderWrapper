### Architektura

Wszystkie $35$ przebiegów używa tych samych czterech komponentów, różniąc się wyłącznie członami straty nałożonymi na ich wyjścia.

#### Enkoder

Trzy bloki konwolucyjne `Conv1d → LayerNorm → ReLU`, następnie wąskie gardło `Flatten → Linear → LayerNorm`. Liczba kanałów rośnie, a następnie maleje, natomiast oś widmowa kurczy się przy każdym kroku:

| etap | operacja | kanały | kernel | stride | szerokość wyjścia |
|---|---|---|---|---|---|
| wejście | — | $1$ | — | — | $1273$ |
| blok 1 | `Conv1d + LayerNorm + ReLU` | $1 \to 32$ | $5$ | $3$ | $423$ |
| blok 2 | `Conv1d + LayerNorm + ReLU` | $32 \to 16$ | $7$ | $3$ | $139$ |
| blok 3 | `Conv1d + LayerNorm + ReLU` | $16 \to 8$ | $3$ | $3$ | $46$ |
| projekcja | `Flatten(8 \times 46 = 368) → Linear(368, 10) → LayerNorm` | — | — | — | $L = 10$ |

Architektura jest analogiczna do analizy rekonstrukcyjnej, przy czym zmieniłem rozmiar pierwszego i drugiego filtra. Liczba parametrów pozostała ta sama, więc nie wprowadza to dodatkowej zmienności, a celem było zmniejszenie szumu wynikającego z długości obwiedni.

> Uwaga:
> Wszystkie normalizacje to `LayerNorm`, nie `BatchNorm`. Jest to wymóg kary kontraktywnej:
> przy `BatchNorm` Jakobian $\partial\,\mathrm{enc}(x)/\partial x$ zależałby od pozostałych
> elementów batcha przez statystyki bieżące, więc nie byłby dobrze określony jako pochodna
> funkcji pojedynczej próbki.

#### Dekoder

Odbicie lustrzane enkodera. `Linear → LayerNorm → Reshape` rozwija kod $L$-wymiarowy z powrotem do kształtu $(8, 46)$, następnie trzy bloki `ConvTranspose1d` w odwrotnej kolejności kanałów ($8 \to 16 \to 32 \to 1$, z tymi samymi kernelami i strideami) odtwarzają szerokość $M = 1273$. Wartość `output_padding` w każdym bloku jest dobrana tak, żeby szerokość wyjścia zgadzała się dokładnie z odpowiadającą szerokością wejścia enkodera, więc nie ma przycinania.

Każdy blok poza ostatnim to `ConvTranspose1d → LayerNorm → ReLU`. Ostatni to `ConvTranspose1d → softplus → normalizacja TIC`, gdzie `softplus` wymusza nieujemne intensywności, a normalizacja TIC sprowadza rekonstrukcję na tę samą skalę co wejście.

#### Projektor

`Linear(10, 10) → LayerNorm → ReLU → Linear(10, 64)`. Odwzorowuje kod $z$ na osobną projekcję $g(z)$, używaną wyłącznie przez stratę kontrastywną.

#### Głowa klasyfikacyjna

`Linear(10, 128) → ReLU → Dropout(0.1) → Linear(128, 508)`, działa na $z$. Zwraca surowe logity, nie prawdopodobieństwa, ponieważ sigmoida i softplus są stosowane wewnątrz implementacji strat głowy, co daje stabilniejszy numerycznie rachunek.

***
