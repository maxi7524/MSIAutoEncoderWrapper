### Cel modelu

Model ma utworzyć reprezentację ukrytą widm MSI, która zachowuje informacje
istotne dla widma i jednocześnie umożliwia jego rekonstrukcję. Dla wejściowego
widma $x \in \mathbb{R}^{M}$ enkoder wyznacza kod $z \in \mathbb{R}^{L}$, a
dekoder odtwarza widmo $\hat{x}$ z tego kodu. W dalszych analizach badane jest,
czy reprezentacja zachowuje geometrię przydatną dla rekonstrukcji, predykcji i
porównywania regularizacji.

### Bieżąca konstrukcja modelu

Aktualny model składa się z 
- enkodera konwolucyjnego, dekodera, 
- projektora dla straty kontrastywnej 
- wieloetykietowej głowy klasyfikacyjnej. 

Enkoder przekształca widmo do kodu o wymiarze $L = 10$. Dekoder rekonstruuje widmo na tej samej osi binów, projektor mapuje kod do przestrzeni używanej wyłącznie przezInfoNCE, a głowa predykcyjna zwraca logity klas molekuła/addukt.

Szczegółowe kontrakty między warstwami, kształty tensorów i parametry architektury znajdują się w części „Autoenkoder i rekonstrukcja widma”.

### Ustawienia kampanii predykcyjnej

Kampania używa tkanki nerki oraz warstwowo wybranego podzbioru `10%` pikseli.
Widma są binowane liniowo w zakresie $[200, 900]$ ze skokiem `0.55` i
normalizowane metodą TIC. Dane są dzielone grupowo po `dataset_id` w proporcjach
`0.8 / 0.1 / 0.1` dla treningu, walidacji i testu.

Jedna architektura konwolucyjna jest porównywana w siedmiu ablacjach funkcji
celu, z pięcioma powtórzeniami każdej ablacji. Trening trwa maksymalnie `15` epok
przy batch size `64`, z AdamW (`lr = 0.001`, `weight_decay = 0.0001`),
przycinaniem normy gradientu do `5.0` oraz przywracaniem najlepszego checkpointu
walidacyjnego.

Szczegółowa konfiguracja znajduje się w
[architecture_predictive_experiment.yaml](../experiment_runs_configs/23_08_26_architecture_predictive/architecture_predictive_experiment.yaml).

### Zakres raportu

Raport rozdziela podstawy teoretyczne, metodologię każdej analizy i jej wyniki.
Na obecnym etapie wypełnione są metodologie rekonstrukcji i binningu. Sekcje
wyników pozostają puste do czasu ponownego przygotowania analiz.
