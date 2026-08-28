#### Cel analizy

Określenie wpływu forward binningu i inverse binningu na położenie pików,
intensywności oraz rekonstrukcję widma. Analiza rozdziela błąd transformacji
do wspólnej osi od błędu powstałego podczas jej odwracania.

#### Obiekty porównania

Dla widma źródłowego $X$ rozpatrywane są trzy osobne porównania:

- `forward`: $\mathrm{B}(X)$ względem $X$;
- `inverse_binned`: $\mathrm{INB}(X)$ względem $\mathrm{B}(X)$;
- `inverse_original`: $\mathrm{INB}(X)$ względem $X$.

Nie należy łączyć tych trzech porównań w jednej metryce, tabeli ani wykresie.

#### Dopasowanie pików i metryki

Piki są dopasowywane po współrzędnej $m/z$ w zadanej tolerancji. Raportowane
miary obejmują błąd lokalizacji, recall i precision pików, zachowaną intensywność,
błąd TIC, odległość Massersteina, cosine similarity oraz spectral angle.

Warianty normalizacji `raw`, `tic` i `max` są liczone oddzielnie. Nie należy
porównywać ich na wspólnej skali ani traktować ich jako tej samej miary.

#### Zakres eksperymentu

Badany jest liniowy binning dla siatki szerokości binów. Szczegółowa siatka
parametrów, tolerancje, metoda inverse binningu i liczność próby muszą być
zapisane na początku notebooka przed uruchomieniem analizy.

> #TODO: teoria binningu musi jawnie opisać założenie o pojedynczym ładunku
> jonów oraz uzasadnić powiązanie szerokości binu z rozdzielczością osi $m/z$.

#### Kryteria decyzji

Najpierw wybierana jest szerokość forward binningu, następnie parametry inverse
binnera. Wybór jest oparty przede wszystkim na błędzie lokalizacji i odległości
Massersteina, ale zawsze sprawdzany razem z pozostałymi metrykami oraz widmami
najlepszymi i najgorszymi według każdej z nich.

#### Artefakty

- [Notebook analizy binningu](../../../../notebooks/13_08_26_architecture_and_binning/binning_analysis_no_model.ipynb)
- [Niezmieniona metodologia źródłowa](../../../source_material/13_08_26_architecture_and_binning/report_binning/methodology.md)
- [Niezmieniony raport roboczy](../../../source_material/13_08_26_architecture_and_binning/report_modele/methodology_binning.md)
