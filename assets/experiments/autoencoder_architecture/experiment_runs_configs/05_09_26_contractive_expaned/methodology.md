# Metodologia: poprawa kary kontraktywnej

## Diagnoza wyjściowa

Obserwacja z dotychczasowej analizy (`part_id_02_contractive`): kara kontraktywna ściąga kody $u$ do otoczenia jednego punktu (rosnący `cloud_asymmetry`, malejąca `effective_rank`/`trace(Cov(u))` z `dimension_usage`). Model wykorzystuje mniejszy wycinek dostępnej sfery, więc różnice aktywacji między klasami maleją i głowa predykcyjna ma słabszy sygnał do rozróżniania klas.

Kara $\lVert J_z\rVert_F^2$ (`contractive_loss.py`) nie ma dolnego ograniczenia — jej globalne minimum to enkoder stały. To strukturalna przyczyna zapadania, nie efekt uboczny jednej konkretnej wagi.

## Plan

### 1. Poprawa kary kontraktywnej

**1a. Wybór metryki Jakobianu.** Trzy warianty, przy ustalonym $\lambda_{CAE}$ (obecne $0.001$):

| wariant | wzór | uwagi |
|---|---|---|
| `frobenius` (obecny) | $\lVert J\rVert_F^2$, dokładnie lub Hutchinson | już zaimplementowane |
| `spectral` | $\sigma_{max}(J)^2$, iteracja potęgowa (2–3 kroki JVP/VJP) | kara na najgorszy kierunek zamiast średniej po kierunkach; to jest "wersja z innym liczeniem Jakobianu" z Twojej listy, i porównywać ją trzeba właśnie wg metryki maksimum (`angular_sensitivity_curve` przy dużym $\varepsilon$), a nie wg wartości uśrednionej |
| `hinged` | $\max(0,\ \lVert J\rVert_F^2 - \tau^2)$ | zawias na już liczonej normie Frobeniusa — najtańsza z trzech zmian, patrz niżej |

**1b. Weryfikacja wybranej metryki.** Dla każdego wariantu (3 powtórzenia) sprawdzić `cloud_asymmetry`, `effective_rank`/`trace(Cov(u))`, `angular_sensitivity_curve` względem `grid_0000` (bez kontraktywności) i między wariantami.

**1c. Kara jednorodności na cosinusie** (Twoja propozycja "kary do cosinusa, żeby był równomierny"):

$$L_{unif} = \log \frac{1}{|B|^2}\sum_{i \neq j} \exp(-t\lVert u_i - u_j\rVert^2), \qquad t = 2$$

liczona na kanonizowanym $u$, nie na surowym $z$ (`latent_space`) — uzasadnienie w sekcji "Moje uwagi".

### 2. Przeszukanie hiperparametrów

Po wyborze metryki: $\lambda_{CAE} \in \{0.0001,\ 0.001,\ 0.01\}$ (obecna wartość razy 10 i przez 10), 5 powtórzeń na wartość. Kolejność metryka → waga jest istotna, patrz "Moje uwagi".

### 3. Warunki i decyzja

Porównanie parami po `rep_id` względem `grid_0000` (BCE, bez kontraktywności, bez kontrastywności):

- **kryterium**: `average_precision` (makro, `heads/metrics.py::evaluate_head`) lepsze niż baseline, oceniane na tle rozrzutu międzyziarnowego;
- **warunek sanity**: `cloud_asymmetry` i `effective_rank` nie gorsze niż baseline — bez tego "poprawa" predykcji może być przypadkiem pojedynczego przebiegu, a nie dowodem, że zapadanie geometrii faktycznie zostało naprawione (czyli zjawiska, które ten eksperyment ma zaadresować).

Jeśli spełnione: wariant (metryka + waga + ew. jednorodność) staje się **bazową kombinacją** do dalszych porównań strat głowy (BCE vs nnPU itd.) — nic więcej na tym etapie nie porównujemy. Jeśli niespełnione dla żadnego wariantu: bazową kombinacją pozostaje `grid_0000` (bez kontraktywności), a wynik opisujemy jako negatywny.

## Moje uwagi
- te analizy w notebookach bęe dosyć podobne do tych co są obecnie 

## Moje uwagi / sugestie

- **`hinged` to najtańsza i moim zdaniem najważniejsza zmiana.** Nie wymaga nowego sposobu liczenia Jakobianu (używa już liczonej normy Frobeniusa), tylko dolnego progu $\tau$ — poniżej niego gradient znika i kara przestaje dalej ściskać przestrzeń. $\tau$ proponuję jako medianę $\lVert J\rVert_F$ z `grid_0000` (model bez kary).
- **Jednorodność licz na $u$, nie na $z$.** Na surowym `latent_space` model może rozdmuchać $\gamma$ końcowego `LayerNorm`, zwiększając odległości bez zmiany geometrii — dokładnie ta sama luka, którą obecny `methodology.md` już opisuje dla samej kary kontraktywnej. Wymaga rozszerzenia o dostęp do `model.encoder.bottleneck_layer[-1]` (`gamma`, `beta`) w treningu, nie tylko post-hoc w analizie.
- **Ta sama luka dotyczy wariantów `spectral`/`hinged` kary kontraktywnej.** Jeśli liczysz je nadal na $z$ zamiast na $u=(z-\beta)/\gamma$, model może je obejść tym samym mechanizmem co obecną karę. Sugeruję liczyć wszystkie trzy warianty na $u$ od razu (jedna wspólna zmiana, nie trzy osobne).
- **Kolejność metryka → waga jest konieczna**, dopóki kara jest redukowalna przez $\gamma$: przeszukanie $\lambda$ przy takiej metryce mierzy głównie reakcję `LayerNorm`, nie efekt regularyzacji.
- Jeśli po `hinged` + jednorodności geometria nadal się zapada — tania alternatywa na później, poza zakresem tego etapu: denoising zamiast kary Jakobianowej (zaszumione wejście, ten sam kod, rekonstrukcja czystego widma), bo nie wymaga liczenia Jakobianu w ogóle.

## Artefakty

- Obecna implementacja kary: `src/msi_autoencoder_wrapper/training/criterions/autoencoder/regularization/contractive_loss.py`
- Metryki geometrii: `src/msi_autoencoder_wrapper/analysis/autoencoder/latent/sphere_geometry.py`
- Metryki predykcji: `src/msi_autoencoder_wrapper/analysis/autoencoder/heads/metrics.py`
- Metodologia i wyniki ablacji `grid_0000` vs `grid_0001`: `../../report/sections/analysis/part_id_02_contractive/methodology.md`, `analysis.md`
