# Do zaimplementowania (dla codexa)

Kontekst i uzasadnienie: `methodology.md` w tym samym katalogu. Ten dokument opisuje wyłącznie brakujący kod — nie zmienia zakresu ustalonego w `methodology.md`.

## 1. `ContractiveLoss` — warianty metryki (`penalty_metric`)

Plik: `src/msi_autoencoder_wrapper/training/criterions/autoencoder/regularization/contractive_loss.py`

Obecny stan: tylko $\lVert J\rVert_F^2$, dwie metody liczenia (`calculation_method`: `exact_autograd_jacobian`, `approximate_hutchinson_vjp`). Brak wariantów metryki.

Do dodania:
- nowy parametr konstruktora `penalty_metric: str = "frobenius"`, `SUPPORTED_PENALTY_METRICS = frozenset({"frobenius", "spectral", "hinged"})`, walidacja przez `raise_validation_error` (wzorem istniejącej walidacji `calculation_method`).
- `spectral`: $\sigma_{max}(J)^2$ przez iterację potęgową (2–3 kroki) na naprzemiennym VJP/JVP. Obecny kod liczy tylko VJP (`torch.autograd.grad` na `(latent*direction).sum()` względem `inputs`); potrzebny też JVP (`torch.autograd.grad` z `create_graph=True` na iloczynie z wektorem po stronie wejścia, albo `torch.func.jvp`, w zależności od wersji PyTorch w repo — sprawdzić przed implementacją). Test jednostkowy: na małym modelu porównać wynik z jawnie zmaterializowanym Jakobianem (`exact_autograd_jacobian` już to daje jako punkt odniesienia) i `torch.linalg.svdvals(J)[0]**2`.
- `hinged`: $\max(0,\ \text{penalty} - \tau^2)$ nałożony na wynik dowolnej z powyższych metryk (domyślnie na `frobenius`). Nowy parametr `hinge_threshold: float | None = None`, wymagany (walidacja) gdy `penalty_metric == "hinged"`. Test: gradient musi zerować się poniżej progu (sprawdzić `torch.autograd.grad` po `inputs` daje same zera dla batcha ze sztucznie małym $\lVert J\rVert_F$).

## 2. Przestrzeń kanonizowana ($u$ zamiast $z$)

Ten sam plik. Obecnie `latent = model_outputs[self.latent_source]` to surowe `z` (`latent_space`, po `LayerNorm`, przed kanonizacją) — kara jest redukowalna przez $\gamma$ (opisane już w `report/sections/analysis/part_id_02_contractive/methodology.md`, sekcja "Wpływ LayerNorm...").

Do dodania:
- zaimplementować hook `on_phase_start(self, model, dataset, transient_cache)` (kontrakt w `training/criterions/base_criterion.py`): zapisać `self._layer_norm = model.encoder.bottleneck_layer[-1]` (ten sam moduł, który `sphere_geometry.encoder_layer_norm_parameters` czyta post-hoc, ale tu **bez** `.detach()` — potrzebny graf autograd na `weight`/`bias`).
- w `forward`: gdy `penalized_space == "u"`, `u = (latent - self._layer_norm.bias) / self._layer_norm.weight` i różniczkować penalty po `u` zamiast po `latent` (VJP/JVP mnożone przez `u`).
- nowy parametr `penalized_space: str = "z"`, `{"z", "u"}`; domyślne `"z"` zachowuje dokładną zgodność wsteczną z `grid_0000`/`grid_0001`.
- Test: model z $\gamma \neq 1$ — kara na `u` nie powinna maleć przy samym przeskalowaniu $\gamma$ (w przeciwieństwie do kary na `z`, gdzie powinna).

## 3. Nowa kara: `UniformityLoss`

Nowy plik: `src/msi_autoencoder_wrapper/training/criterions/autoencoder/regularization/uniformity_loss.py`.

- rejestracja: `@CriterionsManager.register_criterion("autoencoder", "regularization", "UniformityLoss")`, dziedziczy `MSIRegularizationCriterion` (`autoencoder_base_criterions.py`).
- $L_{unif} = \log\frac{1}{|B|^2}\sum_{i \neq j} \exp(-t\lVert u_i-u_j\rVert^2)$, $t$ konfigurowalny (domyślnie `2.0`), liczona na kanonizowanym $u$.
- Potrzebuje tego samego dostępu do `gamma`/`beta` co punkt 2 — rozważyć wspólny mixin (np. `CanonicalizedLatentMixin` z hookiem `on_phase_start` i metodą `canonicalize(latent)`), żeby nie duplikować logiki między `ContractiveLoss` i `UniformityLoss`.
- `requires_input_grad = False` (liczona wyłącznie na `model_outputs`, bez różniczkowania po wejściu) — inaczej niż `ContractiveLoss`.
- Test: syntetyczny fixture — $u$ rozłożone jednorodnie na sferze musi dawać niższą wartość niż $u$ skupione w jednym punkcie (bez treningu, bez GPU).

## 4. Konfiguracja eksperymentu (YAML)

Do utworzenia dopiero po zatwierdzeniu `methodology.md`: `experiment_runs_configs/05_09_26_contractive_expaned/*.yaml`, wzorowane na strukturze `23_08_26_architecture_predictive/architecture_predictive_experiment.yaml` (`objectives.values`, blok `regularization.contractive.params`, YAML anchory dla współdzielonych bloków `reconstruction`/`heads`).

- Etap 1 (wybór metryki): 3 warianty `penalty_metric` × 3 powtórzenia = 9 tasków, $\lambda_{CAE}=0.001$ ustalone.
- Etap 2 (waga): zwycięska metryka × $\lambda \in \{0.0001, 0.001, 0.01\}$ × 5 powtórzeń = 15 tasków.
- Etap 3 (jednorodność, opcjonalnie jeśli etap 2 nie spełnia warunku sanity): zwycięzec + `UniformityLoss` × 5 powtórzeń.

Nie jest to zadanie kodowe (poza `criterions_manager`/rejestracją, która już istnieje generycznie dla dowolnej liczby nazwanych wpisów w `regularization`), tylko konfiguracyjne — codex może to zrobić po tym, jak punkty 1–3 będą gotowe i przetestowane.

## Co NIE wymaga implementacji (już istnieje)

- Metryki geometrii: `cloud_asymmetry`, `dimension_usage` (`effective_rank`, `trace`), `angular_sensitivity_curve`, `rsa_spearman` — wszystkie w `analysis/autoencoder/latent/sphere_geometry.py`.
- Metryka predykcji: `average_precision` (makro) w `analysis/autoencoder/heads/metrics.py::evaluate_head`.
- Kanonizacja $u$ do celów analizy post-hoc (numpy, po treningu) — `sphere_geometry.canonicalize`. Brakuje wyłącznie wersji różniczkowalnej *wewnątrz* treningu (punkt 2 powyżej) — to jest jedyna nowa część tej funkcjonalności.
- Wielo-komponentowe sumowanie strat w jednej kategorii (`CompositeLoss` w `criterions_manager.py` już sumuje dowolną liczbę nazwanych wpisów per kategoria) — dodanie `UniformityLoss` obok `ContractiveLoss` pod `regularization:` w YAML nie wymaga zmian w orkiestracji.
