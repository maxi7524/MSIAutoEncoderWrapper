---
format:
  html:
    toc: true
    toc-depth: 5
    number-sections: true
  pdf:
    toc: true
    number-sections: true
    keep-tex: true
---

# Raport z analiz architektury autoenkodera

## Wprowadzenie

{{< include sections/introduction/introduction.md >}}

## Podstawy teoretyczne

### Autoenkoder i rekonstrukcja widma

{{< include sections/theory/part_id_01_autoencoder_reconstruction.md >}}

### Binning i inverse binning

{{< include sections/theory/part_id_02_binning.md >}}

### Geometria przestrzeni ukrytej

{{< include sections/theory/part_id_03_latent_geometry.md >}}

### Uczenie predykcyjne i klasy wieloetykietowe

{{< include sections/theory/part_id_04_predictive_learning.md >}}

### Autoenkoder kontraktywny

{{< include sections/theory/part_id_05_contractive.md >}}

### Uczenie kontrastywne

{{< include sections/theory/part_id_06_contrastive.md >}}

### Uczenie positive-unlabelled i nnPU

{{< include sections/theory/part_id_07_nnpu.md >}}

### Porównywanie ablacjami

{{< include sections/theory/part_id_08_ablation_comparisons.md >}}

## Analizy wyników

### Binning oraz podstawowa architektura 

{{< include sections/analysis/part_id_00_binning_and_base_AE_architecture/methodology.md >}}

{{< include sections/analysis/part_id_00_binning_and_base_AE_architecture/analysis.md >}}

### Baseline architektury predykcyjnej 


{{< include sections/analysis/part_id_01_prediction_architecutre_baseline/methodology.md >}}

{{< include sections/analysis/part_id_01_prediction_architecutre_baseline/analysis.md >}}


### Kontraktywność

{{< include sections/analysis/part_id_02_contractive/methodology.md >}}

{{< include sections/analysis/part_id_02_contractive/analysis.md >}}

### Kontrastywność

{{< include sections/analysis/part_id_03_contrastive/methodology.md >}}

{{< include sections/analysis/part_id_03_contrastive/analysis.md >}}



### Rekonstrukcja widm

{{< include sections/analysis/reconstruction/methodology.md >}}

{{< include sections/analysis/reconstruction/analysis.md >}}

### Porównanie rekonstrukcji 

### Predykcyjność reprezentacji

### Przestrzeń ukryta

### nnPU

## Synteza i decyzje

### Rekonstrukcja - pełna reprezetnacja bazy 

#### pomysł 

Można wygenerować sztucznie widma, na całym zakresie, żęby zasymulować całą baze. Coś takeig jak w pep-compassie się robi z peptydami nieoznakowanymi. 

Wtedy można zrobić dwie fazy treninwogwe
1. trenowanie bazy 
2. trenowanie modelu 

Nie widziałem czegoś takiego a wydaje mi sie ciekawe

#### Implementacja 
Ja bym to zrobił tak, że najpierw ternujemy ogólny AE na pewnej reprezentacji widma (zakres m/z) żeby umiał odtwrzac baze, potem oddajemy te widma. 

Chodzi mi o ten przypadek:
![alt text](<tmp_negatywny przypadek_widma_brak_reprezentacji.png>)

### Uczenie kontraktywne 

#### Pogorszeniewyników - zmiana bazy

Można zmienićbaez opo któerj jacobian jest zlcizony, nie po zmianie intestnwynosci w danych m/z, tylko sam błąd lokal;izacjyny ,czy nie bralibyśmy pod uwagę, jak zmieina się intensywność, tylko bralibyśmy pod uwagę jak zmienia sie lokalizacja peaków. Peaków wtedy możęmy otrzymac lepsze wyniki ....

### Uczenie cosinuse

#### Dać akę za mały rozkął cosinusa xd. 

### Co był zrobił (co teraz trzeba wykonać) 

- wybrać jakiś contrastive wybrać, on ewidetnie poprawia precyzje
- sprawdzić raz ejszcze contractive, mozę będzie poprawała jak torhcę inaczje paraemtry doadmyalbo coś w tym rodzaju (problem moim zndiaem polga na tym ze rperzestrzeń się zapada, trzeb by ja rozszerzyć jakos (dać karę na tego consunowa czy coś)
- no srpawdzić te jaccardy o bce bez tej normalizacji klasowej

WAŻNE - teraz trzeba zroibć to na dobrym datasetcie, to jest split po lklasach równo mierny, żęby te predykcje poprawić. 


czyli taki trening trzeba puścić (#TODO - nasępne weekend) 
- ustalmy splita po labelach (łączymy teraz te dodane datasety) 
- UWAGA: NIE ROBIMY PONOWNEGO MERGE'a - byśmy musieli wszystko od nowa pusśic i to zepsuje wyniki które aktualnie mamy - bęzdie to nie odtwarzalne (później całośc porpawimy, to jest dodamy końcowy poprawiony dataset, puścimy jeszcze raz wszystko wtedy)
- powtórna walidacja
  - dodać koszt w contractive do tego rozkładu \theta, że ma byc szeroki (żeby cąłą przestrzeń to wykorzystywało bo zapada sie rozkąłd tgo i bce itd. nei mogą tego dobrze wykryć - moim zdaniem z tego są te błedy, bo topoligcznie to działa) - BĘDĘ WALCZYĆ  
  - zrobić contrastive bez contarctive z porpawnymi parametrami (sprawdzić tą nie normalizowaną po weights) (tutaj dwa warianty bce bysmy wtedy zrobili) 
  - nnPU nie działa - olewamy (ciężko nam też o priora) 
  - DODATKOWO DO KAŻDEGO (nowe warianty) 
    - pretrain (uczenie bazy)
      - ja bym dodał pretrening, żeby baze zrobić, wtedy można by zarządząć szerokiego rozłóżenia pików po $S$, oraz nauczyć go całej bazy (jak to wtedy zrobić ???) 
      - można dodać do tego preterningu uczenia jakiegoś na podstawie baz, są tam infromacje dtycząe w jakich m/z związki się powinny znaleźć, więc możemy zrobić taki ogólny predykator (to by było mocne) 
      - ważne: - tutaj go właściwei przeuczamy, bo ma na pamiec się tego nauczyć !!!
      - pomysły 
        - dobieramy jakoś permtuacyjne 
          - pojedynczone widma
          - kilka widma it.d 
          - na róznych zakresach
          - oraż żeby różrnzac zkardsy m/z żęby to mapwanie było cwanńzzse 
    - WAŻNE - rozszerzenie dziedziny widm
      - trzeba wymyślić jak można zrobić coś takiego, żeby go poza wspólnmy zakresem trenowc (bo wtedy dosatyjemy widmo na podstrpzensti bazy, więc żeby nei uzwał na siłe pozostąłych zakrsów (no sygnał zero, ale model by musiał nauczyć isę rozróżniąć żęb yszuu nei robić !!!! - że zamiast zer jakas inna operacja żeby bałagnu nie robić - bo maym inpfirancje na wejściu z ilu elentów to sie składa - podczas mapwoani można byto jakosrpzechzywyic ??? ))
- 

## Załączniki

### Artykuły

[Indeks artykułów](articles/README.md)

### Notebooki

- [Binning bez modelu — 13_08_26](../notebooks/13_08_26_architecture_and_binning/binning_analysis_no_model.ipynb)
- [Rekonstrukcja i architektury — 13_08_26](../notebooks/13_08_26_architecture_and_binning/reconstruction_architecures_analysis.ipynb)
- [Zbiór notebooków predykcyjnych — 23_08_26](../notebooks/23_08_26_architecture_predictive/)

### Konfiguracje i materiały źródłowe

- [Konfiguracje przebiegów](../experiment_runs_configs/)
- [Raporty, wykresy i wyniki przeniesione bez edycji](source_material/)
