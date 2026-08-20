# 13_08_26_architecture_and_binning

## Wstęp

### Cel 
Celem jest dobranie odpowiedniej architektury AE, do odtwarzania danych, oraz binningu który dobrze rekonstruuje oś. m/z, gdzie przez dobrą rekonstrukcje mamy na myśli widma sparowane na odległość $\pm 1 \mathrm{m\backslash z}$, ponieważ korzystamy z tego że MALDI, wszystkie metabolity mają ładunek $\pm 1$. 

## Metodologia 

### Binning 
#### Forward binning (binning) 
Tutaj stosujemy zwykły liniowy binning, zyli tworzym wspólny oś z pewnymi przedziałami i przypisujemy dczy peak ląduje w danym przedziale. Jak mamy kilka przypasowań w tym samym przedziale, to bierzemy ich sumę, ponieważ wartości w tym otoczeniu, to powinny być te same molekuły, tylko że z lekką pertrubacją stanu w układzie. 

Robimy testy dla podziału $\{0.01, 0.45, 0.5, 0.55, 1.0, 2. \}$, żeby mieć informacje o minimalnym błędzie ($\Delta \mathrm{m\backslash z} = 0.01$), oraz interpretacje metryki masserstein'a ($\Delta \mathrm{m\backslash z} \in \{1, 2\}$)

#### Inverse binning 
#TODO - opiasć te metody dokładniej, nie trzeba badać każdej wystarczy przebadanie tego centroidalnego, ponieważ jak będzei stabilny numeryzcnie to nei ma sensu sprawdząć kocnepcyjnie gorszych metod. Poprzednie eksperymenty mówią nam jaka jest interpretacja gorszej metody. 

Wszędzie trzeba zastosowac normalizacje TIC 


### Architektury 

W przypadku dobrania architektury, musimy rozwiązać dwa problemy
1. Rekonstrukcja obrazu
2. Stworzenie odpowiedniej geometrii w latencie

Najpierw przebadamy najlepszą architkeurę AE. Dla tak dobranej architektury (backbone), będziemy porównywać następnie regularyzacje na nią wpływają.  

#### Rekonstrukcja  

##### MLP-AE (Baseline)
Najprostsza architektura którą można zaimplementować jest to MLP-AE. Intuicyjnie bierze ona pod uwagę informacje globalną. W przypadku zmniejszonej osi możemy to zaimplementować, ponieważ wymiar ma rozmiar około $\approx 1200 - 1500$.  

Artykuł w których się pojawia:
- [(Abdelmoula et. al 2021](https://www.nature.com/articles/s41467-021-25744-8), tam 

##### 1D Convolutional AE
Intuicja jest za tym taka, że wykrywamy peaki za pomocą filtrów. Każdy taki filtr powinien zasadniczo wykrywac peaki o odpowiedniej obwiedni i potem na bazie tych kilku peaków stwierdzić czy dany związek występuje czy nie. 

Jest to nasza aktualna arhcitekrua ale osobiscie uważam że nie jest to dobry pomysł, i te sieci osiagają dore wyniki, poniewaz nie ma odpowiedniego testu walidacyjnego, moim zdaniem te sieci powinny byc przeuczone. Moim zdaniem żeby miało to sens to trzeba by robić batche po otoczeniu, wtedy trzeba te tkanki oryginalne zachować. 

Pojawia się on tutaj:
- [Bitto el. al. 2024](https://www.researchgate.net/publication/380913783_Enhancing_mass_spectrometry_imaging_accessibility_using_convolutional_autoencoders_for_deriving_hypoxia-associated_peptides_from_tumors)

#### Geometria w latencie 

##### Contractive AE
Formalnie dodajemy regularyzacje, po enkoderze, to jest:
$$\lambda \vert J_E(x) \vert$$
Intuicyjnie implementuje regularyzacje która ma własność "kontraktywności", czyli żeby każdy punkt był unikalnie przypisywany. 

Pojawio się to na icml'u:
- [Salah RIFAI 2011](https://icml.cc/Conferences/2011/papers.php.html)

#### VAE 

W funkcji kosztu gdzie parametrem jets miara Kulbacka-Leiblera, sprawdzał bym $\beta \in \{0.1, 0, 0.5\}$ 

## Eksperymenty 

Co dokładnie testowaliśmy

