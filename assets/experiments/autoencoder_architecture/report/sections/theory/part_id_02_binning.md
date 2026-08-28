### Binning i inverse binning

#### Forward binning

Forward binning odwzorowuje rzadkie widmo na wspólną, gęstą oś $m/z$ o zadanej szerokości binu. Intensywności punktów przypisanych do tego samego przedziału są sumowane. Transformacja ujednolica wymiar wejścia autoenkodera, ale może łączyć bliskie piki i wprowadzać błąd położenia.

Nadaje on jednocześnie interpretacje "bazy" naszej przestrzeni spektrometrycznej. Idealnie zatem było by mieć **jak największą gęstość**, która jednocześnei jest na wystarczająco mała, żeby model mógł uchwycic nature widm, to jest "nature obwiedni".

#### Inverse binning

Inverse binning tworzy rzadką reprezentację na podstawie widma po binningu. Jego ocena musi być rozdzielona od oceny forward binningu: porównujemy osobno $\mathrm{B}(X)$ z $X$, $\mathrm{INB}(X)$ z $\mathrm{B}(X)$ oraz $\mathrm{INB}(X)$ z $X$.

#### Założenie o ładunku jonów

Zakładamy, że wszystkie jony mają ładunek $1$. 

> #TODO: opisać dokładnie założenie, że analizowane jony mają pojedynczy ładunek,
> oraz wyprowadzić jego konsekwencje dla odległości na osi $m/z$, tolerancji
> dopasowania i interpretacji szerokości binu.

#### Odległość Massersteina

Odległość Massersteina interpretuje różnicę między widmami jako koszt transportu intensywności po osi mas. Jest używana jako miara błędu lokalizacji i powinna być interpretowana wraz z metrykami dopasowania pików.

<!-- Powiązany artykuł: ../../articles/masserstein_loss.pdf -->
