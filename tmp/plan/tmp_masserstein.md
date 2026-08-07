

## Co trzeba zaimplementować
### wizualacja 
  - przeymśleć co chcemy wizualizować, żeby móć szybkie raporty robić i to zestawiać
    - będziemy mieli w domysle moduł od robienia plotówm, tam będą wszystkie wykresy 
    - jak będziemy mieli autoamtyzacje to zrobimy skrypty który pozwala wykonywac szybko taką analize będzie to tak działać, że część rzeczy bnęzcei sama porównać drobić ploty i zamieszczać od razu w jakimś .md, dzięki temu cały proces będzie można zautoamtyzować.
### Testy numeryczne - jakie hiperparametry są sensowne
  - trzeba dodać cały format pod snakemake'a, zastanoawić się czy to dać do głónwego repo czy osobno, wtedy za pomocą snakemake'a będzie można puścić całą serie różnych testów z różnymi kombinacjami i będzie można to sobie porównać. głownie nas będzie intertesowało jak ta rekosnturkcja przebiega i czy jest to sensowne. 
  - tutaj mogła by się przydać jaka pomocnicza funkcja która odpowiedno by sprawdza tye obrazy to jest odległści pomiedzy oryignalnym i obrazem po tych ibnnerach itd. żeby zobacyć na zcym błąd polega - moze byc problem z binnerami . inverse binnerami wtedy będzie widać na różnicah tyhc metryk że funkcja kosztu zero, ale dystans jest duży (wtedy to jest bez sensu) 


### Implementacje (tutaj jest wiekszy porządek) 

1. Bierzemy obrazki z opisem 
2. nastęnei tworzymy z tego dużą próbkę biorąc po prostu konkarttynacje tych obrakzów z tymi opisami
3. trenujemy ogólny encoder decoder na ten typ tkanki (czyli na dany typ tkanki / dany gatunek) i wtedy klasmai są
   1. condition - zdrowa chora (jaka przypadłość)
   2. molekuł które występują 



wtedy implementacja
- skrypt, który pobiera te obrazki odpowiednio z metapsace'a i kataloguej je w workspace'cie 
- skrypt, który który zbiera z tych obrazków tych otagowane zdjęcia i tworzy jeden wspólny dataset
- skrypt którty przechodzi te zdjęcia z opisami (głownie metaspace) i zbiera je w jeden obrazek
  - uwaga to nei jest takei trywialne, poniewaz przenosimy opisy to trzeba zrobić jakies mapowanie indeksów odopwiednioe żęby tego nie pogubić, będzie itak kilka róznych plików zrobić, na ten moment myśle o:
    - 
- dataset dodać który zbiera odpowiednio metadane, 

- następnie model który tego się uczy
  - autoencoder dowolny - to dotyczy dowolnego obrazka, więc to nie jest problematyczne 
  - trezba nadal te dwa problemty rozwiąząc
    - jeden związany z regularyzcją 
    - drugi z klasyfikacją 
    - idea jest taka że AE drobiym per typ ale rozrózniamy: stan pacjenta (jakie choroboy / objawy ma), molekluły (jaka jest adnotacja z `metaspace'a np.) 

- jak to zrobimyto analiza, trzeba odpoweidnio wizualacje zaimpemtnować itd. 
  - najpierw zrobimy .ipynb żeby sobie to prztetsowac - będziemy ścieżkez tym wytrenowanmy modelem i daasatetme podłączać, jak ustliamy spójnąwersje jak to bęziem analizowac to dodamyten skrpyt o którym wczęsniej pisaliśmy 


Wtedy analiza:
- jak kategoryzacja na head'ach wygląda i czy jest dora klasyfikcja
- jak odtwarznaie widm wygląda
  - można per obraazek z którego były wzięte, można zrobić jakis taki metadane dodać do imzML, który to trzeba albo osobny csv który zawiera inforamcje o mapowaniu 
- tSNE - czy to jest paplanina czy to się grupuje - czy się gruopują molekuły, czy warunki chory / zdrowy się wyróżniają czy mozemy cos tutaj wychwycić. oraz inne metody porównawcze
- do analizy będzie trzeba jeszzce dodać elementy związane z analizą samego "złącznego" datasetu, czyli 
- można też zrobićjeden eksperymetny gdzie byśmy mieli po prostu dwuwymiar i trój wymiar żeby zrozumiec jak to się możę grupować (tak na pałe - można się opbawić :) )


dodakowo (to co wczęsniej pisałem) trzeba zaimpelntwowac 
- ogólniony sposób robienia tych runnó, ja bym zrobił to tak, że w assests, zrobiłbym jaisfolder który za to odpowiada i tam wtedy odpwiednioe skrypty zamieścił itd. (bo to ja był chcciał przez snakemake'a robić, czyli byśmy wtedy podawali configa odpowiednio i tego runa robii, wtedy byto było najczystszte, 
- skrypty związen z dataseteami ozcywicie w `assets/scripts/datasets` 
- problem z łązcnym encoderem i dodawaniem tej implemetnacji ze wspólnym decoderem zarzucamy - to na eten moemn nie ma snesu 
- jak mamy to zarządzanei zdjęciami, to zmieinamy struktue w workspace minimalnie, teraz robimytak zenie w imgs jest jest wszystko tylko imgs/<ims_folder_name>/pliki zde zdjęciem. będzie to koniecnze z powodum, 
- jak te skrypty do analizy z datsatem to można dodac moduł który zajmuję się opisanymi zdjęicami i właśnei zwaraca takei informacje jak ilośc elementów w klaasch itd




# Na teraz 27.07.2026

## Sprawdzenie 
- [x] Przejrzenie PR 
- [ ] sprawdzenei funkcjonananosci - jakies testy, przejrzec na małych obrazach (na ten moent nie bawić isę jakoś mocno filtrami)
- [ ] następnei jak jużto mamy to sprawdaamy jak działa robienie tego dataset
- [ ] ze zrobionym datasetem robiym trenowanie i chcemy otrzymac normalnie dizałający jmodel
  - [ ] potem testumey czy to tworzy latent itd .

## Do zrobieni 

- [ ] zrobić optymalne filtry, które pozwolą nam na dobranie właściwych próbek
  - [ ] UWAGA: nie będzie wogóle tła, to może powodowac że będzie **na siłę przewidywał etykiety**
    - [ ] ja bym dodał jakiś sposób na znajdywanie tego szumu, żeby też była klasa "bez opisu" - to też jako klasę trzeba dodać
      - [ ] jak cwanie ten szum znajdywać. 
    - [ ] 
- [ ] zrobić narzędzia  związane z wizualiacją / analiza 
  - [ ] co chcemy badać, jakie informacje do tego potrzebujemy, jak to zrobić sprawnie 
  - [ ] to trzeba do przodu wymyślić, oraz trzeba wtedy mieć wszystkie pixele przebadane
    - [ ] te analizy też można grupować jakoś na odpowiednie klasy
      - [ ] jedna klasa do zwykłych metryk - odtwarzanie, jak dobre odtwarzanie itd.
      - [ ] druga klasa do label'i, wtedy dostawalibyśmy grupowanie w przestrzeni itd. 
      - [ ] osobna do badania latentu ?? 
      - [ ] osobna do badania headów i jak one by się miały zachowywać, żeby to dobrze działo 
- [ ] zrobić analize szukajacą błędów
  - [ ] trzeb zbadać same sample czy dobrze odwzorowanie działa
    - [ ] problem 1: możliwośśc enkodowania i dekodowania sampli (czy sam mechanizm dobrze działa na trainie)
    - [ ] problem 2: generalizacja, czy na innych obrazach to dobrze działa. 
  - [ ] trzeba wziąć zdjęcie nie trenowane na tym samplu i zobazcyć jak się zachowuje





# Biblioteka 

## Do poprawy 
Trzeba poprawic uzywanie tych metasapce bo on przeszukuje też modele itd. czy można to rozdzielić, jeśli tak to tutaj powinny byc wywoływane wyłązcnie straetgia datasetow i żeby nei uruchamąło calej paczki bo to mocno spowałania, to można znaczniej szybciej zrobić, poniewaz na ten mometn to dziął wolno i jest niewyodne 

Zapisuje pliki w złym miejsuc (dostłem to  na usera) 


***

## Do implenetacji 

***

### ujednolicenie formatu

#### Nazewnictwo m2aia (na standadrowe pythonwoy syntax funkcji) 
zrezygunejmy ostatecznie z tych elemtnów związanych z m2aia - trzeba dać jendolite nazwy które z get sepctrum bedą kojarzone i będą w formacie pythona (z małej) - strasznie się mi to myli a tam jest kontekst że przenosili funkcje z C++ więc u nich to ujednolica - u nas robi dymy

#### Ujednolicenie configów (jak już cała biblioteka będzie gotowa to bedzie rzeba ujednolicic format configów)
Aktualnie mamy kilka róznych formatów i nie jest to czytelne ani łatwo konfigurowalne. Zrobimy tak, że kazdy ten amanger itd. będzie miał swój get config itd, i mixny będą tylko wywołwały odpowiownednia pętle w dól albo coś w tym stylu, żeby zebrać cały ten config w jeden plik. Wtedy będzie można to prościej odtworzyc - 
**IDEA**
każdy moduł swój własny config ogarnia i swój wlańsy setup z tego cofigu żeby konflitków nei ył 

***

### Dokumentacja 

#### Samouczki 

##### Datasety
- jaka jest struktura oraz idea za tymi datasetami
- w jaki sposób to pobierać merogwac itd. - jak używanie wygląa
- filtrowanie
  - w jaki sposób sensownei filtrować te wyniki, żeby móc sobie swój przypadek wytrenować, 
  - jak dokładnie te filtry działają. 

format tego najlepiej jakby był taki (wszystko pisz po angielskui w końcowym pliku, ja piszę tak, poniewaz mi jest łatwiej) (<Library_name> nie wstawiaj, będzie trzeba dać jakaś fajną nazwę na tą bibliotekę)

```md
# <Library_name> datasets configuration tutorial / documentation
#TODO - co bedzie tlumazcył strutkruę tego oraz jak są te informacje pobierane filtrowane, jak jest to zarządzane 

## usage

### Run commands
#TODO - tutaj taki quick tutorail jak to uruchomić, czyli mamy przykładowy run tego, gdzie są ustawieina i jakieś najszybsze zmiany które trzeba wprowadzić, żeby otrzymac to co się potrzebujes i te stage porozbijać tak
#TODO - czyli tutaj skupiamy sie na uruchmieniu, tłumazcy jak korzystac z CLI (można nawet to nazwac CLI tutorial) 

#### Step 1: Catalog   (nazwa jest kretyńska, ja bym to zmienił Query albo coś takeigo bo tutaj jest zebranie tych danych) 

#### Step 2: Download 

#### Step 3: Merge 

### Configuration explanation
#TODO - tutaj wyłuamzcenie w jaki sposób dokładnie zrobić ten plik json żeby to dobrze działało i wytłumacze jak te filtry ustawiac itd. źle to opisąłes więcn ie daje wsakzuje jak to rodzleić ale powinno byc rozdielone za pomocą 

#### ...


## Implementation explamantion 
#TODO - dokąłdnei wyłuamaczyc w jaki sposób to działa "pod maską", żęby każdy mógł to szybko zrozumiec nie musząc patrzec w kod, czyli że co dokladnie wykonuje czego nie implementuje w idea bym napisał o tym jaka jets idea tych datasetów, a potem te kroki, jak sa pobiernae dane itdl 

### Idea 

### Certain bigger part

####  Subparts 
```

#### Opis biblioteki
Większość opisu biblioteki będzie zrobiona automatycznie, dodamy jakies docsy które szybko tłumaczą gdzie szukac jakich informacji, żeby 

Tam wczęsneij są też komendy od ustawiania tych configów - na ten moment nie jest to ważne, póxniej te filtry się ogarnie jak nam limity zwiększąl, bo teraz to nie ma sensu 

> UWAGA:
> trzeba dodać kryterium na ilośc pamięci i najlpiej zroibc tak ze ilośc annotacji do rozmiaru pliku - w ten sposob łatwiej nam będzie tym zarządzać ORAZ najpierw będziem pobierac wszysktie i zachowywać te zdjęcia, poniewaz mamy limit pobrań dziennych. 

Tutaj mergujemy te dane 
```bash 
uv run python assets/scripts/datasets/manage_datasets.py merge \
  --workspace-path workspace \
  --config workspace/datasets/merge_kidney_pilot_v2.json
```







# Eskperetmny 

## test annotacji 
Póxniej to wstawimy do tego docsa (na bazie tego) do tutriali jak korzystać z tych mergów itd 


### ręczne pobranie
Miałem pobrałem z pobrainem plików prze api ponieważ mialen (nie wiedziec dlaczego) wykorzystany pełny limit pobrań - mimo iż żadnego datasetu nie pobrale) 




### teraz
#TODO -  rpzeczytac tłuamzceni - sprawdizć zyw sztyskot jasne , nrapwwaic błedy, puscić trening
#TODO - drugi czat ustawic odpoweenido tą drugą baze danych, wtedy pobawić się filtrami w miedzy czasie i porawić skrypt żeby mergowł tyko labelowanei dane + jakiś szum (procent pixeli z labelami albo liczba ??? - jak zaimpletnować ?? - dw oodzielne parametyy i tlko jeden do pdania ??? - domyśłnei bez szumu
#TODO - nasepnei te filty trzeba przejrzeć, żeby to zaimpeltnować poprawnei, to jest z tej drugieog zroibć tak żeby szuakć odpoweidnich datasetów, to powinno działć tak, ze mamy notebooka i jesty nasze api któe pozwla to filtrowc odpiwendio i jak nm sie spodowa ostaetaeczna wersja to eksportuejymdo json'a i potem możemy pobieranei srytpem uruchomic
#TODO - logami trzeba by opdwoednio zarządząc. 

#TODO - trzeba rozibć train set, test set i validation set, poniewz jest mega liba z tym, to powinno tez byc jakoć w configu zpaisyawne względem tego uswaionnego seed'a, poniewaz nie mozna opównywąc miarodajnie tych wyników
- najlepsze modele powinny być braen względem test setu'u 

teraz teraz
#TODO - przenieść ten dataset odpoweidnio do terneingów itd. żebyto ujednolić
#TODO - opoprawic te zbieranei danyhc
#TODO - zrobić pełną analize

