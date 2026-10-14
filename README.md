# Predviđanje cena nekretnina u Beogradu pomoću mašinskog učenja

Projekat predstavlja programski deo diplomskog rada posvećenog predviđanju oglasnih cena stanova u Beogradu. Podaci su prikupljeni sa portala NadjiDom i obrađeni isključivo u akademske svrhe.

## Cilj projekta

Cilj je poređenje pet regresionih metoda: linearne regresije, Random Forest algoritma, XGBoost algoritma, višeslojnog perceptrona (MLP) i CatBoost algoritma. Ciljna promenljiva je ukupna oglasna cena stana izražena u evrima.

## Skup podataka

- 700 sirovih oglasa za prodaju stanova
- 694 oglasa nakon čišćenja
- 20 kolona u konačnom skupu
- prostorni obuhvat: Beograd

Konačni skup sadrži cenu, površinu, broj soba, sprat, ukupan broj spratova, broj kupatila, opštinu, naselje, grejanje, stanje i dodatne karakteristike stana. Kolona `cena_po_m2` koristi se samo u opisnoj analizi. Nije korišćena za obučavanje modela jer je izvedena iz ciljne cene.

Podaci predstavljaju oglasne, a ne konačne kupoprodajne cene. Portal i struktura njegovih stranica mogu se promeniti, pa se uspešno ponovno prikupljanje podataka ne može trajno garantovati.

## Struktura projekta

```text
00_prikupljanje_podataka.py   prikupljanje javno dostupnih oglasa
01_priprema_podataka.py       čišćenje i validacija podataka
02_eksplorativna_analiza.py   tabele i grafikoni opisne analize
03_treniranje_modela.py       obučavanje i poređenje pet modela
data/                         sirovi i konačni skup podataka
rezultati_eda/                rezultati eksplorativne analize
rezultati_modela/             metrike, predikcije, grafikoni i model
```

## Instalacija

Potrebna je instalacija Python-a. U Command Prompt-u, iz glavnog foldera projekta, pokrenuti:

```bash
python -m pip install -r requirements.txt
```

Preporučuje se korišćenje virtuelnog okruženja.

## Pokretanje

Kompletan redosled obrade je:

```bash
python 00_prikupljanje_podataka.py --target-raw 700
python 01_priprema_podataka.py
python 02_eksplorativna_analiza.py
python 03_treniranje_modela.py
```

Pošto su prikupljeni podaci već priloženi, scraper nije potrebno ponovo pokretati da bi se ponovili čišćenje, analiza i modelovanje. Svaka skripta može se pokrenuti i zasebno.

## Eksperimentalna postavka

Podaci su podeljeni na trening skup od 80% i test skup od 20%. Na trening delu sprovedena je petostruka unakrsna validacija. Nedostajuće numeričke vrednosti popunjavaju se medijanom naučenom samo na trening skupu, dok se kategorijske promenljive kodiraju postupkom one-hot encoding. Za linearnu regresiju i MLP primenjuje se i standardizacija ulaznih numeričkih atributa.

Iz modela su isključene kolone `nadjidom_id`, `url` i `cena_po_m2`. Performanse se procenjuju pomoću MAE, RMSE, R² i MAPE metrika. Za ponovljivost eksperimenta korišćen je `random_state=42`.

## Rezultati

Najbolji rezultat prema MAE na izdvojenom test skupu ostvario je XGBoost:

| Model | MAE (€) | RMSE (€) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| XGBoost | 41.975 | 64.472 | 0,832 | 17,21 |
| CatBoost | 42.769 | 64.564 | 0,831 | 17,86 |
| MLP | 50.848 | 78.962 | 0,747 | 21,66 |
| Linearna regresija | 52.405 | 83.266 | 0,719 | 20,24 |
| Random Forest | 56.829 | 97.943 | 0,611 | 24,94 |

Tačne vrednosti i rezultati unakrsne validacije nalaze se u fajlu `rezultati_modela/poredjenje_modela.csv`.

## Metodološke napomene

- Pretprocesiranje je deo svakog modelskog pipeline-a i uči se samo na trening podacima.
- Vrednost 0 kod binarnih osobina znači da karakteristika nije navedena u oglasu; ne dokazuje da ne postoji.
- Sačuvani `.joblib` model treba učitavati samo iz pouzdanog izvora.
- Rezultati opisuju analizirani uzorak oglasa i ne predstavljaju zvaničnu procenu tržišne vrednosti.

## Autor

Programski deo diplomskog rada.
