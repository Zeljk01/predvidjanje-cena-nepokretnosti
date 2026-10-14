# Rezime eksperimenta

U eksperimentu je korišćeno 694 oglasa za prodaju stanova u Beogradu. Skup je
podeljen na 555 instanci za obučavanje (80%) i 139 instanci za završno testiranje
(20%), uz `random_state=42`. Na trening delu sprovedena je petostruka unakrsna
validacija. Modeli su poređeni pomoću metrika MAE, RMSE, R² i MAPE.

| Model | MAE (EUR) | RMSE (EUR) | R² | MAPE |
|---|---:|---:|---:|---:|
| XGBoost | 41.592 | 63.856 | 0,835 | 17,11% |
| CatBoost | 42.769 | 64.564 | 0,831 | 17,86% |
| MLP | 50.848 | 78.962 | 0,747 | 21,66% |
| Linearna regresija | 52.405 | 83.266 | 0,719 | 20,24% |
| Random Forest | 56.811 | 97.936 | 0,611 | 24,93% |

Najbolje rezultate na izdvojenom test skupu ostvario je XGBoost. Njegova
prosečna apsolutna greška iznosila je približno 41.592 EUR, dok koeficijent
determinacije od 0,835 pokazuje da je model objasnio oko 83,5% varijabilnosti
oglasnih cena u test skupu. CatBoost je ostvario vrlo sličan rezultat, dok su
MLP i linearna regresija bili slabiji. Najnižu tačnost u ovoj postavci ostvario
je Random Forest.

Rezultat unakrsne validacije potvrđuje isti poredak vodećih modela. XGBoost je
na trening podelama ostvario prosečan MAE od približno 50.658 EUR i prosečan R²
od 0,833. CatBoost je ostvario prosečan MAE od 52.813 EUR i prosečan R² od
0,806. Razlike između test rezultata i proseka unakrsne validacije očekivane su
zbog relativno malog skupa i različitog sastava pojedinačnih podela.

Permutaciona analiza značaja atributa kod XGBoost modela pokazala je da najveći
doprinos predviđanju imaju površina stana, opština i naselje. Ovaj nalaz je u
skladu sa prirodom problema, jer ukupna cena neposredno zavisi od veličine
nekretnine, dok lokacija predstavlja jedan od ključnih činilaca tržišne
vrednosti. Značaj atributa ne treba tumačiti kao dokaz uzročno-posledične veze,
već kao meru njihovog doprinosa prediktivnoj tačnosti konkretnog modela.

Kolone `nadjidom_id` i `url` nisu korišćene kao prediktori. Kolona
`cena_po_m2` takođe je isključena zato što je izračunata iz ciljne promenljive
`cena_eur`; njeno uključivanje predstavljalo bi curenje podataka i dalo
nerealno visoke rezultate. Imputacija nedostajućih vrednosti, kodiranje
kategorijskih promenljivih i standardizacija izvršeni su unutar cevovoda i
naučeni isključivo na trening podacima.

Dobijene vrednosti odnose se na ovu konkretnu podelu podataka i izabrane
hiperparametre. Pošto skup sadrži oglasne, a ne realizovane kupoprodajne cene,
rezultati predstavljaju procenu vrednosti ponude na analiziranom tržištu.
