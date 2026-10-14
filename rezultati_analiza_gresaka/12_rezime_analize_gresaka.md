# Analiza grešaka XGBoost modela

- Broj test instanci: 139
- MAPE za cenu po m²: 17.21%
- MAE za cenu po m²: 528.11 €/m²
- Broj precenjenih stanova: 81
- Broj potcenjenih stanova: 58
- Segment sa najvećim MAPE: Najjeftinijih 25% (19.96%)
- Segment sa najmanjim MAPE: Niža sredina (15.01%)
- Najveći MAPE među opštinama sa najmanje 5 test instanci: Zemun (28.55%)

## Metodološka napomena

MAPE ukupne cene i MAPE cene po m² imaju jednaku vrednost kada se stvarna i
predviđena ukupna cena dele istom površinom. Zbog toga je dodatno prikazan MAE
izražen u €/m², koji neposredno pokazuje prosečno odstupanje po kvadratnom metru.

Pozitivna potpisana greška označava precenjivanje, a negativna potcenjivanje.
Zaključci po grupama moraju se tumačiti uz broj instanci, naročito kod opština
koje su slabo zastupljene u test skupu.
