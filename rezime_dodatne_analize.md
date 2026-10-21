# Rezime dodatne analize

- Model za detaljnu analizu: XGBoost
- Broj test instanci: 139
- MAPE ukupne cene: 17.21%
- MAPE cene po m2: 17.21%
- MAE cene po m2: 528.11 EUR/m2
- RMSE cene po m2: 711.04 EUR/m2
- Precenjeni stanovi: 81 (58.3%)
- Potcenjeni stanovi: 58 (41.7%)
- Cenovni kvartil sa najvecim MAE: Q4_najvise_cene (83389.51 EUR)
- Cenovni kvartil sa najvecim MAPE: Q1_najnize_cene (19.96%)
- Najveci MAPE medju opstinama sa najmanje 5 test oglasa: Zemun (28.55%)

Napomena: MAPE ukupne cene i MAPE cene po m2 imaju istu vrednost zato sto se
stvarna i predvidjena ukupna cena dele istom povrsinom stana. MAE u EUR/m2
daje dodatno prakticno tumacenje velicine greske.
