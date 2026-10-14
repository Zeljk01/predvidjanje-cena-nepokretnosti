"""Analiza grešaka modela: cena po m², precenjivanje i potcenjivanje."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def argumenti() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Detaljna analiza grešaka modela")
    parser.add_argument("--data", type=Path, default=root / "data" / "stanovi_beograd_final.csv")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "rezultati_modela" / "predikcije_test_skup.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "rezultati_analiza_gresaka",
    )
    parser.add_argument("--model", default="XGBoost")
    return parser.parse_args()


def dodaj_atribute(pred: pd.DataFrame, stanovi: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["redni_broj"] = pd.to_numeric(pred["redni_broj"], errors="raise").astype(int)
    if not pred["redni_broj"].between(0, len(stanovi) - 1).all():
        raise ValueError("Predikcije sadrže redni broj koji ne postoji u skupu podataka.")
    atributi = stanovi.loc[pred["redni_broj"]].reset_index(drop=True)
    rezultat = pd.concat([pred.reset_index(drop=True), atributi], axis=1)
    rezultat["stvarna_cena_po_m2"] = rezultat["stvarna_cena_eur"] / rezultat["povrsina_m2"]
    rezultat["predvidjena_cena_po_m2"] = rezultat["predvidjena_cena_eur"] / rezultat["povrsina_m2"]
    rezultat["greska_eur"] = rezultat["predvidjena_cena_eur"] - rezultat["stvarna_cena_eur"]
    rezultat["greska_eur_m2"] = rezultat["predvidjena_cena_po_m2"] - rezultat["stvarna_cena_po_m2"]
    rezultat["apsolutna_greska_eur_m2"] = rezultat["greska_eur_m2"].abs()
    rezultat["apsolutna_procentualna_greska"] = (
        rezultat["greska_eur"].abs() / rezultat["stvarna_cena_eur"] * 100
    )
    rezultat["smer_greske"] = np.where(
        rezultat["greska_eur"] > 0,
        "Precenjen",
        np.where(rezultat["greska_eur"] < 0, "Potcenjen", "Tačno"),
    )
    return rezultat


def metrike_po_modelu(sve: pd.DataFrame) -> pd.DataFrame:
    return (
        sve.groupby("model", as_index=False)
        .agg(
            MAE_EUR_M2=("apsolutna_greska_eur_m2", "mean"),
            RMSE_EUR_M2=("greska_eur_m2", lambda x: float(np.sqrt(np.mean(x**2)))),
            MAPE_CENA_M2_PROCENATA=("apsolutna_procentualna_greska", "mean"),
        )
        .sort_values("MAE_EUR_M2")
    )


def grupna_analiza(df: pd.DataFrame, kolona: str) -> pd.DataFrame:
    return (
        df.groupby(kolona, observed=False)
        .agg(
            broj_instanci=("stvarna_cena_eur", "size"),
            prosecna_stvarna_cena=("stvarna_cena_eur", "mean"),
            prosecna_predvidjena_cena=("predvidjena_cena_eur", "mean"),
            prosecna_potpisana_greska_eur=("greska_eur", "mean"),
            MAE_EUR=("greska_eur", lambda x: float(x.abs().mean())),
            MAE_EUR_M2=("greska_eur_m2", lambda x: float(x.abs().mean())),
            MAPE_PROCENATA=("apsolutna_procentualna_greska", "mean"),
            udeo_precenjenih_proc=("smer_greske", lambda x: float((x == "Precenjen").mean() * 100)),
        )
        .reset_index()
    )


def main() -> None:
    args = argumenti()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stanovi = pd.read_csv(args.data)
    pred = pd.read_csv(args.predictions)
    sve = dodaj_atribute(pred, stanovi)

    po_modelu = metrike_po_modelu(sve)
    po_modelu.to_csv(args.output_dir / "01_metrike_cene_po_m2.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 5.5))
    redosled = po_modelu.sort_values("MAPE_CENA_M2_PROCENATA")
    ax = sns.barplot(
        data=redosled,
        x="model",
        y="MAPE_CENA_M2_PROCENATA",
        color="#4C78A8",
    )
    ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3)
    ax.set(title="MAPE pri predviđanju cene po m²", xlabel="", ylabel="MAPE (%)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "00_mape_cena_m2_po_modelu.png", dpi=220)
    plt.close()

    najbolji = sve[sve["model"] == args.model].copy()
    if najbolji.empty:
        raise ValueError(f"Model '{args.model}' nije pronađen u predikcijama.")

    najbolji["cenovni_segment"] = pd.qcut(
        najbolji["stvarna_cena_eur"],
        q=4,
        labels=["Najjeftinijih 25%", "Niža sredina", "Viša sredina", "Najskupljih 25%"],
    )
    najbolji["segment_povrsine"] = pd.cut(
        najbolji["povrsina_m2"],
        bins=[0, 40, 60, 80, 120, np.inf],
        labels=["do 40 m²", "40–60 m²", "60–80 m²", "80–120 m²", "preko 120 m²"],
        right=False,
    )

    detaljne_kolone = [
        "redni_broj", "nadjidom_id", "url", "opstina", "naselje", "povrsina_m2",
        "broj_soba", "sprat", "stanje", "novogradnja_navedena",
        "stvarna_cena_eur", "predvidjena_cena_eur", "greska_eur",
        "stvarna_cena_po_m2", "predvidjena_cena_po_m2", "greska_eur_m2",
        "apsolutna_procentualna_greska", "smer_greske", "cenovni_segment",
        "segment_povrsine",
    ]
    najbolji[detaljne_kolone].to_csv(
        args.output_dir / "02_xgboost_test_detaljno.csv", index=False, encoding="utf-8-sig"
    )

    prikaz_kolone = [
        "nadjidom_id", "opstina", "naselje", "povrsina_m2", "broj_soba",
        "stvarna_cena_eur", "predvidjena_cena_eur", "greska_eur",
        "stvarna_cena_po_m2", "predvidjena_cena_po_m2",
        "apsolutna_procentualna_greska", "url",
    ]
    najbolji.nlargest(10, "greska_eur")[prikaz_kolone].to_csv(
        args.output_dir / "03_top10_precenjenih.csv", index=False, encoding="utf-8-sig"
    )
    najbolji.nsmallest(10, "greska_eur")[prikaz_kolone].to_csv(
        args.output_dir / "04_top10_potcenjenih.csv", index=False, encoding="utf-8-sig"
    )

    po_opstini = grupna_analiza(najbolji, "opstina")
    po_opstini.to_csv(args.output_dir / "05_greske_po_opstini.csv", index=False, encoding="utf-8-sig")
    po_ceni = grupna_analiza(najbolji, "cenovni_segment")
    po_ceni.to_csv(args.output_dir / "06_greske_po_cenovnom_segmentu.csv", index=False, encoding="utf-8-sig")
    po_povrsini = grupna_analiza(najbolji, "segment_povrsine")
    po_povrsini.to_csv(args.output_dir / "07_greske_po_povrsini.csv", index=False, encoding="utf-8-sig")
    po_novogradnji = grupna_analiza(najbolji, "novogradnja_navedena")
    po_novogradnji.to_csv(args.output_dir / "08_greske_po_novogradnji.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 7))
    minimum = min(najbolji["stvarna_cena_po_m2"].min(), najbolji["predvidjena_cena_po_m2"].min())
    maksimum = max(najbolji["stvarna_cena_po_m2"].max(), najbolji["predvidjena_cena_po_m2"].max())
    sns.scatterplot(
        data=najbolji,
        x="stvarna_cena_po_m2",
        y="predvidjena_cena_po_m2",
        alpha=0.7,
        color="#4C78A8",
    )
    plt.plot([minimum, maksimum], [minimum, maksimum], "r--", linewidth=1)
    plt.title("Stvarna i predviđena cena po m² – XGBoost")
    plt.xlabel("Stvarna cena po m² (€/m²)")
    plt.ylabel("Predviđena cena po m² (€/m²)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "00_stvarna_i_predvidjena_cena_m2.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    sns.histplot(najbolji["greska_eur"], bins=30, kde=True, color="#4C78A8")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Raspodela grešaka XGBoost modela")
    plt.xlabel("Predviđena cena − stvarna cena (EUR)")
    plt.ylabel("Broj stanova")
    plt.tight_layout()
    plt.savefig(args.output_dir / "09_raspodela_gresaka.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=najbolji,
        x="stvarna_cena_eur",
        y="greska_eur",
        hue="smer_greske",
        palette={"Precenjen": "#E45756", "Potcenjen": "#4C78A8", "Tačno": "#54A24B"},
        alpha=0.75,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Greška modela u odnosu na stvarnu cenu")
    plt.xlabel("Stvarna oglasna cena (EUR)")
    plt.ylabel("Predviđena cena − stvarna cena (EUR)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "10_greska_prema_stvarnoj_ceni.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=najbolji, x="cenovni_segment", y="greska_eur", showfliers=False, color="#72B7B2")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Greške prema cenovnom segmentu")
    plt.xlabel("")
    plt.ylabel("Predviđena cena − stvarna cena (EUR)")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(args.output_dir / "11_greske_po_cenovnom_segmentu.png", dpi=220)
    plt.close()

    mape_m2 = float(najbolji["apsolutna_procentualna_greska"].mean())
    mae_m2 = float(najbolji["apsolutna_greska_eur_m2"].mean())
    broj_precenjenih = int((najbolji["smer_greske"] == "Precenjen").sum())
    broj_potcenjenih = int((najbolji["smer_greske"] == "Potcenjen").sum())
    najgori_cenovni = po_ceni.loc[po_ceni["MAPE_PROCENATA"].idxmax()]
    najbolji_cenovni = po_ceni.loc[po_ceni["MAPE_PROCENATA"].idxmin()]
    validne_opstine = po_opstini[po_opstini["broj_instanci"] >= 5]
    najgora_opstina = validne_opstine.loc[validne_opstine["MAPE_PROCENATA"].idxmax()]

    tekst = f"""# Analiza grešaka XGBoost modela

- Broj test instanci: {len(najbolji)}
- MAPE za cenu po m²: {mape_m2:.2f}%
- MAE za cenu po m²: {mae_m2:.2f} €/m²
- Broj precenjenih stanova: {broj_precenjenih}
- Broj potcenjenih stanova: {broj_potcenjenih}
- Segment sa najvećim MAPE: {najgori_cenovni['cenovni_segment']} ({najgori_cenovni['MAPE_PROCENATA']:.2f}%)
- Segment sa najmanjim MAPE: {najbolji_cenovni['cenovni_segment']} ({najbolji_cenovni['MAPE_PROCENATA']:.2f}%)
- Najveći MAPE među opštinama sa najmanje 5 test instanci: {najgora_opstina['opstina']} ({najgora_opstina['MAPE_PROCENATA']:.2f}%)

## Metodološka napomena

MAPE ukupne cene i MAPE cene po m² imaju jednaku vrednost kada se stvarna i
predviđena ukupna cena dele istom površinom. Zbog toga je dodatno prikazan MAE
izražen u €/m², koji neposredno pokazuje prosečno odstupanje po kvadratnom metru.

Pozitivna potpisana greška označava precenjivanje, a negativna potcenjivanje.
Zaključci po grupama moraju se tumačiti uz broj instanci, naročito kod opština
koje su slabo zastupljene u test skupu.
"""
    (args.output_dir / "12_rezime_analize_gresaka.md").write_text(tekst, encoding="utf-8")

    print(tekst)
    print(f"Rezultati su sačuvani u: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
