"""Eksplorativna analiza cena stanova u Beogradu.

Pokretanje:
    python 02_eksplorativna_analiza.py stanovi_beograd_final.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = "viridis"
NUMERIC = [
    "cena_eur", "povrsina_m2", "broj_soba", "sprat",
    "ukupno_spratova", "broj_kupatila", "cena_po_m2"
]


def save(name: str, output: Path) -> None:
    plt.tight_layout()
    plt.savefig(output / name, dpi=220, bbox_inches="tight")
    plt.close()


def eur_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = lambda x, _: f"{x:,.0f}".replace(",", ".")
    if axis == "y":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))


def analyze(csv_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="notebook")

    # Tabele za rad i proveru rezultata.
    df[NUMERIC].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.to_csv(
        output / "osnovna_statistika.csv", encoding="utf-8-sig"
    )
    by_municipality = df.groupby("opstina").agg(
        broj_oglasa=("cena_eur", "size"),
        prosecna_cena_eur=("cena_eur", "mean"),
        medijalna_cena_eur=("cena_eur", "median"),
        prosecna_cena_po_m2=("cena_po_m2", "mean"),
        medijalna_cena_po_m2=("cena_po_m2", "median"),
        medijalna_povrsina_m2=("povrsina_m2", "median"),
    ).sort_values("broj_oglasa", ascending=False)
    by_municipality.round(2).to_csv(output / "statistika_po_opstinama.csv", encoding="utf-8-sig")
    (df.isna().sum().to_frame("broj_nedostajucih")
       .assign(procenat=lambda x: (x["broj_nedostajucih"] / len(df) * 100).round(2))
       .to_csv(output / "nedostajuce_vrednosti.csv", encoding="utf-8-sig"))

    # 1. Raspodela ukupnih cena; ograničenje prikaza na 99. percentil čuva čitljivost.
    limit_price = df["cena_eur"].quantile(.99)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(df.loc[df.cena_eur <= limit_price, "cena_eur"], bins=35, kde=True, ax=ax, color="#1f77b4")
    ax.axvline(df.cena_eur.median(), color="#d62728", linestyle="--", label="Medijana")
    ax.set(title="Raspodela oglasnih cena stanova", xlabel="Cena (€), prikaz do 99. percentila", ylabel="Broj oglasa")
    eur_axis(ax, "x")
    ax.legend()
    save("01_raspodela_cena.png", output)

    # 2. Raspodela površine.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(df["povrsina_m2"], bins=35, kde=True, ax=ax, color="#2ca02c")
    ax.axvline(df.povrsina_m2.median(), color="#d62728", linestyle="--", label="Medijana")
    ax.set(title="Raspodela površine stanova", xlabel="Površina (m²)", ylabel="Broj oglasa")
    ax.legend()
    save("02_raspodela_povrsine.png", output)

    # 3. Zastupljenost opština.
    counts = df.opstina.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    counts.plot.barh(ax=ax, color=sns.color_palette(PALETTE, len(counts)))
    ax.set(title="Broj oglasa po opštinama", xlabel="Broj oglasa", ylabel="Opština")
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)
    save("03_broj_oglasa_po_opstinama.png", output)

    # 4. Medijalna ukupna cena; medijana je otpornija na luksuzne autlajere.
    med_price = by_municipality.medijalna_cena_eur.sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    med_price.plot.barh(ax=ax, color=sns.color_palette(PALETTE, len(med_price)))
    ax.set(title="Medijalna oglasna cena po opštinama", xlabel="Medijalna cena (€)", ylabel="Opština")
    eur_axis(ax, "x")
    save("04_medijalna_cena_po_opstinama.png", output)

    # 5. Medijalna cena po kvadratnom metru.
    med_m2 = by_municipality.medijalna_cena_po_m2.sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    med_m2.plot.barh(ax=ax, color=sns.color_palette(PALETTE, len(med_m2)))
    ax.set(title="Medijalna cena po m² prema opštini", xlabel="Cena po m² (€)", ylabel="Opština")
    eur_axis(ax, "x")
    save("05_medijalna_cena_m2_po_opstinama.png", output)

    # 6. Površina i ukupna cena, obojeno po opštinama sa najmanje 20 oglasa.
    major = counts[counts >= 20].index
    plot_df = df.assign(opstina_prikaz=df.opstina.where(df.opstina.isin(major), "Ostale"))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.scatterplot(data=plot_df, x="povrsina_m2", y="cena_eur", hue="opstina_prikaz", alpha=.65, s=42, ax=ax)
    ax.set(title="Odnos površine i ukupne cene", xlabel="Površina (m²)", ylabel="Cena (€)")
    eur_axis(ax, "y")
    ax.legend(title="Opština", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save("06_povrsina_i_cena.png", output)

    # 7. Cena po broju soba; prikazuju se česte kategorije.
    rooms = sorted(df.loc[df.broj_soba.between(.5, 6), "broj_soba"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[df.broj_soba.isin(rooms)], x="broj_soba", y="cena_eur", showfliers=False, ax=ax, color="#76b7b2")
    ax.set(title="Oglasna cena prema broju soba", xlabel="Broj soba", ylabel="Cena (€), bez prikaza ekstremnih tačaka")
    eur_axis(ax, "y")
    save("07_cena_prema_broju_soba.png", output)

    # 8. Spearmanova korelacija je primerena nelinearnim i asimetričnim raspodelama.
    corr = df[NUMERIC].corr(method="spearman")
    corr.to_csv(output / "spearman_korelacije.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearmanova korelaciona matrica")
    save("08_korelaciona_matrica.png", output)

    # 9. Nedostajuće vrednosti.
    missing = (df.isna().mean() * 100).sort_values()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    missing.plot.barh(ax=ax, color="#e15759")
    ax.set(title="Nedostajuće vrednosti u konačnom skupu", xlabel="Udeo nedostajućih vrednosti (%)", ylabel="Atribut")
    ax.set_xlim(0, max(30, missing.max() * 1.15))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)
    save("09_nedostajuce_vrednosti.png", output)

    summary = f"""# Rezime eksplorativne analize

- Broj instanci: {len(df)}
- Broj opština: {df.opstina.nunique()}
- Medijalna ukupna cena: {df.cena_eur.median():,.0f} €
- Prosečna ukupna cena: {df.cena_eur.mean():,.0f} €
- Medijalna površina: {df.povrsina_m2.median():.1f} m²
- Medijalna cena po m²: {df.cena_po_m2.median():,.0f} €/m²
- Najzastupljenija opština: {df.opstina.value_counts().idxmax()} ({df.opstina.value_counts().max()} oglasa)
- Najviša medijalna cena po m²: {by_municipality.medijalna_cena_po_m2.idxmax()} ({by_municipality.medijalna_cena_po_m2.max():,.0f} €/m²)
- Najniža medijalna cena po m²: {by_municipality.medijalna_cena_po_m2.idxmin()} ({by_municipality.medijalna_cena_po_m2.min():,.0f} €/m²)

Napomena: cena po m² koristi se samo u opisnoj analizi, ne kao ulaz za model koji predviđa ukupnu cenu.
"""
    (output / "rezime_eda.md").write_text(summary, encoding="utf-8")
    print(f"EDA završena. Rezultati: {output.resolve()}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Eksplorativna analiza stanova u Beogradu")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=project_dir / "data" / "stanovi_beograd_final.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "rezultati_eda",
    )
    args = parser.parse_args()
    analyze(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
