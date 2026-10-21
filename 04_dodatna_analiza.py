"""Dodatna analiza gresaka modela za diplomski rad.

Skripta koristi vec sacuvane predikcije. Ne pokrece novi scraping i ne
obucava modele ponovo. Izracunava greske cene po kvadratnom metru i detaljno
analizira stanove koje najbolji model precenjuje ili potcenjuje.

Pokretanje iz glavnog foldera projekta:
    python 04_dodatna_analiza.py
"""

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
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="MAPE po m2 i analiza precenjenih/potcenjenih stanova."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "data" / "stanovi_beograd_final.csv",
    )
    parser.add_argument(
        "--predikcije",
        type=Path,
        default=project_dir / "rezultati_modela" / "predikcije_test_skup.csv",
    )
    parser.add_argument(
        "--poredjenje-modela",
        type=Path,
        default=project_dir / "rezultati_modela" / "poredjenje_modela.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "rezultati_dodatne_analize",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model za detaljnu analizu. Ako nije naveden, bira se najmanji MAE.",
    )
    return parser.parse_args()


def proveri_kolone(df: pd.DataFrame, kolone: set[str], naziv: str) -> None:
    nedostaju = sorted(kolone - set(df.columns))
    if nedostaju:
        raise ValueError(f"U fajlu {naziv} nedostaju kolone: {nedostaju}")


def ucitaj_i_spoji(args: argparse.Namespace) -> pd.DataFrame:
    podaci = pd.read_csv(args.data).reset_index().rename(columns={"index": "redni_broj"})
    pred = pd.read_csv(args.predikcije)

    proveri_kolone(
        podaci,
        {"redni_broj", "nadjidom_id", "url", "cena_eur", "povrsina_m2", "opstina", "naselje"},
        args.data.name,
    )
    proveri_kolone(
        pred,
        {"redni_broj", "model", "stvarna_cena_eur", "predvidjena_cena_eur"},
        args.predikcije.name,
    )

    opisne = [
        "redni_broj", "nadjidom_id", "url", "povrsina_m2", "broj_soba",
        "opstina", "naselje", "grejanje", "stanje",
        "novogradnja_navedena", "uknjizen_navedeno", "lift_naveden",
        "terasa_navedena", "parking_garaza_navedeni", "klima_navedena",
        "podrum_naveden",
    ]
    spojeno = pred.merge(podaci[opisne], on="redni_broj", how="left", validate="many_to_one")
    if spojeno["povrsina_m2"].isna().any():
        raise ValueError("Neke predikcije nisu povezane sa povrsinom stana.")
    if not np.allclose(
        spojeno["stvarna_cena_eur"],
        spojeno["redni_broj"].map(podaci.set_index("redni_broj")["cena_eur"]),
    ):
        raise ValueError("Stvarne cene u predikcijama i skupu podataka se ne podudaraju.")
    return spojeno


def dodaj_greske(df: pd.DataFrame) -> pd.DataFrame:
    rezultat = df.copy()
    rezultat["greska_eur"] = rezultat["predvidjena_cena_eur"] - rezultat["stvarna_cena_eur"]
    rezultat["apsolutna_greska_eur"] = rezultat["greska_eur"].abs()
    rezultat["procentualna_greska"] = (
        rezultat["apsolutna_greska_eur"] / rezultat["stvarna_cena_eur"] * 100
    )
    rezultat["stvarna_cena_m2"] = rezultat["stvarna_cena_eur"] / rezultat["povrsina_m2"]
    rezultat["predvidjena_cena_m2"] = rezultat["predvidjena_cena_eur"] / rezultat["povrsina_m2"]
    rezultat["greska_cena_m2"] = rezultat["predvidjena_cena_m2"] - rezultat["stvarna_cena_m2"]
    rezultat["apsolutna_greska_m2"] = rezultat["greska_cena_m2"].abs()
    rezultat["smer_greske"] = np.select(
        [rezultat["greska_eur"] > 0, rezultat["greska_eur"] < 0],
        ["precenjeno", "potcenjeno"],
        default="bez_greske",
    )
    return rezultat


def metrike_po_modelu(df: pd.DataFrame) -> pd.DataFrame:
    redovi: list[dict[str, float | int | str]] = []
    for model, deo in df.groupby("model", sort=False):
        redovi.append(
            {
                "model": model,
                "broj_test_instanci": len(deo),
                "MAE_EUR": deo["apsolutna_greska_eur"].mean(),
                "MAPE_PROCENATA": deo["procentualna_greska"].mean(),
                "MAE_EUR_M2": deo["apsolutna_greska_m2"].mean(),
                "MEDIJANA_APSOLUTNE_GRESKE_EUR_M2": deo["apsolutna_greska_m2"].median(),
                "RMSE_EUR_M2": np.sqrt(np.mean(deo["greska_cena_m2"] ** 2)),
                "MAPE_M2_PROCENATA": (
                    deo["apsolutna_greska_m2"] / deo["stvarna_cena_m2"] * 100
                ).mean(),
                "broj_precenjenih": int((deo["smer_greske"] == "precenjeno").sum()),
                "broj_potcenjenih": int((deo["smer_greske"] == "potcenjeno").sum()),
            }
        )
    return pd.DataFrame(redovi).sort_values("MAE_EUR").reset_index(drop=True)


def izaberi_model(args: argparse.Namespace, metrike: pd.DataFrame) -> str:
    if args.model is not None:
        if args.model not in set(metrike["model"]):
            raise ValueError(f"Model '{args.model}' ne postoji u predikcijama.")
        return args.model
    if args.poredjenje_modela.exists():
        poredjenje = pd.read_csv(args.poredjenje_modela)
        proveri_kolone(poredjenje, {"model", "MAE_EUR"}, args.poredjenje_modela.name)
        return str(poredjenje.sort_values("MAE_EUR").iloc[0]["model"])
    return str(metrike.iloc[0]["model"])


def detaljne_tabele(df: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    smer = (
        df.groupby("smer_greske", observed=True)
        .agg(
            broj_instanci=("redni_broj", "size"),
            prosecna_stvarna_cena_eur=("stvarna_cena_eur", "mean"),
            prosecna_predvidjena_cena_eur=("predvidjena_cena_eur", "mean"),
            prosecna_potpisana_greska_eur=("greska_eur", "mean"),
            MAE_EUR=("apsolutna_greska_eur", "mean"),
            MAPE_PROCENATA=("procentualna_greska", "mean"),
            MAE_EUR_M2=("apsolutna_greska_m2", "mean"),
        )
        .reset_index()
    )
    smer["udeo_proc"] = smer["broj_instanci"] / len(df) * 100

    q_labels = ["Q1_najnize_cene", "Q2", "Q3", "Q4_najvise_cene"]
    df = df.copy()
    df["cenovni_kvartil"] = pd.qcut(
        df["stvarna_cena_eur"], q=4, labels=q_labels, duplicates="drop"
    )
    kvartili = (
        df.groupby("cenovni_kvartil", observed=True)
        .agg(
            broj_instanci=("redni_broj", "size"),
            minimalna_cena_eur=("stvarna_cena_eur", "min"),
            maksimalna_cena_eur=("stvarna_cena_eur", "max"),
            MAE_EUR=("apsolutna_greska_eur", "mean"),
            MAPE_PROCENATA=("procentualna_greska", "mean"),
            prosecna_potpisana_greska_eur=("greska_eur", "mean"),
            MAE_EUR_M2=("apsolutna_greska_m2", "mean"),
            broj_precenjenih=("smer_greske", lambda x: int((x == "precenjeno").sum())),
            broj_potcenjenih=("smer_greske", lambda x: int((x == "potcenjeno").sum())),
        )
        .reset_index()
    )

    df["grupa_povrsine"] = pd.cut(
        df["povrsina_m2"],
        bins=[0, 50, 75, 100, np.inf],
        labels=["do_50_m2", "51_do_75_m2", "76_do_100_m2", "preko_100_m2"],
        include_lowest=True,
    )
    povrsine = (
        df.groupby("grupa_povrsine", observed=True)
        .agg(
            broj_instanci=("redni_broj", "size"),
            MAE_EUR=("apsolutna_greska_eur", "mean"),
            MAPE_PROCENATA=("procentualna_greska", "mean"),
            prosecna_potpisana_greska_eur=("greska_eur", "mean"),
            MAE_EUR_M2=("apsolutna_greska_m2", "mean"),
            broj_precenjenih=("smer_greske", lambda x: int((x == "precenjeno").sum())),
            broj_potcenjenih=("smer_greske", lambda x: int((x == "potcenjeno").sum())),
        )
        .reset_index()
    )

    opstine = (
        df.groupby("opstina", observed=True)
        .agg(
            broj_instanci=("redni_broj", "size"),
            prosecna_stvarna_cena_eur=("stvarna_cena_eur", "mean"),
            MAE_EUR=("apsolutna_greska_eur", "mean"),
            MAPE_PROCENATA=("procentualna_greska", "mean"),
            prosecna_potpisana_greska_eur=("greska_eur", "mean"),
            MAE_EUR_M2=("apsolutna_greska_m2", "mean"),
            broj_precenjenih=("smer_greske", lambda x: int((x == "precenjeno").sum())),
            broj_potcenjenih=("smer_greske", lambda x: int((x == "potcenjeno").sum())),
        )
        .sort_values("broj_instanci", ascending=False)
        .reset_index()
    )

    najvise = df.nlargest(15, "apsolutna_greska_eur")[
        [
            "redni_broj", "nadjidom_id", "url", "opstina", "naselje",
            "povrsina_m2", "broj_soba", "stvarna_cena_eur",
            "predvidjena_cena_eur", "greska_eur", "apsolutna_greska_eur",
            "procentualna_greska", "stvarna_cena_m2",
            "predvidjena_cena_m2", "greska_cena_m2", "smer_greske",
        ]
    ]

    tabele = {
        "smer": smer,
        "kvartili": kvartili,
        "povrsine": povrsine,
        "opstine": opstine,
        "najvece": najvise,
        "instance": df,
    }
    for naziv, tabela in tabele.items():
        if naziv == "instance":
            fajl = "predikcije_sa_dodatnom_analizom.csv"
        elif naziv == "smer":
            fajl = "rezime_precenjivanja_potcenjivanja.csv"
        elif naziv == "kvartili":
            fajl = "greske_po_cenovnim_kvartilima.csv"
        elif naziv == "povrsine":
            fajl = "greske_po_povrsini.csv"
        elif naziv == "opstine":
            fajl = "greske_po_opstinama.csv"
        else:
            fajl = "najvece_pojedinacne_greske.csv"
        tabela.to_csv(output / fajl, index=False, encoding="utf-8-sig")
    return tabele


def sacuvaj_grafikone(
    metrike: pd.DataFrame,
    model_df: pd.DataFrame,
    tabele: dict[str, pd.DataFrame],
    model: str,
    output: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    nazivi_modela = {
        "XGBoost": "XGBoost",
        "CatBoost": "CatBoost",
        "MLP": "Неуронска мрежа",
        "Linearna regresija": "Линеарна регресија",
        "Random Forest": "Случајна шума",
    }
    prikaz_metrika = metrike.copy()
    prikaz_metrika["model_prikaz"] = prikaz_metrika["model"].map(nazivi_modela).fillna(prikaz_metrika["model"])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=prikaz_metrika, x="model_prikaz", y="MAPE_M2_PROCENATA", color="#4C78A8")
    ax.set(title="MAPE за цену по квадратном метру", xlabel="", ylabel="MAPE (%)")
    ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(output / "01_mape_cena_m2_po_modelu.png", dpi=220)
    plt.close()

    smer_redosled = ["precenjeno", "potcenjeno", "bez_greske"]
    smer = tabele["smer"].set_index("smer_greske").reindex(smer_redosled).dropna().reset_index()
    smer["smer_prikaz"] = smer["smer_greske"].map(
        {"precenjeno": "Прецењено", "potcenjeno": "Потцењено", "bez_greske": "Без грешке"}
    )
    plt.figure(figsize=(8, 5.5))
    ax = sns.barplot(
        data=smer,
        x="smer_prikaz",
        y="broj_instanci",
        hue="smer_prikaz",
        palette={"Прецењено": "#E15759", "Потцењено": "#4E79A7", "Без грешке": "#59A14F"},
        legend=False,
    )
    ax.set(title=f"Смер грешке – {model}", xlabel="", ylabel="Број станова")
    ax.bar_label(ax.containers[0], padding=3)
    plt.tight_layout()
    plt.savefig(output / "02_broj_precenjenih_i_potcenjenih.png", dpi=220)
    plt.close()

    kvartili = tabele["kvartili"]
    kvartili = kvartili.copy()
    kvartili["kvartil_prikaz"] = kvartili["cenovni_kvartil"].map(
        {
            "Q1_najnize_cene": "Q1 – најниже цене",
            "Q2": "Q2",
            "Q3": "Q3",
            "Q4_najvise_cene": "Q4 – највише цене",
        }
    )
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=kvartili, x="kvartil_prikaz", y="MAE_EUR", color="#F28E2B")
    ax.set(title=f"MAE према ценовном квартилу – {model}", xlabel="Ценовни квартил", ylabel="MAE (EUR)")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)
    plt.tight_layout()
    plt.savefig(output / "03_mae_po_cenovnim_kvartilima.png", dpi=220)
    plt.close()

    opstine = tabele["opstine"].query("broj_instanci >= 5").sort_values("prosecna_potpisana_greska_eur")
    boje = ["#4E79A7" if x < 0 else "#E15759" for x in opstine["prosecna_potpisana_greska_eur"]]
    plt.figure(figsize=(10, 6.5))
    ax = sns.barplot(
        data=opstine,
        x="prosecna_potpisana_greska_eur",
        y="opstina",
        hue="opstina",
        palette=dict(zip(opstine["opstina"], boje)),
        legend=False,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        title=f"Просечна потписана грешка по општини – {model}",
        xlabel="Предвиђена минус стварна цена (EUR)",
        ylabel="",
    )
    plt.tight_layout()
    plt.savefig(output / "04_potpisana_greska_po_opstinama.png", dpi=220)
    plt.close()

    minimum = min(model_df["stvarna_cena_m2"].min(), model_df["predvidjena_cena_m2"].min())
    maksimum = max(model_df["stvarna_cena_m2"].max(), model_df["predvidjena_cena_m2"].max())
    plt.figure(figsize=(7.5, 6.5))
    model_df = model_df.copy()
    model_df["smer_prikaz"] = model_df["smer_greske"].map(
        {"precenjeno": "Прецењено", "potcenjeno": "Потцењено", "bez_greske": "Без грешке"}
    )
    ax = sns.scatterplot(
        data=model_df,
        x="stvarna_cena_m2",
        y="predvidjena_cena_m2",
        hue="smer_prikaz",
        palette={"Прецењено": "#E15759", "Потцењено": "#4E79A7", "Без грешке": "#59A14F"},
        alpha=0.75,
    )
    ax.plot([minimum, maksimum], [minimum, maksimum], "k--", linewidth=1)
    ax.set(
        title=f"Стварна и предвиђена цена по m² – {model}",
        xlabel="Стварна цена (EUR/m²)",
        ylabel="Предвиђена цена (EUR/m²)",
    )
    ax.legend(title="Смер грешке")
    plt.tight_layout()
    plt.savefig(output / "05_stvarna_i_predvidjena_cena_m2.png", dpi=220)
    plt.close()


def napisi_rezime(
    metrike: pd.DataFrame,
    model_df: pd.DataFrame,
    tabele: dict[str, pd.DataFrame],
    model: str,
    output: Path,
) -> None:
    red = metrike.loc[metrike["model"] == model].iloc[0]
    precenjeno = int((model_df["smer_greske"] == "precenjeno").sum())
    potcenjeno = int((model_df["smer_greske"] == "potcenjeno").sum())
    kvartili = tabele["kvartili"]
    najveca_mae = kvartili.loc[kvartili["MAE_EUR"].idxmax()]
    najveci_mape = kvartili.loc[kvartili["MAPE_PROCENATA"].idxmax()]
    dostupne_opstine = tabele["opstine"].query("broj_instanci >= 5")
    najveca_opstina = dostupne_opstine.loc[dostupne_opstine["MAPE_PROCENATA"].idxmax()]

    tekst = f"""# Rezime dodatne analize

- Model za detaljnu analizu: {model}
- Broj test instanci: {len(model_df)}
- MAPE ukupne cene: {red['MAPE_PROCENATA']:.2f}%
- MAPE cene po m2: {red['MAPE_M2_PROCENATA']:.2f}%
- MAE cene po m2: {red['MAE_EUR_M2']:.2f} EUR/m2
- RMSE cene po m2: {red['RMSE_EUR_M2']:.2f} EUR/m2
- Precenjeni stanovi: {precenjeno} ({precenjeno / len(model_df) * 100:.1f}%)
- Potcenjeni stanovi: {potcenjeno} ({potcenjeno / len(model_df) * 100:.1f}%)
- Cenovni kvartil sa najvecim MAE: {najveca_mae['cenovni_kvartil']} ({najveca_mae['MAE_EUR']:.2f} EUR)
- Cenovni kvartil sa najvecim MAPE: {najveci_mape['cenovni_kvartil']} ({najveci_mape['MAPE_PROCENATA']:.2f}%)
- Najveci MAPE medju opstinama sa najmanje 5 test oglasa: {najveca_opstina['opstina']} ({najveca_opstina['MAPE_PROCENATA']:.2f}%)

Napomena: MAPE ukupne cene i MAPE cene po m2 imaju istu vrednost zato sto se
stvarna i predvidjena ukupna cena dele istom povrsinom stana. MAE u EUR/m2
daje dodatno prakticno tumacenje velicine greske.
"""
    (output / "rezime_dodatne_analize.md").write_text(tekst, encoding="utf-8")


def main() -> None:
    args = argumenti()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sve = dodaj_greske(ucitaj_i_spoji(args))
    metrike = metrike_po_modelu(sve)
    model = izaberi_model(args, metrike)
    model_df = sve.loc[sve["model"] == model].copy()

    metrike.to_csv(
        args.output_dir / "metrike_cena_po_m2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tabele = detaljne_tabele(model_df, args.output_dir)
    sacuvaj_grafikone(metrike, model_df, tabele, model, args.output_dir)
    napisi_rezime(metrike, model_df, tabele, model, args.output_dir)

    print(f"Detaljno analiziran model: {model}")
    print(f"Rezultati su sacuvani u: {args.output_dir.resolve()}")
    print(
        metrike[["model", "MAE_EUR_M2", "RMSE_EUR_M2", "MAPE_M2_PROCENATA"]]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
