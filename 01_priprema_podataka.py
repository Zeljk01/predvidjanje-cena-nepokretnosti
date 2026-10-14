"""Čišćenje skupa oglasa za predviđanje cena stanova u Beogradu.

Primer:
    python 01_priprema_podataka.py nadjidom_beograd_raw.csv

Skripta ne menja ulazni fajl. Rezultat je stanovi_beograd_final.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BINARY_RENAME = {
    "novogradnja": "novogradnja_navedena",
    "uknjizen": "uknjizen_navedeno",
    "lift": "lift_naveden",
    "terasa": "terasa_navedena",
    "parking_garaza": "parking_garaza_navedeni",
    "klima": "klima_navedena",
    "podrum": "podrum_naveden",
}

# Ispravke samo tamo gde naziv oglasa nedvosmisleno navodi opštinu.
OPSTINA_BY_ID = {
    2304285: "Zvezdara",
    2304284: "Zvezdara",
    2259814: "Obrenovac",
    2066545: "Grocka",
}

FINAL_COLUMNS = [
    "nadjidom_id",
    "url",
    "cena_eur",
    "povrsina_m2",
    "broj_soba",
    "sprat",
    "ukupno_spratova",
    "broj_kupatila",
    "opstina",
    "naselje",
    "grejanje",
    "stanje",
    "novogradnja_navedena",
    "uknjizen_navedeno",
    "lift_naveden",
    "terasa_navedena",
    "parking_garaza_navedeni",
    "klima_navedena",
    "podrum_naveden",
    "cena_po_m2",
]


def clean(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(input_path)
    initial = len(df)

    df = df.drop_duplicates(subset=["nadjidom_id"], keep="last").copy()
    removed_duplicates = initial - len(df)

    numeric_columns = [
        "nadjidom_id", "cena_eur", "povrsina_m2", "broj_soba", "sprat",
        "ukupno_spratova", "broj_kupatila"
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Portal je u rezultatima za Beograd vratio četiri oglasa sa Zlatibora.
    outside_belgrade = df["naslov"].fillna("").str.contains("Zlatibor", case=False, regex=False)
    removed_outside = int(outside_belgrade.sum())
    df = df.loc[~outside_belgrade].copy()

    df["opstina"] = df.apply(
        lambda row: OPSTINA_BY_ID.get(int(row["nadjidom_id"]), row["opstina"])
        if pd.notna(row["nadjidom_id"]) else row["opstina"],
        axis=1,
    )
    df["opstina"] = df["opstina"].replace("Stan", pd.NA).fillna("Nepoznato")
    df["naselje"] = df["naselje"].replace({"Stan": pd.NA, "Prodaja": pd.NA}).fillna("Nepoznato")

    # Uklanjaju se samo očigledno nemogući zapisi.
    valid_ranges = df["cena_eur"].between(10_000, 3_000_000) & df["povrsina_m2"].between(10, 500)
    removed_invalid_ranges = int((~valid_ranges).sum())
    df = df.loc[valid_ranges].copy()

    # Bez sobnosti ostaje samo jedan oglas; vrednost se ne može pouzdano zaključiti.
    missing_rooms = df["broj_soba"].isna()
    removed_missing_rooms = int(missing_rooms.sum())
    df = df.loc[~missing_rooms].copy()

    for old, new in BINARY_RENAME.items():
        # 1 znači da je osobina navedena u oglasu; 0 da nije navedena.
        df[new] = pd.to_numeric(df[old], errors="coerce").fillna(0).clip(0, 1).astype("int8")

    for column in ["opstina", "naselje", "grejanje", "stanje"]:
        df[column] = df[column].fillna("Nepoznato").astype(str).str.strip()

    df["cena_po_m2"] = (df["cena_eur"] / df["povrsina_m2"]).round(2)
    df["nadjidom_id"] = df["nadjidom_id"].astype("int64")
    df = df[FINAL_COLUMNS].sort_values("nadjidom_id").reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    stats = {
        "pocetni_broj": initial,
        "uklonjeni_duplikati": removed_duplicates,
        "uklonjeni_van_beograda": removed_outside,
        "uklonjeni_nevalidni_rasponi": removed_invalid_ranges,
        "uklonjeni_bez_sobnosti": removed_missing_rooms,
        "konacni_broj": len(df),
    }
    return df, stats


def validate(df: pd.DataFrame) -> None:
    assert not df["nadjidom_id"].duplicated().any(), "Postoje duplirani ID-jevi."
    assert not df["url"].duplicated().any(), "Postoje duplirani URL-ovi."
    assert df["cena_eur"].between(10_000, 3_000_000).all(), "Cena van dozvoljenog raspona."
    assert df["povrsina_m2"].between(10, 500).all(), "Površina van dozvoljenog raspona."
    assert not df[["cena_eur", "povrsina_m2", "broj_soba", "opstina", "naselje"]].isna().any().any()
    assert "Stan" not in set(df["opstina"]), "Pogrešna opština 'Stan' je ostala u skupu."
    for column in BINARY_RENAME.values():
        assert set(df[column].unique()).issubset({0, 1}), f"{column} nije binarna kolona."


def write_report(df: pd.DataFrame, stats: dict[str, int], report_path: Path) -> None:
    missing = (df.isna().mean() * 100).round(1)
    lines = [
        "# Izveštaj o čišćenju skupa podataka",
        "",
        f"- Početni broj oglasa: {stats['pocetni_broj']}",
        f"- Uklonjeni duplikati: {stats['uklonjeni_duplikati']}",
        f"- Uklonjeni oglasi van Beograda: {stats['uklonjeni_van_beograda']}",
        f"- Uklonjeni zapisi sa nevalidnom cenom/površinom: {stats['uklonjeni_nevalidni_rasponi']}",
        f"- Uklonjeni oglasi bez broja soba: {stats['uklonjeni_bez_sobnosti']}",
        f"- Konačni broj instanci: {stats['konacni_broj']}",
        "",
        "## Važne metodološke napomene",
        "",
        "- Cena po m² je zadržana samo za opisnu analizu i ne sme biti ulaz u model koji predviđa ukupnu cenu.",
        "- Kod binarnih kolona 1 znači da je osobina navedena u oglasu, a 0 da nije navedena; 0 ne dokazuje da osobina ne postoji.",
        "- Nedostajuće numeričke vrednosti nisu popunjene u ovom koraku. Imputacija se uči samo na trening skupu.",
        "- Ekstremno skupi stanovi nisu automatski uklonjeni jer mogu predstavljati validne luksuzne nekretnine.",
        "",
        "## Preostale nedostajuće vrednosti",
        "",
    ]
    lines.extend(f"- {column}: {missing[column]:.1f}%" for column in df.columns if missing[column] > 0)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Priprema konačnog skupa stanova u Beogradu")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=project_dir / "data" / "nadjidom_beograd_raw.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "stanovi_beograd_final.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "izvestaj_o_ciscenju.md",
    )
    args = parser.parse_args()

    final_df, stats = clean(args.input, args.output)
    validate(final_df)
    write_report(final_df, stats, args.report)
    print(f"Konačni broj instanci: {len(final_df)}")
    print(f"Sačuvano: {args.output.resolve()}")
    print(f"Izveštaj: {args.report.resolve()}")


if __name__ == "__main__":
    main()
