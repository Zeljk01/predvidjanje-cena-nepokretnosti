"""Akademsko prikupljanje javnih podataka o stanovima sa NadjiDom.com.

Ne prikuplja imena, telefone, adrese agencija niti druge kontakt podatke.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://www.nadjidom.com"
LISTING = BASE + "/sr/nekretnine/prodaja/stanovi%2BBeograd"
USER_AGENT = "AcademicRealEstateResearch/1.0 (student project; respectful crawling)"
COLS = [
    "url", "nadjidom_id", "naslov", "cena_eur", "povrsina_m2",
    "broj_soba", "sprat", "ukupno_spratova", "broj_kupatila",
    "opstina", "naselje", "grejanje", "tip_objekta", "stanje",
    "novogradnja", "uknjizen", "lift", "terasa", "parking_garaza",
    "klima", "podrum", "datum_azuriranja", "datum_prikupljanja"
]


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "sr-RS,sr;q=0.9,en;q=0.7",
    })
    retry = Retry(
        total=4, connect=4, read=4, backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",), raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def check_robots(s: requests.Session) -> None:
    robots_url = BASE + "/robots.txt"
    response = s.get(robots_url, timeout=30)
    response.raise_for_status()
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    if not parser.can_fetch(USER_AGENT, LISTING):
        raise RuntimeError("robots.txt ne dozvoljava automatizovan pristup ovoj listi.")


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\xa0", " ")).strip()


def numeric(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d[\d\s.,]*", value)
    if not match:
        return None
    x = re.sub(r"\s+", "", match.group()).rstrip(".,")
    if "," in x and "." in x:
        x = x.replace(".", "").replace(",", ".")
    elif "," in x:
        x = x.replace(",", ".")
    elif x.count(".") > 1 or (x.count(".") == 1 and len(x.rsplit(".", 1)[1]) == 3):
        x = x.replace(".", "")
    try:
        return float(x)
    except ValueError:
        return None


def get(s: requests.Session, url: str) -> str:
    response = s.get(url, timeout=40)
    response.raise_for_status()
    if "text/html" not in response.headers.get("Content-Type", ""):
        raise RuntimeError(f"Neocekivan odgovor za {url}")
    return response.text


def listing_url(offset: int) -> str:
    return LISTING if offset == 0 else f"{LISTING}&offset={offset}"


def detail_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    result = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if re.search(r"/sr/details/\d+/", href):
            result.append(urljoin(BASE, href).split("#")[0])
    return list(dict.fromkeys(result))


def attributes(soup: BeautifulSoup) -> dict[str, str]:
    result: dict[str, str] = {}
    heading = next((h for h in soup.find_all(["h2", "h3"]) if clean_text(h.get_text()).lower() == "podaci"), None)
    if heading:
        for node in heading.find_all_next():
            if node.name in {"h2", "h3"} and node is not heading:
                break
            if node.name not in {"li", "tr"}:
                continue
            text = clean_text(node.get_text(" ", strip=True))
            if ":" in text and len(text) < 150:
                key, value = text.split(":", 1)
                result[clean_text(key).lower()] = clean_text(value)
    # Rezervni prolaz ako se HTML struktura promeni.
    if not result:
        for node in soup.select("li, tr"):
            text = clean_text(node.get_text(" ", strip=True))
            if ":" in text and len(text) < 150:
                key, value = text.split(":", 1)
                result[clean_text(key).lower()] = clean_text(value)
    return result


def section_text(soup: BeautifulSoup, title: str) -> str:
    heading = next(
        (h for h in soup.find_all(["h2", "h3"]) if clean_text(h.get_text()).lower() == title.lower()),
        None,
    )
    if not heading:
        return ""
    pieces: list[str] = []
    for node in heading.find_all_next():
        if node.name in {"h2", "h3"} and node is not heading:
            break
        if node.name in {"li", "p"}:
            pieces.append(clean_text(node.get_text(" ", strip=True)))
    return clean_text(" ".join(pieces))


def pick(attrs: dict[str, str], *keys: str) -> str | None:
    for wanted in keys:
        for key, value in attrs.items():
            if wanted == key or wanted in key:
                return value
    return None


def flag(text: str, positives: tuple[str, ...], negatives: tuple[str, ...] = ()) -> int | None:
    lower = text.lower()
    if any(x in lower for x in negatives):
        return 0
    if any(x in lower for x in positives):
        return 1
    return None


def parse_location(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    title = clean_text(soup.title.get_text() if soup.title else "")
    # Tipican naslov: Beograd, Vračar, Kalenić, Stan, Prodaja, ...
    parts = [clean_text(x) for x in title.split(",")]
    if parts and parts[0].lower() == "beograd":
        opstina = parts[1] if len(parts) > 1 else None
        naselje = parts[2] if len(parts) > 2 else None
        return opstina, naselje
    return None, None


def parse_detail(html: str, url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    attrs = attributes(soup)
    page_text = clean_text(soup.get_text(" ", strip=True))
    features = section_text(soup, "Karakteristike")
    opstina, naselje = parse_location(soup)
    id_match = re.search(r"/details/(\d+)/", url)
    h1 = soup.find("h1")

    floor_raw = pick(attrs, "sprat")
    floor = numeric(floor_raw)
    if floor is None and floor_raw:
        mapping = {"prizemlje": 0.0, "visoko prizemlje": 0.5, "suteren": -1.0}
        floor = mapping.get(floor_raw.lower())

    return {
        "url": url,
        "nadjidom_id": id_match.group(1) if id_match else None,
        "naslov": clean_text(h1.get_text(" ", strip=True)) if h1 else clean_text(soup.title.get_text() if soup.title else ""),
        "cena_eur": numeric(pick(attrs, "cena")),
        "povrsina_m2": numeric(pick(attrs, "kvadratura")),
        "broj_soba": numeric(pick(attrs, "broj soba")),
        "sprat": floor,
        "ukupno_spratova": numeric(pick(attrs, "broj spratova")),
        "broj_kupatila": numeric(pick(attrs, "broj kupatila")),
        "opstina": opstina,
        "naselje": naselje,
        "grejanje": pick(attrs, "grejanje"),
        "tip_objekta": pick(attrs, "tip"),
        "stanje": "renoviran" if "renoviran" in page_text.lower() else ("izvorno" if "izvornom stanju" in page_text.lower() else None),
        "novogradnja": flag(features, ("novogradnja",), ("starogradnja", "stara gradnja")),
        "uknjizen": flag(features, ("uknjiženo", "uknjižen", "uknjizeno", "uknjizen"), ("nije uknjižen", "nije uknjizen")),
        "lift": flag(features, ("lift",), ("bez lifta", "nema lift")),
        "terasa": flag(features, ("terasa", "lođa", "lodja"), ("bez terase",)),
        "parking_garaza": flag(features, ("parking", "garaža", "garaza"), ("bez parkinga",)),
        "klima": flag(features, ("klima", "klimatizovan"), ("bez klime",)),
        "podrum": flag(features, ("podrum",), ("bez podruma",)),
        "datum_azuriranja": pick(attrs, "ažuriran", "azuriran"),
        "datum_prikupljanja": datetime.now().strftime("%Y-%m-%d"),
    }


def append(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in COLS})


def clean(raw: Path, output: Path) -> pd.DataFrame:
    df = pd.read_csv(raw)
    df = df.drop_duplicates(subset="nadjidom_id", keep="last")
    for col in ["cena_eur", "povrsina_m2", "broj_soba", "sprat", "ukupno_spratova", "broj_kupatila"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["cena_eur"].between(10_000, 3_000_000)]
    df = df[df["povrsina_m2"].between(10, 500)]
    df["cena_po_m2"] = (df["cena_eur"] / df["povrsina_m2"]).round(2)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return df


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "nadjidom_beograd_raw.csv"
    cleaned = out / "nadjidom_beograd_clean.csv"
    s = session()
    print("Proveravam pravila pristupa sajtu...")
    check_robots(s)

    existing: set[str] = set()
    if raw.exists():
        existing = set(pd.read_csv(raw, usecols=["url"])["url"].dropna())
        print(f"Nastavljam od {len(existing)} vec sacuvanih oglasa.")

    offset = 0
    page = 1
    while len(existing) < args.target_raw and page <= args.max_pages:
        url = listing_url(offset)
        html = get(s, url)
        links = detail_links(html)
        print(f"Stranica {page}: pronadjeno {len(links)} oglasa.")
        if not links:
            (out / "nadjidom_debug.html").write_text(html, encoding="utf-8")
            print("Nema linkova; HTML je sacuvan u data/nadjidom_debug.html.")
            break

        new_on_page = 0
        for link in links:
            if link in existing or len(existing) >= args.target_raw:
                continue
            try:
                row = parse_detail(get(s, link), link)
                append(raw, row)
                existing.add(link)
                new_on_page += 1
                print(f"  [{len(existing)}/{args.target_raw}] {row['naslov'][:65]}")
            except Exception as exc:
                print(f"  Preskocen {link}: {type(exc).__name__}: {exc}")
            time.sleep(random.uniform(args.min_delay, args.max_delay))

        if new_on_page == 0 and all(link in existing for link in links):
            print("Na ovoj stranici nema novih oglasa; prelazim dalje.")
        offset += 20
        page += 1
        time.sleep(random.uniform(args.min_delay, args.max_delay))

    if raw.exists():
        df = clean(raw, cleaned)
        print(f"\nSirovih oglasa: {len(existing)}")
        print(f"Validnih oglasa posle osnovnog ciscenja: {len(df)}")
        print(f"Rezultat: {cleaned.resolve()}")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NadjiDom: prodaja stanova u Beogradu")
    p.add_argument("--target-raw", type=int, default=700)
    p.add_argument("--max-pages", type=int, default=50)
    p.add_argument("--min-delay", type=float, default=1.5)
    p.add_argument("--max-delay", type=float, default=3.0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
