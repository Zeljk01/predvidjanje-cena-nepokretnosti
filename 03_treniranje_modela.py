"""Obucavanje i poredjenje modela za predvidjanje cena stanova u Beogradu.

Modeli: linearna regresija, Random Forest, XGBoost, MLP i CatBoost.
Skripta sprecava curenje podataka tako sto imputaciju, kodiranje i skaliranje
uci samo na trening podacima. Kolona cena_po_m2 se ne koristi jer je izvedena
iz ciljne promenljive cena_eur.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "Nedostaje xgboost. Pokrenite: python -m pip install xgboost"
    ) from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise SystemExit(
        "Nedostaje catboost. Pokrenite: python -m pip install catboost"
    ) from exc


SEED = 42
TARGET = "cena_eur"
NE_KORISTITI = ["nadjidom_id", "url", "cena_po_m2"]


def argumenti() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poredjenje pet regresionih modela za cene stanova."
    )
    project_dir = Path(__file__).resolve().parent
    podrazumevani_ulaz = project_dir / "data" / "stanovi_beograd_final.csv"
    parser.add_argument("--input", type=Path, default=podrazumevani_ulaz)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "rezultati_modela",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=5)
    return parser.parse_args()


def ucitaj_podatke(putanja: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not putanja.exists():
        raise FileNotFoundError(f"CSV fajl nije pronadjen: {putanja}")
    df = pd.read_csv(putanja)
    obavezne = {TARGET, *NE_KORISTITI}
    nedostaju = sorted(obavezne - set(df.columns))
    if nedostaju:
        raise ValueError(f"Nedostaju ocekivane kolone: {nedostaju}")

    df = df.drop_duplicates(subset=["nadjidom_id"]).copy()
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET, *NE_KORISTITI])
    return X, y


def pretprocesor(X: pd.DataFrame, skaliranje: bool) -> ColumnTransformer:
    numericke = X.select_dtypes(include=np.number).columns.tolist()
    kategorijske = X.select_dtypes(exclude=np.number).columns.tolist()

    numericki_koraci: list[tuple[str, object]] = [
        ("imputacija", SimpleImputer(strategy="median"))
    ]
    if skaliranje:
        numericki_koraci.append(("skaliranje", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numericki_koraci), numericke),
            (
                "cat",
                Pipeline(
                    [
                        ("imputacija", SimpleImputer(strategy="most_frequent")),
                        (
                            "kodiranje",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                kategorijske,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def modeli(X: pd.DataFrame) -> dict[str, object]:
    konfiguracije = {
        "Linearna regresija": (
            LinearRegression(),
            True,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=SEED,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=700,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=SEED,
            ),
            False,
        ),
        "MLP": (
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=SEED,
            ),
            True,
        ),
        "CatBoost": (
            CatBoostRegressor(
                iterations=700,
                depth=6,
                learning_rate=0.03,
                loss_function="RMSE",
                l2_leaf_reg=5,
                verbose=False,
                allow_writing_files=False,
                random_seed=SEED,
                thread_count=-1,
            ),
            False,
        ),
    }

    rezultat = {}
    for naziv, (regresor, skaliranje) in konfiguracije.items():
        osnovni = Pipeline(
            [("priprema", pretprocesor(X, skaliranje)), ("model", regresor)]
        )
        if naziv == "MLP":
            # Neuronska mreza je stabilnija kada se i cilj standardizuje.
            rezultat[naziv] = TransformedTargetRegressor(
                regressor=osnovni, transformer=StandardScaler()
            )
        else:
            # Ostali modeli uce logaritam cene; rezultat se vraca u evre.
            rezultat[naziv] = TransformedTargetRegressor(
                regressor=osnovni,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
    return rezultat


def metrike(y_stvarno: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE_EUR": mean_absolute_error(y_stvarno, y_pred),
        "RMSE_EUR": mean_squared_error(y_stvarno, y_pred) ** 0.5,
        "R2": r2_score(y_stvarno, y_pred),
        "MAPE_PROCENATA": float(
            np.mean(np.abs((np.asarray(y_stvarno) - y_pred) / np.asarray(y_stvarno)))
            * 100
        ),
    }


def sacuvaj_grafikone(
    rezultati: pd.DataFrame,
    predikcije: pd.DataFrame,
    izlaz: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    prikaz = rezultati.melt(
        id_vars="model",
        value_vars=["MAE_EUR", "RMSE_EUR"],
        var_name="metrika",
        value_name="greska_eur",
    )
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=prikaz, x="model", y="greska_eur", hue="metrika")
    ax.set(title="Poređenje grešaka modela", xlabel="", ylabel="Greška (EUR)")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(izlaz / "01_poredjenje_mae_rmse.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=rezultati, x="model", y="R2", color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Koeficijent determinacije po modelu", xlabel="", ylabel="R²")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(izlaz / "02_poredjenje_r2.png", dpi=220)
    plt.close()

    broj = len(rezultati)
    fig, ose = plt.subplots(2, 3, figsize=(15, 10))
    ose = ose.ravel()
    minimum = float(predikcije["stvarna_cena_eur"].min())
    maksimum = float(predikcije["stvarna_cena_eur"].max())
    for i, model in enumerate(rezultati["model"]):
        ax = ose[i]
        deo = predikcije[predikcije["model"] == model]
        ax.scatter(
            deo["stvarna_cena_eur"], deo["predvidjena_cena_eur"],
            alpha=0.65, s=28
        )
        ax.plot([minimum, maksimum], [minimum, maksimum], "r--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Stvarna cena (EUR)")
        ax.set_ylabel("Predviđena cena (EUR)")
    for i in range(broj, len(ose)):
        ose[i].axis("off")
    fig.suptitle("Stvarne i predviđene cene na test skupu", fontsize=15)
    fig.tight_layout()
    fig.savefig(izlaz / "03_stvarne_i_predvidjene_cene.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = argumenti()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = ucitaj_podatke(args.input)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED
    )
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=SEED)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    redovi = []
    sve_predikcije = []
    nauceni_modeli: dict[str, object] = {}
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print(f"Ukupno instanci: {len(X)}")
    print(f"Trening: {len(X_train)} | Test: {len(X_test)}")
    print(f"Broj ulaznih atributa pre kodiranja: {X.shape[1]}\n")

    for naziv, model in modeli(X_train).items():
        print(f"Obucavanje: {naziv}...")
        cv_rez = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        test_m = metrike(y_test, pred)

        redovi.append(
            {
                "model": naziv,
                **test_m,
                "CV_MAE_EUR_PROSEK": -cv_rez["test_mae"].mean(),
                "CV_MAE_EUR_SD": cv_rez["test_mae"].std(),
                "CV_RMSE_EUR_PROSEK": -cv_rez["test_rmse"].mean(),
                "CV_R2_PROSEK": cv_rez["test_r2"].mean(),
                "CV_R2_SD": cv_rez["test_r2"].std(),
            }
        )
        sve_predikcije.append(
            pd.DataFrame(
                {
                    "redni_broj": y_test.index,
                    "model": naziv,
                    "stvarna_cena_eur": y_test.to_numpy(),
                    "predvidjena_cena_eur": pred,
                    "apsolutna_greska_eur": np.abs(y_test.to_numpy() - pred),
                }
            )
        )
        nauceni_modeli[naziv] = model

    rezultati = pd.DataFrame(redovi).sort_values("MAE_EUR").reset_index(drop=True)
    predikcije = pd.concat(sve_predikcije, ignore_index=True)
    rezultati.to_csv(args.output_dir / "poredjenje_modela.csv", index=False)
    predikcije.to_csv(args.output_dir / "predikcije_test_skup.csv", index=False)

    najbolji_naziv = rezultati.loc[0, "model"]
    najbolji = nauceni_modeli[najbolji_naziv]
    perm = permutation_importance(
        najbolji,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=SEED,
        n_jobs=-1,
    )
    znacaj = pd.DataFrame(
        {
            "atribut": X_test.columns,
            "porast_MAE_pri_mesanje": perm.importances_mean,
            "standardna_devijacija": perm.importances_std,
        }
    ).sort_values("porast_MAE_pri_mesanje", ascending=False)
    znacaj.to_csv(args.output_dir / "znacaj_atributa_najboljeg_modela.csv", index=False)

    plt.figure(figsize=(10, 7))
    prikaz_znacaja = znacaj.head(15).sort_values("porast_MAE_pri_mesanje")
    sns.barplot(
        data=prikaz_znacaja,
        x="porast_MAE_pri_mesanje",
        y="atribut",
        color="#59A14F",
    )
    plt.title(f"Značaj atributa – {najbolji_naziv}")
    plt.xlabel("Porast MAE nakon permutacije atributa")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(args.output_dir / "04_znacaj_atributa.png", dpi=220)
    plt.close()

    sacuvaj_grafikone(rezultati, predikcije, args.output_dir)
    joblib.dump(najbolji, args.output_dir / "najbolji_model.joblib")

    metapodaci = {
        "ulazni_fajl": args.input.name,
        "broj_instanci": len(X),
        "broj_trening_instanci": len(X_train),
        "broj_test_instanci": len(X_test),
        "broj_ulaznih_atributa": X.shape[1],
        "ulazni_atributi": X.columns.tolist(),
        "izbacene_kolone": NE_KORISTITI,
        "ciljna_promenljiva": TARGET,
        "test_size": args.test_size,
        "broj_cv_podela": args.cv,
        "random_seed": SEED,
        "transformacija_cilja": (
            "log1p/expm1 za LR, RF, XGBoost i CatBoost; "
            "standardizacija cilja za MLP"
        ),
        "najbolji_model_prema_test_MAE": najbolji_naziv,
    }
    (args.output_dir / "metapodaci_eksperimenta.json").write_text(
        json.dumps(metapodaci, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nRezultati na izdvojenom test skupu:")
    print(
        rezultati[["model", "MAE_EUR", "RMSE_EUR", "R2", "MAPE_PROCENATA"]]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nNajbolji model prema test MAE: {najbolji_naziv}")
    print(f"Rezultati su sacuvani u: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
