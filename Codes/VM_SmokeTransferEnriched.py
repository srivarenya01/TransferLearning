"""
Tiny local warm-vs-cold check on the enriched TESTING_STUFF tables.

Uses sklearn MLP (local TensorFlow is broken under NumPy 2). Subsamples rows,
expands only numeric *common* columns, 5 random 80/20 splits. Prints median
NRMSE and Warm−Cold so we can see whether a thicker shared feature space moves
the needle before a full Grace run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "Data"
SOY_PATH = DATA_DIR / "VM_SOYBEAN_TESTING_STUFF.csv"
RICE_PATH = DATA_DIR / "VM_RICE_TESTING_STUFF.csv"

SOY_SAMPLE = 800
RICE_SAMPLE = 800
N_SPLITS = 5
RANDOM_STATE = 42
HIDDEN = (64, 32)

# Original shared weather set before this enrichment (approx.).
BASELINE_COMMON = [
    "Avg_Humidity", "Cumulative_Precipitation", "Flooding_Days",
    "GDD_Avg", "GDD_Cumulative", "GDD_Max", "GDD_Min", "Heavy_Heats",
    "Heavy_Wind_Occurrences", "Precipitation_P", "RelativeHumidity_P",
    "SolarRadiation_P", "Solar_Avg", "Solar_Cumulative", "Solar_Max",
    "Temperature_P", "WindSpeed_P", "days_from_year_start_Planting",
    "lat", "long",
]

# Columns that must never be model inputs (targets, IDs, calendar, harvest outcomes).
BLOCKLIST = {
    "Sl", "Yield", "Year", "GermplasmName_or_Variety", "Maturity_Group",
    "Station_or_Location", "Location", "Variety", "State", "Spacing", "GPS",
    "Planting_Date", "Harvesting_Date", "Emergence_Date",
    "Harvest.moisture.LSU_01.0000136", "Harvest.weight.LSU_01.0000137",
    "Lodging.LSU_01.0000135", "Plant.height.LSU_01.0000132",
    "Plot.stand.near.flood.LSU_01.0000133", "Seedling.vigor.LSU_01.0000131",
    "days_from_year_start_harvest", "latitude", "longitude",
}


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE / mean(y_true) * 100."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.mean(y_true) * 100.0)


def common_numeric(soy: pd.DataFrame, rice: pd.DataFrame) -> list[str]:
    soy_cols = {
        c for c in soy.columns
        if c not in BLOCKLIST and pd.api.types.is_numeric_dtype(soy[c])
    }
    rice_cols = {
        c for c in rice.columns
        if c not in BLOCKLIST and pd.api.types.is_numeric_dtype(rice[c])
    }
    cols = sorted(soy_cols & rice_cols)
    # Drop columns that are entirely missing on either crop after enrichment.
    keep = []
    for col in cols:
        if soy[col].notna().mean() < 0.5 or rice[col].notna().mean() < 0.5:
            continue
        keep.append(col)
    return keep


def matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = df.dropna(subset=["Yield"] + cols).copy()
    frame = frame[frame["Yield"] > 0]
    x = frame[cols].to_numpy(dtype=float)
    y = frame["Yield"].to_numpy(dtype=float)
    return x, y


def fit_mlp(x_train: np.ndarray, y_train: np.ndarray, warm_weights=None, seed: int = 0) -> MLPRegressor:
    """
    Fit an MLP. Warm path: one init step, overwrite with soybean weights, then
    continue with warm_start (no early_stopping — sklearn breaks if toggled mid-fit).
    """
    if warm_weights is None:
        model = MLPRegressor(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model

    coefs, intercepts = warm_weights
    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=1,
        early_stopping=False,
        random_state=seed,
        warm_start=True,
    )
    model.fit(x_train, y_train)
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        if model.coefs_[i].shape != w.shape:
            raise ValueError(f"Warm weight shape mismatch at layer {i}")
        model.coefs_[i] = w.copy()
        model.intercepts_[i] = b.copy()
    model.set_params(max_iter=300)
    model.fit(x_train, y_train)
    return model


def extract_weights(model: MLPRegressor):
    return [c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_]


def eval_feature_set(name: str, cols: list[str], soy_s: pd.DataFrame, rice_s: pd.DataFrame) -> None:
    """Print warm vs cold smoke results for one common-feature set."""
    print(f"\n===== {name}: {len(cols)} features =====")
    x_soy, y_soy = matrix(soy_s, cols)
    x_rice, y_rice = matrix(rice_s, cols)
    print(f"Soy rows: {len(y_soy)} | Rice rows: {len(y_rice)}")

    sx = StandardScaler().fit(x_soy)
    sy = StandardScaler().fit(y_soy.reshape(-1, 1))
    soy_model = fit_mlp(sx.transform(x_soy), sy.transform(y_soy.reshape(-1, 1)).ravel(), seed=RANDOM_STATE)
    warm_w = extract_weights(soy_model)

    cold_scores, warm_scores = [], []
    for split in range(N_SPLITS):
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_rice, y_rice, test_size=0.2, random_state=RANDOM_STATE + split
        )
        rx = StandardScaler().fit(x_tr)
        ry = StandardScaler().fit(y_tr.reshape(-1, 1))
        x_tr_z, x_te_z = rx.transform(x_tr), rx.transform(x_te)
        y_tr_z = ry.transform(y_tr.reshape(-1, 1)).ravel()

        cold = fit_mlp(x_tr_z, y_tr_z, seed=RANDOM_STATE + split)
        warm = fit_mlp(x_tr_z, y_tr_z, warm_weights=warm_w, seed=RANDOM_STATE + split)
        cold_pred = ry.inverse_transform(cold.predict(x_te_z).reshape(-1, 1)).ravel()
        warm_pred = ry.inverse_transform(warm.predict(x_te_z).reshape(-1, 1)).ravel()
        cold_scores.append(nrmse(y_te, cold_pred))
        warm_scores.append(nrmse(y_te, warm_pred))
        print(
            f"  split {split}: Cold={cold_scores[-1]:.2f}  Warm={warm_scores[-1]:.2f}  "
            f"diff(W-C)={warm_scores[-1] - cold_scores[-1]:+.2f}"
        )

    print(
        f"  median Cold={np.median(cold_scores):.2f}  Warm={np.median(warm_scores):.2f}  "
        f"Warm-Cold={np.median(warm_scores) - np.median(cold_scores):+.2f}"
    )


def main() -> None:
    soy = pd.read_csv(SOY_PATH)
    rice = pd.read_csv(RICE_PATH)
    enriched = common_numeric(soy, rice)
    baseline = [c for c in BASELINE_COMMON if c in enriched]
    print(f"Enriched common features ({len(enriched)}):")
    print(enriched)

    soy_s = soy.sample(n=min(SOY_SAMPLE, len(soy)), random_state=RANDOM_STATE)
    rice_s = rice.sample(n=min(RICE_SAMPLE, len(rice)), random_state=RANDOM_STATE)

    eval_feature_set("BASELINE_COMMON (old ~20)", baseline, soy_s, rice_s)
    eval_feature_set("ENRICHED_COMMON (soil+shared)", enriched, soy_s, rice_s)
    print("\n(Tiny local sklearn smoke only — not paper stats.)")


if __name__ == "__main__":
    main()
