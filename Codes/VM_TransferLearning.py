"""
Soybean-to-rice neural transfer learning.

Tunes a source network on soybean, maps its first-layer weights onto rice,
and compares warm vs cold start on the full rice feature set and on the
features shared with soybean. Every result is reported as RMSE and NRMSE.

Shared features are always in soybean units, so transferred weights see the
units they were trained on. Rice-only features and yield are scaled on each
split's training rows. Warm and cold networks use the same seeds and training
schedule, so the only difference between them is the starting weights.
"""
import os
# One op thread per worker process. Has to be set before NumPy/TensorFlow start their thread pools.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import seaborn as sns
from scipy.stats import ttest_ind
import warnings



# TensorFlow is not fork-safe, so workers are spawned fresh.
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

# Paths
REPO_DIR = Path(__file__).resolve().parent.parent
SOY_DATASET_FILE = str(REPO_DIR / "DATA" / "soybean_with_harvesting_only.csv")
RICE_DATASET_FILE = str(REPO_DIR / "DATA" / "rice_with_harvesting_only.csv")
RESULTS_DIR = str(REPO_DIR / "Results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Seeds
GLOBAL_SEED = 42
REPRESENTATIVE_SEED = 0

# Global Constants
NUM_ITERATIONS = 100
# os.cpu_count() reports every core on a shared node, not the job's allocation.
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count())
YEAR_COLUMN_CANDIDATES = ("Year", "year", "Harvest_Year", "HarvestYear", "Crop_Year", "CropYear")
DATE_COLUMN_CANDIDATES = (
    "Harvesting_Date",
    "Harvesting_DT",
    "Planting_Date",
    "Planting_DT",
    "Emergence_Date",
    "Emergence_DT",
)
MIN_LOYO_TRAIN_ROWS = 5
MIN_LOYO_TEST_ROWS = 1
ID_COLUMNS = ("Sl", "GPS")
# Yield is computed from harvested grain weight corrected to a standard moisture.
HARVEST_OUTCOME_PREFIXES = ("harvest.weight", "harvest.moisture")

# Training schedule shared by warm and cold networks
HEAD_EPOCHS = 10
MAX_FINE_TUNE_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 32

# (name, features, input_dim, initial weights, column indices scaled per split)
Scenario = Tuple[str, np.ndarray, int, Optional[List[np.ndarray]], Optional[np.ndarray]]


def folder_creation() -> None:
    """
    Create the Results/ subfolders every run writes into.
    """
    for folder in ["Cleaned_Data", "Errors", "Models", "Graphs"]:
        os.makedirs(f"{RESULTS_DIR}/{folder}", exist_ok=True)


def jsonable(value):
    """
    Convert numpy/pandas scalars into JSON-serializable Python types.

    Args:
        value: Nested dict, list, or scalar that may contain numpy types.

    Returns:
        A structure that json.dump can write without extra converters.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Union[str, Path], payload: dict) -> None:
    """
    Write a JSON artifact with stable indentation.

    Args:
        path: Destination file.
        payload: JSON-serializable (or numpy-bearing) mapping.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=4)


def persist_feature_inventory(
    common: List[str],
    soy_only: List[str],
    rice_only: List[str],
    soy: pd.DataFrame,
    rice: pd.DataFrame
) -> Dict[str, Union[int, dict, List[str]]]:
    """
    Save which encoded features are shared, soybean-only, and rice-only.

    Args:
        common: Feature names present in both encoded crops, excluding Yield.
        soy_only: Encoded soybean columns with no rice counterpart.
        rice_only: Encoded rice columns with no soybean counterpart.
        soy: Cleaned soybean frame, used for row counts and yield summary.
        rice: Cleaned rice frame, used for row counts and yield summary.

    Returns:
        Inventory dictionary that is also written under Cleaned_Data/.
    """
    inventory = {
        "soy_rows": int(len(soy)),
        "rice_rows": int(len(rice)),
        "n_common_features": len(common),
        "n_soy_only_features": len(soy_only),
        "n_rice_only_features": len(rice_only),
        "soy_yield": soy["Yield"].describe().to_dict(),
        "rice_yield": rice["Yield"].describe().to_dict(),
        "common_features": common,
        "soy_only_features": soy_only,
        "rice_only_features": rice_only,
    }
    save_json(f"{RESULTS_DIR}/Cleaned_Data/feature_inventory_{TIMESTAMP}.json", inventory)
    pd.DataFrame({"feature": common}).to_csv(
        f"{RESULTS_DIR}/Cleaned_Data/common_features_{TIMESTAMP}.csv", index=False
    )
    pd.DataFrame({"feature": soy_only}).to_csv(
        f"{RESULTS_DIR}/Cleaned_Data/soy_only_features_{TIMESTAMP}.csv", index=False
    )
    pd.DataFrame({"feature": rice_only}).to_csv(
        f"{RESULTS_DIR}/Cleaned_Data/rice_only_features_{TIMESTAMP}.csv", index=False
    )
    pd.concat(
        {"Soybean": soy["Yield"].describe(), "Rice": rice["Yield"].describe()},
        axis=1
    ).to_csv(f"{RESULTS_DIR}/Cleaned_Data/yield_description_{TIMESTAMP}.csv")
    return inventory


def resolve_year_series(df: pd.DataFrame, dataset_label: str) -> pd.Series:
    """
    Resolves the crop year used for leave-one-year-out validation.

    Args:
        df: Raw dataset before categorical encoding.
        dataset_label: Human-readable crop or dataset name for error messages.

    Returns:
        A Series of integer years aligned to the input DataFrame index.
    """
    for column in YEAR_COLUMN_CANDIDATES:
        if column in df.columns:
            years = pd.to_numeric(df[column], errors='coerce')
            if years.notna().all():
                return years.astype(int)

    candidate_columns = list(DATE_COLUMN_CANDIDATES)
    candidate_columns.extend([col for col in df.columns if "date" in col.lower() and col not in candidate_columns])

    for column in candidate_columns:
        if column in df.columns:
            years = pd.to_datetime(df[column], errors='coerce').dt.year
            if years.notna().all():
                return years.astype(int)

    raise ValueError(
        f"Could not resolve a complete year series for {dataset_label}. "
        f"Expected one of {YEAR_COLUMN_CANDIDATES} or parseable date columns."
    )


def find_calendar_columns(df: pd.DataFrame) -> List[str]:
    """
    List year and calendar-date columns that must not be model inputs.

    A raw year is out of range for any held-out LOYO year, and date strings
    one-hot encode into per-date columns that act as year labels. Day-of-year
    columns (for example `days_from_year_start_Planting`) are kept because they
    describe season timing without identifying the year.

    Args:
        df: Raw dataset before categorical encoding.

    Returns:
        Column names to drop from the feature set.
    """
    return [
        col for col in df.columns
        if col in YEAR_COLUMN_CANDIDATES
        or col in DATE_COLUMN_CANDIDATES
        or ("date" in col.lower() and not col.lower().startswith("days_from"))
    ]


def find_harvest_outcome_columns(df: pd.DataFrame) -> List[str]:
    """
    List harvest measurements that the yield target is derived from.

    Matching is by name prefix after normalizing separators, so trait-ontology
    suffixes such as `.LSU_01.0000137` and `_` vs `.` spellings are covered.

    Args:
        df: Raw dataset before categorical encoding.

    Returns:
        Column names to drop from the feature set.
    """
    def normalize(name: str) -> str:
        return name.lower().replace("_", ".").replace(" ", ".")

    return [
        col for col in df.columns
        if normalize(col).startswith(HARVEST_OUTCOME_PREFIXES)
    ]


def clean_data(
    path_to_file: str,
    return_years: bool = False,
    dataset_label: str = "dataset"
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Cleans the input CSV by removing ID, calendar, and harvest-outcome columns, handling missing values, and encoding categorical variables.

    The crop year is read before calendar columns are dropped, so LOYO folds
    still work while the year itself never reaches the model.

    Args:
        path_to_file: The system path to the CSV dataset.
        return_years: When True, also returns years aligned to cleaned rows.
        dataset_label: Human-readable crop or dataset name for error messages.

    Returns:
        A cleaned and pre-processed pandas DataFrame, optionally with aligned years.
    """
    raw_df = pd.read_csv(path_to_file).drop(columns=list(ID_COLUMNS), errors='ignore')
    year_series = resolve_year_series(raw_df, dataset_label) if return_years else None
    calendar_columns = find_calendar_columns(raw_df)
    harvest_columns = find_harvest_outcome_columns(raw_df)
    print(f"{dataset_label}: dropping calendar columns {calendar_columns}")
    print(f"{dataset_label}: dropping harvest-outcome columns {harvest_columns}")
    raw_df = raw_df.drop(columns=calendar_columns + harvest_columns)

    yields = pd.to_numeric(raw_df['Yield'], errors='coerce')
    valid_yield = yields > 0
    print(f"{dataset_label}: dropping {int((~valid_yield).sum())} rows with missing or non-positive Yield")
    raw_df = raw_df.loc[valid_yield].assign(Yield=yields[valid_yield])

    df = raw_df.dropna(axis=1, how='all').fillna(raw_df.median(numeric_only=True))
    # Full one-hot (no drop_first) so a shared dummy means the same category in both crops.
    df = pd.get_dummies(df, drop_first=False).dropna()
    file_name = Path(path_to_file).stem + "_cleaned.csv"
    cleaned_data = df.loc[:, (df != df.iloc[0]).any()]
    cleaned_data.to_csv(f"{RESULTS_DIR}/Cleaned_Data/{file_name}", index=False)

    if return_years:
        aligned_years = year_series.loc[cleaned_data.index].to_numpy(dtype=int)
        return cleaned_data, aligned_years

    return cleaned_data


def load_data() -> dict:
    """
    Load both crops, align their shared features, and prepare model inputs.

    Shared features are always expressed in soybean units (a scaler fit on
    soybean rows only), in both the shared-feature and full rice matrices, so
    transferred first-layer weights read the same units they were trained on.
    Rice-only columns are returned unscaled and listed in `rice_only_idx`;
    evaluation code scales them per split, so no held-out rice row ever informs
    a scaler.

    Returns:
        A dictionary of feature matrices, raw targets, the soybean scalers used
        to train the source network, and feature bookkeeping.
    """
    soy_p = SOY_DATASET_FILE
    rice_p = RICE_DATASET_FILE
    print(f"Soybean data: {soy_p}")
    print(f"Rice data:    {rice_p}")

    soy = clean_data(soy_p, dataset_label="Soybean")
    rice, rice_years = clean_data(rice_p, return_years=True, dataset_label="Rice")

    common = sorted(list((set(soy.columns) & set(rice.columns)) - {'Yield'}))
    soy_only = sorted(list(set(soy.columns) - set(rice.columns)))
    rice_only = sorted(list(set(rice.columns) - set(soy.columns)))
    persist_feature_inventory(common, soy_only, rice_only, soy, rice)

    print("\n--- Column Comparison ---")
    print(f"Common Features ({len(common)}): {common}")
    print(f"Soybean Only ({len(soy_only)}): {soy_only[:5]}...")
    print(f"Rice Only ({len(rice_only)}): {rice_only[:5]}...")
    print("------------------------\n")

    print(f"Features Identified: {len(common)} Common | Soy: {len(soy)} rows | Rice: {len(rice)} rows")

    soy_X_raw = soy[common].to_numpy(dtype=float)
    soy_y_raw = soy['Yield'].to_numpy(dtype=float)
    scaler_soy_X = StandardScaler().fit(soy_X_raw)
    scaler_soy_y = StandardScaler().fit(soy_y_raw.reshape(-1, 1))

    rice_features_full = list(rice.drop(columns=['Yield']).columns)
    rice_X_comm = scaler_soy_X.transform(rice[common].to_numpy(dtype=float))

    rice_X_full = rice[rice_features_full].to_numpy(dtype=float)
    common_idx = np.array([rice_features_full.index(col) for col in common], dtype=int)
    rice_X_full[:, common_idx] = rice_X_comm
    rice_only_idx = np.setdiff1d(np.arange(len(rice_features_full)), common_idx)

    return {
        'soy_X_raw': soy_X_raw,
        'soy_y_raw': soy_y_raw,
        'soy_X': scaler_soy_X.transform(soy_X_raw),
        'soy_y_z': scaler_soy_y.transform(soy_y_raw.reshape(-1, 1)).flatten(),
        'rice_X_full': rice_X_full,
        'rice_only_idx': rice_only_idx,
        'rice_X_comm': rice_X_comm,
        'rice_y_raw': rice['Yield'].to_numpy(dtype=float),
        'rice_years': rice_years,
        'input_dim_soy': len(common),
        'rice_features_full': rice_features_full,
        'common_features': common,
        'soy_only_features': soy_only,
        'rice_only_features': rice_only,
        'n_common_features': len(common),
        'soy_rows': int(len(soy)),
        'rice_rows': int(len(rice)),
    }


def prepare_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_raw: np.ndarray,
    fit_columns: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize one split using statistics from its training rows only.

    Args:
        X_train: Training features.
        X_test: Evaluation features.
        y_train_raw: Training yield in original units.
        fit_columns: Column indices to standardize with this split's training
            rows. Other columns are already on a fixed scale (shared features
            in soybean units) and are left untouched. None scales nothing.

    Returns:
        Scaled training features, scaled test features, standardized training
        targets, and the target scaler for inverting predictions.
    """
    if fit_columns is not None and len(fit_columns) > 0:
        X_train = np.array(X_train, dtype=float, copy=True)
        X_test = np.array(X_test, dtype=float, copy=True)
        x_scaler = StandardScaler().fit(X_train[:, fit_columns])
        X_train[:, fit_columns] = x_scaler.transform(X_train[:, fit_columns])
        X_test[:, fit_columns] = x_scaler.transform(X_test[:, fit_columns])
    y_scaler = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_z = y_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
    return X_train, X_test, y_train_z, y_scaler


def build_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    params: dict,
    weights: Optional[List[np.ndarray]] = None,
    seed: Optional[int] = None
):
    """
    Construct and train a network with the schedule shared by every scenario.

    Phase 1 trains only the output layer for HEAD_EPOCHS at the tuned learning
    rate. Phase 2 unfreezes all layers at half that rate with early stopping.
    Warm and cold runs follow the same steps; they differ only in whether
    `weights` seeds the network.

    Args:
        X_train: Training features, already scaled.
        y_train: Standardized training targets.
        input_dim: Number of input features.
        params: Dictionary containing 'n_layers', 'lr', and units per layer.
        weights: Optional pre-trained weights for transfer learning.
        seed: Seeds Python, NumPy, and TensorFlow for initialization, shuffling,
            and the validation split. Runs are repeatable per process; threaded
            Optuna trials can still interleave.

    Returns:
        A trained Keras Model object.
    """

    import tensorflow as tf

    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for i in range(params['n_layers']):
        model.add(tf.keras.layers.Dense(params[f'n_units_l{i}'], activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    if weights is not None:
        model.set_weights(weights)

    # Phase 1: output layer only
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')
    model.fit(X_train, y_train, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # Phase 2: whole network at half the learning rate
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr'] / 2), loss='mse')
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )
    model.fit(
        X_train, y_train, validation_split=0.2, epochs=MAX_FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE, verbose=0, callbacks=[es]
    )
    return model


def calculate_error_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    """
    Calculates raw RMSE and normalized RMSE percentage for yield predictions.

    Args:
        y_true_raw: Observed yield values in original units.
        y_pred_raw: Predicted yield values in original units.

    Returns:
        Dictionary with RMSE and NRMSE percentage.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    mean_yield = float(np.mean(y_true_raw))
    nrmse = np.nan if np.isclose(mean_yield, 0.0) else (rmse / mean_yield) * 100
    return {"RMSE": rmse, "NRMSE_Percent": float(nrmse)}


def fit_and_score(
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    input_dim: int,
    params: dict,
    weights: Optional[List[np.ndarray]],
    fit_columns: Optional[np.ndarray],
    seed: int
):
    """
    Scale one split, train on it, and score predictions in original yield units.

    Args:
        X_train: Unscaled (or fixed-scale) training features.
        y_train_raw: Training yield in original units.
        X_test: Evaluation features on the same footing as X_train.
        y_test_raw: Evaluation yield in original units.
        input_dim: Number of input features.
        params: Hyperparameters selected by Optuna.
        weights: Optional transfer weights.
        fit_columns: Column indices to standardize on this split's training rows.
        seed: Seed passed to build_and_train.

    Returns:
        Tuple of (metrics dict, trained model, predictions in original units).
    """
    Xt, Xv, yt_z, y_scaler = prepare_split(X_train, X_test, y_train_raw, fit_columns)
    model = build_and_train(Xt, yt_z, input_dim, params, weights, seed=seed)
    pred_z = model.predict(Xv, verbose=0).reshape(-1, 1)
    pred_raw = y_scaler.inverse_transform(pred_z).flatten()
    return calculate_error_metrics(y_test_raw, pred_raw), model, pred_raw


def worker_task(
    seed: int,
    X: np.ndarray,
    y_raw: np.ndarray,
    dim: int,
    params: dict,
    weights: Optional[List[np.ndarray]],
    fit_columns: Optional[np.ndarray]
) -> Dict[str, Union[int, float, str]]:
    """
    Evaluate one random 80/20 split and return both RMSE and NRMSE.

    Args:
        seed: Random state for the split and for network initialization.
        X: Feature matrix for the scenario.
        y_raw: Observed yield in original units.
        dim: Number of input features.
        params: Hyperparameters selected by Optuna.
        weights: Optional transfer weights.
        fit_columns: Column indices to standardize on the training rows.

    Returns:
        Iteration index, RMSE, NRMSE percentage, and an error message. Failed
        splits return NaN metrics with the exception text.
    """
    try:
        Xt, Xv, yt, yv = train_test_split(X, y_raw, test_size=0.2, random_state=seed)
        metrics, _, _ = fit_and_score(Xt, yt, Xv, yv, dim, params, weights, fit_columns, seed)
        return {"Iteration": int(seed), **metrics, "Error": ""}
    except Exception as exc:
        return {"Iteration": int(seed), "RMSE": np.nan, "NRMSE_Percent": np.nan, "Error": str(exc)}


def evaluate_source_holdout(
    data: Dict,
    params: dict,
    seed: int = REPRESENTATIVE_SEED
) -> Dict[str, Union[int, float]]:
    """
    Score the soybean source network on an 80/20 holdout before full-set training.

    The transfer weights themselves still come from a later fit on all soybean
    rows. This holdout is only a reported source-domain number for the paper.

    Args:
        data: Loaded data dictionary with unscaled soybean features and targets.
        params: Hyperparameters selected by Optuna.
        seed: Random state for the soybean holdout split and initialization.

    Returns:
        RMSE and NRMSE on the soybean validation fold, plus fold size.
    """
    Xt, Xv, yt, yv = train_test_split(
        data['soy_X_raw'], data['soy_y_raw'], test_size=0.2, random_state=seed
    )
    metrics, _, _ = fit_and_score(
        Xt, yt, Xv, yv, data['input_dim_soy'], params, None,
        fit_columns=np.arange(data['input_dim_soy']), seed=seed
    )
    metrics["Iteration"] = int(seed)
    metrics["Train_Rows"] = int(len(Xt))
    metrics["Test_Rows"] = int(len(Xv))
    save_json(f"{RESULTS_DIR}/Errors/source_holdout_{TIMESTAMP}.json", {
        "crop": "soybean",
        "split": "random_80_20",
        **metrics,
    })
    return metrics


def save_trained_model(model, stem: str) -> str:
    """
    Persist a Keras model next to the Optuna parameter dump.

    Prefers the native `.keras` format and falls back to HDF5 when the
    installed TensorFlow build does not accept it.

    Args:
        model: Trained Keras model.
        stem: Filename stem without extension.

    Returns:
        Path of the file that was written.
    """
    keras_path = f"{RESULTS_DIR}/Models/{stem}_{TIMESTAMP}.keras"
    hdf5_path = f"{RESULTS_DIR}/Models/{stem}_{TIMESTAMP}.h5"
    try:
        model.save(keras_path)
        return keras_path
    except Exception:
        model.save(hdf5_path)
        return hdf5_path


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str
) -> None:
    """
    Save a measured-vs-predicted scatter with a 1:1 reference line.

    Args:
        y_true: Observed yield in original units.
        y_pred: Predicted yield in original units.
        title: Figure title.
        filename: Destination PNG name under Results/Graphs.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="black")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Measured Yield")
    plt.ylabel("Predicted Yield")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close()


def run_representative_diagnostics(
    data: Dict,
    scenarios: List[Scenario],
    best_params: Dict[str, Union[int, float]]
) -> None:
    """
    Refit each rice scenario on one 80/20 split for scatter plots and saved models.

    The 100-iteration workers do not return Keras objects. This seed-0 refit is a
    diagnostic companion, not the best model from those 100 splits.

    Args:
        data: Loaded rice features and raw targets.
        scenarios: Named transfer scenarios from the main loop.
        best_params: Hyperparameters selected by Optuna.
    """
    print("\nSaving representative rice diagnostics (seed "
          f"{REPRESENTATIVE_SEED})...")
    for name, X, dim, weights, fit_columns in scenarios:
        Xt, Xv, yt, yv = train_test_split(
            X, data['rice_y_raw'], test_size=0.2, random_state=REPRESENTATIVE_SEED
        )
        _, model, pred_raw = fit_and_score(
            Xt, yt, Xv, yv, dim, best_params, weights, fit_columns, REPRESENTATIVE_SEED
        )
        plot_predicted_vs_actual(
            yv,
            pred_raw,
            title=f"Predicted vs Measured Yield ({name})",
            filename=f"NN_Scatter_{name}_{TIMESTAMP}.png",
        )
        saved = save_trained_model(model, f"rice_{name.lower()}")
        print(f"  {name}: scatter + model -> {saved}")


def evaluate_loyo_fold(
    iteration: int,
    scenario_name: str,
    left_out_year: int,
    X: np.ndarray,
    y_raw: np.ndarray,
    years: np.ndarray,
    input_dim: int,
    params: dict,
    weights: Optional[List[np.ndarray]],
    fit_columns: Optional[np.ndarray]
) -> Dict[str, Union[str, int, float]]:
    """
    Trains on all years except one and evaluates on the held-out year.

    Args:
        iteration: Repeat index, used as the seed for shuffling and initialization.
        scenario_name: Name of the transfer learning scenario.
        left_out_year: Year excluded from training and used for testing.
        X: Feature matrix for the scenario.
        y_raw: Raw target values.
        years: Year labels aligned to rows in X/y.
        input_dim: Number of input features.
        params: Hyperparameters selected by Optuna.
        weights: Optional transfer weights.
        fit_columns: Column indices to standardize on the training years.

    Returns:
        One LOYO result row with metrics and fold metadata.
    """
    test_mask = years == left_out_year
    train_mask = ~test_mask
    train_rows = int(np.sum(train_mask))
    test_rows = int(np.sum(test_mask))

    result = {
        "Iteration": iteration,
        "Scenario": scenario_name,
        "Left_Out_Year": int(left_out_year),
        "Train_Rows": train_rows,
        "Test_Rows": test_rows,
        "RMSE": np.nan,
        "NRMSE_Percent": np.nan,
        "Status": "Completed",
        "Error": "",
    }

    if train_rows < MIN_LOYO_TRAIN_ROWS or test_rows < MIN_LOYO_TEST_ROWS:
        result["Status"] = "Skipped"
        result["Error"] = (
            f"Insufficient rows for LOYO fold "
            f"(train={train_rows}, test={test_rows})."
        )
        return result

    try:
        # Keras validation_split takes the last rows unshuffled; masked rows are still in file order.
        train_idx = np.random.default_rng(iteration).permutation(np.flatnonzero(train_mask))
        metrics, _, _ = fit_and_score(
            X[train_idx], y_raw[train_idx], X[test_mask], y_raw[test_mask],
            input_dim, params, weights, fit_columns, iteration
        )
        result.update(metrics)
    except Exception as exc:
        result["Status"] = "Failed"
        result["Error"] = str(exc)

    return result


def plot_loyo_metric(completed_df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    """
    Save a year-by-year bar plot for one LOYO metric.

    Args:
        completed_df: Completed LOYO rows only.
        metric: Column to plot, typically NRMSE_Percent or RMSE.
        ylabel: Axis label.
        filename: Destination PNG name under Results/Graphs.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(data=completed_df, x="Left_Out_Year", y=metric, hue="Scenario")
    plt.title("Leave-One-Year-Out Transfer Learning Evaluation (100 Iterations)")
    plt.xlabel("Held-Out Year")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.legend(title="Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close()


def plot_loyo_results(loyo_df: pd.DataFrame) -> None:
    """
    Saves year-by-year NRMSE and RMSE comparison plots for completed LOYO folds.

    Args:
        loyo_df: DataFrame returned by run_loyo_evaluation.
    """
    completed_df = loyo_df[loyo_df["Status"] == "Completed"].copy()
    if completed_df.empty:
        print("LOYO plotting skipped: no completed folds.")
        return

    plot_loyo_metric(
        completed_df,
        "NRMSE_Percent",
        "Normalized RMSE (%)",
        f"NN_LOYO_NRMSE_{TIMESTAMP}.png",
    )
    plot_loyo_metric(
        completed_df,
        "RMSE",
        "RMSE",
        f"NN_LOYO_RMSE_{TIMESTAMP}.png",
    )


def run_loyo_evaluation(
    data: Dict,
    scenarios: List[Scenario],
    best_params: Dict[str, Union[int, float]]
) -> pd.DataFrame:
    """
    Runs leave-one-year-out evaluation after the random-split benchmark.

    Args:
        data: Loaded data dictionary containing rice features, raw targets, and years.
        scenarios: Transfer learning scenarios to evaluate.
        best_params: Hyperparameters selected for the neural network.

    Returns:
        DataFrame containing one row per scenario, held-out year, and iteration.
    """
    years = np.asarray(data['rice_years'], dtype=int)
    unique_years = sorted(np.unique(years))

    if len(unique_years) < 2:
        raise ValueError("LOYO requires at least two years of target-crop data.")

    print(f"\nStarting LOYO Evaluation across {len(unique_years)} years: {unique_years}")
    rows = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futs = []
        for name, X, dim, weights, fit_columns in scenarios:
            for left_out_year in unique_years:
                for i in range(NUM_ITERATIONS):
                    futs.append(executor.submit(
                        evaluate_loyo_fold,
                        i,
                        name,
                        left_out_year,
                        X,
                        data['rice_y_raw'],
                        years,
                        dim,
                        best_params,
                        weights,
                        fit_columns
                    ))

        print("Processing LOYO Iterations...")
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                rows.append(f.result())
            except Exception as exc:
                print(f"LOYO worker failed: {exc}")

    loyo_df = pd.DataFrame(rows)
    loyo_results_path = f"{RESULTS_DIR}/Errors/loyo_results_{TIMESTAMP}.csv"
    loyo_df.to_csv(loyo_results_path, index=False)
    plot_loyo_results(loyo_df)

    print("\n--- LOYO Performance by Scenario ---")
    completed_df = loyo_df[loyo_df["Status"] == "Completed"]
    if completed_df.empty:
        print("No completed LOYO folds.")
    else:
        for scenario_name, scenario_df in completed_df.groupby("Scenario"):
            print(
                f"{scenario_name:12}: "
                f"median NRMSE={scenario_df['NRMSE_Percent'].median():.2f}% | "
                f"median RMSE={scenario_df['RMSE'].median():.2f}"
            )
        completed_df.groupby(["Scenario", "Left_Out_Year"])[["RMSE", "NRMSE_Percent"]].describe().to_csv(
            f"{RESULTS_DIR}/Errors/loyo_error_description_{TIMESTAMP}.csv"
        )

    return loyo_df


def optimize_hyperparameters(
    data: Dict,
    n_trials: int = 20,
    n_jobs: int = NUM_WORKERS
) -> Dict[str, Union[int, float]]:
    """
    Search the soybean network's architecture and learning rate with Optuna.

    Each trial splits soybean rows using its trial number as the seed, fits
    scalers on that trial's training rows, and scores validation MSE on
    standardized yield.

    Args:
        data: Dictionary containing 'soy_X_raw', 'soy_y_raw', and 'input_dim_soy'.
        n_trials: Number of trials to run.
        n_jobs: Number of trials run in parallel threads.

    Returns:
        Best hyperparameters found (n_layers, lr, and units per layer).
    """

    def objective(t: optuna.Trial) -> float:
        """Train one candidate network and return its validation MSE."""
        p = {
            'n_layers': t.suggest_int('n_layers', 1, 3),
            'lr': t.suggest_float('lr', 1e-4, 1e-3, log=True)
        }

        for i in range(p['n_layers']):
            p[f'n_units_l{i}'] = t.suggest_int(f'n_units_l{i}', 32, 128)

        Xt, Xv, yt, yv = train_test_split(
            data['soy_X_raw'],
            data['soy_y_raw'],
            test_size=0.2,
            random_state=t.number
        )
        Xt, Xv, yt_z, y_scaler = prepare_split(
            Xt, Xv, yt, fit_columns=np.arange(data['input_dim_soy'])
        )
        yv_z = y_scaler.transform(yv.reshape(-1, 1)).flatten()

        m = build_and_train(Xt, yt_z, data['input_dim_soy'], p, seed=t.number)
        return float(m.evaluate(Xv, yv_z, verbose=0))

    print("Starting Optuna Optimization...")
    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    os.makedirs(f"{RESULTS_DIR}/Models", exist_ok=True)
    save_json(
        f"{RESULTS_DIR}/Models/optuna_best_params_{TIMESTAMP}.json",
        {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials_run": len(study.trials),
        },
    )

    return study.best_params


def format_ttest(label: str, left: pd.Series, right: pd.Series) -> str:
    """
    Format one independent two-sample t-test for the stats log.

    Args:
        label: Human-readable comparison name.
        left: First sample of NRMSE (or RMSE) values.
        right: Second sample of the same metric.

    Returns:
        One line of t-statistic and p-value text.
    """
    result = ttest_ind(left, right)
    return f"{label}: t={result.statistic:.2f}, p={result.pvalue:.4e}"


def plot_metric_boxplot(values_by_scenario: Dict[str, List[float]], ylabel: str, title: str, filename: str) -> None:
    """
    Draw a labeled boxplot for one metric across the four scenarios.

    Args:
        values_by_scenario: Mapping of scenario name to per-split scores.
        ylabel: Axis label.
        title: Figure title.
        filename: Destination PNG name under Results/Graphs.
    """
    labels = list(values_by_scenario.keys())
    series = [values_by_scenario[name] for name in labels]
    plt.figure(figsize=(12, 7))
    plt.boxplot(series, patch_artist=True, tick_labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    medians = [float(np.median(values)) for values in series]
    for index, median in enumerate(medians, start=1):
        plt.text(index, median, f"{median:.2f}", ha="center", va="bottom", fontsize=10, color="blue")
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close()


if __name__ == '__main__':
    folder_creation()
    data = load_data()

    best_params = optimize_hyperparameters(data, n_trials=50)

    print("Evaluating soybean source holdout...")
    source_metrics = evaluate_source_holdout(data, best_params)
    print(
        f"Soybean holdout: RMSE={source_metrics['RMSE']:.2f}, "
        f"NRMSE={source_metrics['NRMSE_Percent']:.2f}%"
    )

    print("Training Base Soybean Model...")
    # Soybean rows are in file order; shuffle so validation_split is not one block of stations/years.
    soy_order = np.random.default_rng(GLOBAL_SEED).permutation(data['soy_rows'])
    base_m = build_and_train(
        data['soy_X'][soy_order], data['soy_y_z'][soy_order], data['input_dim_soy'],
        best_params, seed=GLOBAL_SEED
    )
    soy_model_path = save_trained_model(base_m, "soybean_source")
    print(f"Saved soybean source model -> {soy_model_path}")
    w_soy = base_m.get_weights()

    # Full rice model: soybean first-layer rows go to the matching rice columns by name,
    # rice-only columns start small and random.
    init_rng = np.random.default_rng(GLOBAL_SEED)
    w0_rice_full = init_rng.normal(scale=0.01, size=(data['rice_X_full'].shape[1], w_soy[0].shape[1]))
    rice_cols = data['rice_features_full']
    for i, col_name in enumerate(data['common_features']):
        w0_rice_full[rice_cols.index(col_name), :] = w_soy[0][i, :]
    weights_full = [w0_rice_full] + w_soy[1:]

    scenarios: List[Scenario] = [
        ("Warm_Full", data['rice_X_full'], data['rice_X_full'].shape[1], weights_full, data['rice_only_idx']),
        ("Cold_Full", data['rice_X_full'], data['rice_X_full'].shape[1], None, data['rice_only_idx']),
        ("Warm_Common", data['rice_X_comm'], data['input_dim_soy'], w_soy, None),
        ("Cold_Common", data['rice_X_comm'], data['input_dim_soy'], None, None)
    ]

    nrmse_metrics = {s[0]: [] for s in scenarios}
    rmse_metrics = {s[0]: [] for s in scenarios}
    tidy_rows = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for name, X, dim, weights, fit_columns in scenarios:
            print(f"Processing Scenario: {name}")
            futs = [
                executor.submit(worker_task, i, X, data['rice_y_raw'], dim, best_params, weights, fit_columns)
                for i in range(NUM_ITERATIONS)
            ]
            for f in tqdm(as_completed(futs), total=NUM_ITERATIONS):
                try:
                    res = f.result()
                except Exception as exc:
                    print(f"{name} worker failed: {exc}")
                    continue
                tidy_rows.append({"Scenario": name, **res})
                if not np.isnan(res["NRMSE_Percent"]):
                    nrmse_metrics[name].append(res["NRMSE_Percent"])
                    rmse_metrics[name].append(res["RMSE"])

    combined_df = pd.DataFrame.from_dict(nrmse_metrics, orient='index').transpose()
    combined_df.to_csv(f"{RESULTS_DIR}/Errors/final_results_{TIMESTAMP}.csv")
    rmse_df = pd.DataFrame.from_dict(rmse_metrics, orient='index').transpose()
    rmse_df.to_csv(f"{RESULTS_DIR}/Errors/final_rmse_results_{TIMESTAMP}.csv")
    tidy_df = pd.DataFrame(tidy_rows)
    tidy_df.to_csv(f"{RESULTS_DIR}/Errors/random_split_metrics_{TIMESTAMP}.csv", index=False)
    combined_df.describe().to_csv(f"{RESULTS_DIR}/Errors/random_split_nrmse_description_{TIMESTAMP}.csv")
    rmse_df.describe().to_csv(f"{RESULTS_DIR}/Errors/random_split_rmse_description_{TIMESTAMP}.csv")

    run_representative_diagnostics(data, scenarios, best_params)
    loyo_df = run_loyo_evaluation(data, scenarios, best_params)

    plot_metric_boxplot(
        nrmse_metrics,
        ylabel="Normalized RMSE (%)",
        title="NN Transfer Learning",
        filename=f"NN_Boxplot_{TIMESTAMP}.png",
    )
    plot_metric_boxplot(
        rmse_metrics,
        ylabel="RMSE",
        title="NN Transfer Learning (RMSE)",
        filename=f"NN_Boxplot_RMSE_{TIMESTAMP}.png",
    )

    plt.figure(figsize=(12, 7))
    for col in combined_df.columns:
        sns.kdeplot(combined_df[col], label=col, fill=True, alpha=0.3)
    plt.title('Distribution of Normalized RMSE')
    plt.xlabel('Normalized RMSE (%)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", f"NN_Density_{TIMESTAMP}.png"), dpi=300)
    plt.close()

    print("\n--- Final Performance Medians ---")
    for name in nrmse_metrics:
        print(
            f"{name:12}: NRMSE {np.median(nrmse_metrics[name]):.2f}% | "
            f"RMSE {np.median(rmse_metrics[name]):.2f}"
        )

    completed_loyo = loyo_df[loyo_df["Status"] == "Completed"]
    stats_lines = [
        f"Best architecture: {best_params}",
        f"Common encoded features: {data['n_common_features']}",
        f"Soybean rows: {data['soy_rows']} | Rice rows: {data['rice_rows']}",
        (
            f"Soybean source holdout (seed {REPRESENTATIVE_SEED}): "
            f"RMSE={source_metrics['RMSE']:.4f}, "
            f"NRMSE={source_metrics['NRMSE_Percent']:.4f}%"
        ),
        "",
        "--- Statistical Test Results (Independent Two-Sample T-test on NRMSE) ---",
        format_ttest("Warm (Full) vs Cold (Full)", nrmse_metrics["Warm_Full"], nrmse_metrics["Cold_Full"]),
        format_ttest("Warm (Common) vs Cold (Common)", nrmse_metrics["Warm_Common"], nrmse_metrics["Cold_Common"]),
        format_ttest("Warm (Common) vs Warm (Full)", nrmse_metrics["Warm_Common"], nrmse_metrics["Warm_Full"]),
    ]
    if not completed_loyo.empty:
        # No t-test here: iterations within a year share identical train/test rows,
        # so their spread is initialization noise, not independent samples.
        loyo_medians = completed_loyo.pivot_table(
            index="Scenario", columns="Left_Out_Year", values="NRMSE_Percent", aggfunc="median"
        )
        stats_lines.extend([
            "",
            "--- LOYO Median NRMSE (%) by Held-Out Year ---",
            loyo_medians.round(2).to_string(),
        ])
    stats_output = "\n".join(stats_lines) + "\n"

    print("Pipeline completed.")
    print(stats_output)

    with open(f"{RESULTS_DIR}/Errors/statistical_tests_{TIMESTAMP}.txt", "w") as f:
        f.write(stats_output)
