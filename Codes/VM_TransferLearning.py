"""
Soybean-to-rice neural transfer learning.

Tunes a network on soybean, trains NUM_SOURCE_MODELS soybean source networks
with different seeds, maps their weights onto rice, and compares warm vs cold
start on the full rice feature set and on the features shared with soybean.

One run covers every combination of:

    shared scaling   the modes listed in VM_SHARED_SCALING (see below)
    training size    each of TRAIN_FRACTIONS of the rice training rows
    evaluation       random 80/20 splits and leave-one-year-out (LOYO)

Warm repeat i starts from source network i % NUM_SOURCE_MODELS, so warm
results include the variation between source networks. Runs that share a
repeat index share the split, the training subsample, and the seed, so
scenarios and scaling modes are compared split by split. A mean-yield
baseline is scored on the same splits. Every result is reported as RMSE and
NRMSE.

Rice-only features and yield are scaled on each split's training rows. Rice's
shared features are scaled in one of two ways:

    soybean   Soybean's scaler, so transferred weights see the units they
              were trained on.
    per_crop  Rice's own training rows, like soybean is scaled on its own
              rows. Both crops are then centered at 0, and a value means
              "relative to that crop's usual season".

Warm and cold networks use the same training schedule, so the only
difference between them is the starting weights.
"""
import os
# One op thread per worker process. Has to be set before NumPy/TensorFlow start their thread pools.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import json
import time
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
from scipy.stats import t as t_dist, wilcoxon
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

# Experiment grid
NUM_RANDOM_SPLITS = 50
NUM_LOYO_REPEATS = 25
NUM_SOURCE_MODELS = 5
TRAIN_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
TEST_FRACTION = 0.2
OPTUNA_TRIALS = 50
MIN_TRAIN_ROWS = 50
MIN_TEST_ROWS = 1
CHECKPOINT_EVERY = 500

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
ID_COLUMNS = ("Sl", "GPS")
# Yield is computed from harvested grain weight corrected to a standard moisture.
HARVEST_OUTCOME_PREFIXES = ("harvest.weight", "harvest.moisture")

# Scaling of rice's shared features (see module docstring). Comma-separated.
SHARED_SCALING_MODES = ("soybean", "per_crop")
SHARED_SCALING_SETTING = os.environ.get("VM_SHARED_SCALING", "soybean,per_crop")

# Training schedule shared by warm and cold networks
HEAD_EPOCHS = 10
MAX_FINE_TUNE_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 32

# Scenario name -> (rice feature set, starts from soybean weights)
SCENARIOS = {
    "Warm_Full": ("full", True),
    "Cold_Full": ("full", False),
    "Warm_Common": ("common", True),
    "Cold_Common": ("common", False),
}
BASELINE_NAME = "Mean_Yield_Baseline"
# Pairs compared split by split. Differences are always first minus second.
SCENARIO_PAIRS = (
    ("Warm_Full", "Cold_Full"),
    ("Warm_Common", "Cold_Common"),
    ("Warm_Full", "Warm_Common"),
    ("Cold_Full", "Cold_Common"),
)
SCENARIO_COLORS = {
    "Warm_Full": "tab:blue",
    "Cold_Full": "tab:orange",
    "Warm_Common": "tab:green",
    "Cold_Common": "tab:red",
    BASELINE_NAME: "gray",
}
# Columns that identify the rows a fit was trained and scored on
PAIR_KEYS = ["Evaluation", "Left_Out_Year", "Train_Fraction", "Iteration"]

# Set in each worker process by init_worker
_WORKER_STATE: Dict = {}


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


def parse_scaling_modes(setting: str) -> Tuple[str, ...]:
    """
    Turn a comma-separated VM_SHARED_SCALING value into scaling modes.

    Args:
        setting: For example "soybean,per_crop" or "per_crop".

    Returns:
        Modes in the given order, without duplicates.

    Raises:
        ValueError: If the setting is empty or names an unknown mode.
    """
    modes = tuple(dict.fromkeys(part.strip() for part in setting.split(",") if part.strip()))
    unknown = [mode for mode in modes if mode not in SHARED_SCALING_MODES]
    if not modes or unknown:
        raise ValueError(
            f"VM_SHARED_SCALING='{setting}' is invalid. Use a comma-separated list of {SHARED_SCALING_MODES}."
        )
    return modes


def load_data() -> dict:
    """
    Load both crops, align their shared features, and standardize soybean.

    Soybean features and yield are standardized on all soybean rows to train
    the source networks. Rice features are returned unscaled; see
    build_rice_inputs for how each scaling mode prepares them.

    Returns:
        A dictionary of soybean matrices, raw rice matrices, rice years,
        the soybean feature scaler, and feature bookkeeping.
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
    common_idx = np.array([rice_features_full.index(col) for col in common], dtype=int)

    return {
        'soy_X_raw': soy_X_raw,
        'soy_y_raw': soy_y_raw,
        'soy_X': scaler_soy_X.transform(soy_X_raw),
        'soy_y_z': scaler_soy_y.transform(soy_y_raw.reshape(-1, 1)).flatten(),
        'scaler_soy_X': scaler_soy_X,
        'rice_X_full_raw': rice[rice_features_full].to_numpy(dtype=float),
        'rice_X_comm_raw': rice[common].to_numpy(dtype=float),
        'rice_y_raw': rice['Yield'].to_numpy(dtype=float),
        'rice_years': np.asarray(rice_years, dtype=int),
        'common_idx': common_idx,
        'rice_only_idx': np.setdiff1d(np.arange(len(rice_features_full)), common_idx),
        'input_dim_soy': len(common),
        'rice_features_full': rice_features_full,
        'common_features': common,
        'soy_only_features': soy_only,
        'rice_only_features': rice_only,
        'n_common_features': len(common),
        'soy_rows': int(len(soy)),
        'rice_rows': int(len(rice)),
    }


def build_rice_inputs(data: dict, shared_scaling: str) -> dict:
    """
    Prepare the rice feature matrices for one scaling mode.

    soybean   Shared columns are transformed here with the soybean scaler,
              in both matrices. Only rice-only columns are left for
              per-split scaling.
    per_crop  Nothing is transformed here. Every rice column is left for
              per-split scaling on that split's training rows.

    Args:
        data: Output of load_data.
        shared_scaling: One of SHARED_SCALING_MODES.

    Returns:
        X_full and X_common matrices, plus the column indices each one scales
        per split (None means nothing is scaled per split).

    Raises:
        ValueError: If `shared_scaling` is not a known mode.
    """
    if shared_scaling not in SHARED_SCALING_MODES:
        raise ValueError(f"Unknown shared scaling '{shared_scaling}'. Expected one of {SHARED_SCALING_MODES}.")

    X_full = data['rice_X_full_raw'].copy()
    X_common = data['rice_X_comm_raw'].copy()
    if shared_scaling == "soybean":
        X_common = data['scaler_soy_X'].transform(X_common)
        X_full[:, data['common_idx']] = X_common
        return {
            'X_full': X_full, 'full_fit_idx': data['rice_only_idx'],
            'X_common': X_common, 'common_fit_idx': None,
        }
    return {
        'X_full': X_full, 'full_fit_idx': np.arange(X_full.shape[1]),
        'X_common': X_common, 'common_fit_idx': np.arange(X_common.shape[1]),
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
            in soybean units under soybean scaling) and are left untouched.
            None scales nothing.

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


def select_rows(
    evaluation: str,
    iteration: int,
    fraction: float,
    years: np.ndarray,
    left_out_year: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick the training and test rows for one fit.

    The test rows depend only on the evaluation and repeat index, so every
    training fraction is scored on the same rows. Training rows are the first
    `fraction` of one fixed shuffle of the available rows, so a smaller
    fraction is always a subset of a larger one.

    Args:
        evaluation: "random" for an 80/20 split, "loyo" for a held-out year.
        iteration: Repeat index. Seeds the split and the shuffle.
        fraction: Share of the available training rows to keep, in (0, 1].
        years: Rice year per row.
        left_out_year: Held-out year for LOYO; ignored for random splits.

    Returns:
        Training row indices (shuffled) and test row indices.
    """
    if evaluation == "loyo":
        test_idx = np.flatnonzero(years == left_out_year)
        pool = np.flatnonzero(years != left_out_year)
    else:
        pool, test_idx = train_test_split(
            np.arange(len(years)), test_size=TEST_FRACTION, random_state=iteration
        )
    # Shuffled so Keras validation_split (the last rows) is not one block of stations or years.
    pool = np.random.default_rng(iteration).permutation(pool)
    n_train = max(1, int(round(fraction * len(pool))))
    return pool[:n_train], np.sort(test_idx)


def map_source_weights(source_weights: List[np.ndarray], data: dict, source_index: int) -> dict:
    """
    Turn one soybean network's weights into starting weights for both rice feature sets.

    The shared-feature network uses the soybean weights as they are. For the
    full network, soybean first-layer rows go to the matching rice columns by
    name and rice-only columns start small and random.

    Args:
        source_weights: Output of get_weights() on a soybean network.
        data: Output of load_data.
        source_index: Source network number, used to seed the rice-only rows.

    Returns:
        {"common": weights, "full": weights}.
    """
    rng = np.random.default_rng(GLOBAL_SEED + source_index)
    first_layer = rng.normal(
        scale=0.01, size=(len(data['rice_features_full']), source_weights[0].shape[1])
    )
    first_layer[data['common_idx'], :] = source_weights[0]
    return {"common": source_weights, "full": [first_layer] + source_weights[1:]}


def init_worker(state: dict) -> None:
    """
    Store read-only inputs in a worker process once, so tasks stay small.

    Args:
        state: Matrices, targets, parameters, and weights the tasks read.
    """
    global _WORKER_STATE
    _WORKER_STATE = state


def train_source_task(source_index: int) -> Tuple[int, List[np.ndarray], str]:
    """
    Train one soybean source network on all soybean rows and save it.

    Source 0 uses GLOBAL_SEED, the same seed as the single source network of
    earlier runs.

    Args:
        source_index: Source number; the seed is GLOBAL_SEED + source_index.

    Returns:
        Source number, trained weights, and the saved model path.
    """
    state = _WORKER_STATE
    seed = GLOBAL_SEED + source_index
    order = np.random.default_rng(seed).permutation(len(state['soy_y_z']))
    model = build_and_train(
        state['soy_X'][order], state['soy_y_z'][order], state['soy_X'].shape[1],
        state['params'], seed=seed
    )
    path = save_trained_model(
        model, f"soybean_source_{source_index}", state['results_dir'], state['timestamp']
    )
    return source_index, model.get_weights(), path


def run_rice_task(task: dict) -> dict:
    """
    Train and score one rice network described by `task`.

    Args:
        task: Evaluation, Scaling, Scenario, Train_Fraction, Iteration,
            Left_Out_Year, Source_Model, and Save_Diagnostics.

    Returns:
        One result row. Failed or skipped fits keep NaN metrics and say why.
        When Save_Diagnostics is set, the row also carries the saved model
        path and the test-set predictions under "_diagnostics".
    """
    state = _WORKER_STATE
    feature_set, warm = SCENARIOS[task['Scenario']]
    inputs = state['rice_inputs'][task['Scaling']]
    X = inputs[f'X_{feature_set}']
    fit_columns = inputs[f'{feature_set}_fit_idx']
    weights = state['source_weights'][task['Source_Model']][feature_set] if warm else None
    y_raw = state['y_raw']

    train_idx, test_idx = select_rows(
        task['Evaluation'], task['Iteration'], task['Train_Fraction'], state['years'], task['Left_Out_Year']
    )
    row = {key: value for key, value in task.items() if key != 'Save_Diagnostics'}
    row.update({
        "Train_Rows": int(len(train_idx)), "Test_Rows": int(len(test_idx)),
        "RMSE": np.nan, "NRMSE_Percent": np.nan, "Status": "Completed", "Error": "",
    })
    if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) < MIN_TEST_ROWS:
        row.update(Status="Skipped", Error=f"Too few rows (train={len(train_idx)}, test={len(test_idx)}).")
        return row

    try:
        metrics, model, pred_raw = fit_and_score(
            X[train_idx], y_raw[train_idx], X[test_idx], y_raw[test_idx],
            X.shape[1], state['params'], weights, fit_columns, task['Iteration']
        )
        row.update(metrics)
        if task.get('Save_Diagnostics'):
            stem = f"rice_{task['Scaling']}_{task['Scenario'].lower()}"
            row['_diagnostics'] = {
                "model_path": save_trained_model(model, stem, state['results_dir'], state['timestamp']),
                "y_true": y_raw[test_idx],
                "y_pred": pred_raw,
            }
    except Exception as exc:
        row.update(Status="Failed", Error=str(exc))
    return row


def build_tasks(scalings: Sequence[str], years: Sequence[int]) -> List[dict]:
    """
    List every rice fit in the experiment grid, largest training sets first.

    Warm repeat i uses source network i % NUM_SOURCE_MODELS. The full-data
    random split with index REPRESENTATIVE_SEED also saves its model and
    predictions for the scatter plots.

    Args:
        scalings: Shared-feature scaling modes to run.
        years: Rice years available for LOYO.

    Returns:
        Task dictionaries for run_rice_task.
    """
    tasks = []
    for fraction in sorted(TRAIN_FRACTIONS, reverse=True):
        for scaling in scalings:
            for scenario, (_, warm) in SCENARIOS.items():
                splits = [("random", None, i) for i in range(NUM_RANDOM_SPLITS)]
                splits += [("loyo", int(year), i) for year in years for i in range(NUM_LOYO_REPEATS)]
                for evaluation, year, iteration in splits:
                    tasks.append({
                        "Evaluation": evaluation,
                        "Scaling": scaling,
                        "Scenario": scenario,
                        "Train_Fraction": float(fraction),
                        "Iteration": int(iteration),
                        "Left_Out_Year": year,
                        "Source_Model": int(iteration % NUM_SOURCE_MODELS) if warm else None,
                        "Save_Diagnostics": (
                            evaluation == "random" and fraction == 1.0 and iteration == REPRESENTATIVE_SEED
                        ),
                    })
    return tasks


def results_frame(rows: List[dict]) -> pd.DataFrame:
    """
    Tabulate result rows with a stable column order.

    Args:
        rows: Result rows from run_rice_task or mean_baseline_rows.

    Returns:
        DataFrame with numeric year and source columns (NaN where not used).
    """
    columns = [
        "Evaluation", "Scaling", "Scenario", "Train_Fraction", "Iteration", "Left_Out_Year",
        "Source_Model", "Train_Rows", "Test_Rows", "RMSE", "NRMSE_Percent", "Status", "Error",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["Left_Out_Year"] = pd.to_numeric(frame["Left_Out_Year"], errors="coerce")
    frame["Source_Model"] = pd.to_numeric(frame["Source_Model"], errors="coerce")
    return frame


def run_rice_tasks(tasks: List[dict], state: dict) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Run every rice fit in parallel, checkpointing rows as they finish.

    Args:
        tasks: Output of build_tasks.
        state: Shared inputs handed to each worker once.

    Returns:
        Result rows as a DataFrame, and the diagnostics of flagged tasks.
    """
    rows, diagnostics = [], []
    checkpoint_path = f"{RESULTS_DIR}/Errors/partial_results_{TIMESTAMP}.csv"
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker, initargs=(state,)) as executor:
        futures = [executor.submit(run_rice_task, task) for task in tasks]
        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
            try:
                row = future.result()
            except Exception as exc:
                print(f"Rice worker failed: {exc}")
                continue
            extra = row.pop('_diagnostics', None)
            if extra is not None:
                diagnostics.append({**extra, "Scaling": row['Scaling'], "Scenario": row['Scenario']})
            rows.append(row)
            if count % CHECKPOINT_EVERY == 0:
                results_frame(rows).to_csv(checkpoint_path, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return results_frame(rows), diagnostics


def mean_baseline_rows(tasks: List[dict], y_raw: np.ndarray, years: np.ndarray) -> List[dict]:
    """
    Score "predict the mean training yield" on exactly the rows each fit used.

    One baseline row is made per cold-scenario task, since cold tasks cover
    every (scaling, evaluation, fraction, year, repeat) once per feature set.

    Args:
        tasks: Output of build_tasks.
        y_raw: Rice yield in original units.
        years: Rice year per row.

    Returns:
        Result rows with Scenario set to BASELINE_NAME.
    """
    rows = []
    for task in tasks:
        if task['Scenario'] != "Cold_Full":
            continue
        train_idx, test_idx = select_rows(
            task['Evaluation'], task['Iteration'], task['Train_Fraction'], years, task['Left_Out_Year']
        )
        prediction = np.full(len(test_idx), y_raw[train_idx].mean())
        rows.append({
            **{key: value for key, value in task.items() if key != 'Save_Diagnostics'},
            "Scenario": BASELINE_NAME, "Source_Model": None,
            "Train_Rows": int(len(train_idx)), "Test_Rows": int(len(test_idx)),
            **calculate_error_metrics(y_raw[test_idx], prediction),
            "Status": "Completed", "Error": "",
        })
    return rows


def paired_statistics(pairs: pd.DataFrame, evaluation: str) -> dict:
    """
    Summarize split-by-split NRMSE differences between two configurations.

    Random splits get a Wilcoxon signed-rank test and the Nadeau and Bengio
    corrected resampled t-test, which accounts for overlapping training sets.
    LOYO repeats reuse the same rows and differ only in initialization and
    source network, so they get medians and win shares but no p-values.

    Args:
        pairs: Rows with NRMSE_Percent_A, NRMSE_Percent_B, Train_Rows, Test_Rows.
        evaluation: "random" or "loyo".

    Returns:
        Pair count, medians, median difference (A minus B), share of pairs
        where A had the lower error, and p-values where they apply.
    """
    diff = (pairs["NRMSE_Percent_A"] - pairs["NRMSE_Percent_B"]).to_numpy()
    n_pairs = len(diff)
    stats = {
        "N_Pairs": n_pairs,
        "Median_A": float(pairs["NRMSE_Percent_A"].median()),
        "Median_B": float(pairs["NRMSE_Percent_B"].median()),
        "Median_Diff": float(np.median(diff)),
        "A_Better_Share": float(np.mean(diff < 0)),
        "Wilcoxon_p": np.nan,
        "Corrected_t_p": np.nan,
    }
    if evaluation != "random" or n_pairs < 5:
        return stats
    if np.any(diff != 0):
        stats["Wilcoxon_p"] = float(wilcoxon(diff).pvalue)
    variance = float(np.var(diff, ddof=1))
    if variance > 0:
        test_to_train = float(pairs["Test_Rows"].median() / pairs["Train_Rows"].median())
        t_stat = diff.mean() / np.sqrt((1.0 / n_pairs + test_to_train) * variance)
        stats["Corrected_t_p"] = float(2 * t_dist.sf(abs(t_stat), df=n_pairs - 1))
    return stats


def compare_configurations(
    completed: pd.DataFrame,
    fixed_column: str,
    varied_column: str,
    pairs: Sequence[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Compare two values of one column while holding another column fixed.

    For example fixed_column="Scaling", varied_column="Scenario" compares
    Warm_Full with Cold_Full within each scaling mode. Rows are matched on
    PAIR_KEYS, so both sides were trained and scored on the same rows.

    Args:
        completed: Completed result rows.
        fixed_column: Column held constant within a comparison.
        varied_column: Column whose two values are compared.
        pairs: (A, B) value pairs of `varied_column`.

    Returns:
        One row per fixed value, pair, evaluation, held-out year, and fraction.
    """
    rows = []
    for fixed_value, block in completed.groupby(fixed_column):
        for a, b in pairs:
            left = block.loc[block[varied_column] == a, PAIR_KEYS + ["NRMSE_Percent", "Train_Rows", "Test_Rows"]]
            right = block.loc[block[varied_column] == b, PAIR_KEYS + ["NRMSE_Percent"]]
            merged = left.merge(right, on=PAIR_KEYS, suffixes=("_A", "_B"))
            keys = ["Evaluation", "Left_Out_Year", "Train_Fraction"]
            for (evaluation, year, fraction), group in merged.groupby(keys, dropna=False):
                rows.append({
                    "Evaluation": evaluation, "Left_Out_Year": year, "Train_Fraction": fraction,
                    "Held_Constant": f"{fixed_column}={fixed_value}", "A": a, "B": b,
                    **paired_statistics(group, evaluation),
                })
    return pd.DataFrame(rows)


def summarize_results(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Median and interquartile range of NRMSE and RMSE per configuration.

    Args:
        completed: Completed result rows, including the baseline.

    Returns:
        One row per evaluation, scaling, scenario, fraction, and held-out year.
    """
    keys = ["Evaluation", "Scaling", "Scenario", "Train_Fraction", "Left_Out_Year"]
    grouped = completed.groupby(keys, dropna=False)
    summary = grouped.agg(
        N=("NRMSE_Percent", "size"),
        Train_Rows=("Train_Rows", "median"),
        NRMSE_Median=("NRMSE_Percent", "median"),
        NRMSE_Q25=("NRMSE_Percent", lambda s: s.quantile(0.25)),
        NRMSE_Q75=("NRMSE_Percent", lambda s: s.quantile(0.75)),
        RMSE_Median=("RMSE", "median"),
    )
    return summary.reset_index()


def summarize_sources(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Median NRMSE of warm runs per source network.

    Args:
        completed: Completed result rows.

    Returns:
        One row per evaluation, scaling, warm scenario, fraction, held-out
        year, and source network.
    """
    warm = completed[completed["Source_Model"].notna()]
    keys = ["Evaluation", "Scaling", "Scenario", "Train_Fraction", "Left_Out_Year", "Source_Model"]
    return warm.groupby(keys, dropna=False)["NRMSE_Percent"].agg(N="size", NRMSE_Median="median").reset_index()


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


def evaluate_source_holdout(
    data: Dict,
    params: Dict[str, Union[int, float]],
    seed: int = REPRESENTATIVE_SEED
) -> Dict[str, Union[int, float]]:
    """
    Score the soybean network architecture on an 80/20 holdout.

    The transfer weights come from separate fits on all soybean rows. This
    holdout is only a reported source-domain number for the paper.

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


def train_source_models(data: dict, params: dict) -> List[List[np.ndarray]]:
    """
    Train NUM_SOURCE_MODELS soybean source networks in parallel.

    Args:
        data: Output of load_data.
        params: Hyperparameters selected by Optuna.

    Returns:
        Weights of each source network, indexed by source number.
    """
    state = {
        "soy_X": data['soy_X'], "soy_y_z": data['soy_y_z'], "params": params,
        "results_dir": RESULTS_DIR, "timestamp": TIMESTAMP,
    }
    weights: List[Optional[List[np.ndarray]]] = [None] * NUM_SOURCE_MODELS
    workers = max(1, min(NUM_SOURCE_MODELS, NUM_WORKERS))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(state,)) as executor:
        for future in as_completed([executor.submit(train_source_task, k) for k in range(NUM_SOURCE_MODELS)]):
            index, source_weights, path = future.result()
            weights[index] = source_weights
            print(f"  Source {index} -> {path}")
    return weights


def save_trained_model(model, stem: str, results_dir: str, timestamp: str) -> str:
    """
    Persist a Keras model under Results/Models.

    Prefers the native `.keras` format and falls back to HDF5 when the
    installed TensorFlow build does not accept it.

    Args:
        model: Trained Keras model.
        stem: Filename stem without extension.
        results_dir: Results folder of the run.
        timestamp: Run timestamp. Passed explicitly because each worker
            process computes its own TIMESTAMP on import.

    Returns:
        Path of the file that was written.
    """
    keras_path = f"{results_dir}/Models/{stem}_{timestamp}.keras"
    hdf5_path = f"{results_dir}/Models/{stem}_{timestamp}.h5"
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


def ordered_scalings(frame: pd.DataFrame) -> List[str]:
    """
    Scaling modes present in `frame`, in SHARED_SCALING_MODES order, so every plot lays them out the same way.

    Args:
        frame: Any results table with a Scaling column.

    Returns:
        Scaling mode names.
    """
    present = set(frame["Scaling"])
    return [mode for mode in SHARED_SCALING_MODES if mode in present]


def plot_learning_curves(summary: pd.DataFrame, evaluation: str, filename: str) -> None:
    """
    Plot median NRMSE against rice training size, one panel per scaling (and year).

    Shaded bands are the interquartile range. The dashed gray line is the
    mean-yield baseline.

    Args:
        summary: Output of summarize_results.
        evaluation: "random" or "loyo".
        filename: Destination PNG name under Results/Graphs.
    """
    block = summary[summary["Evaluation"] == evaluation]
    if block.empty:
        return
    scalings = ordered_scalings(block)
    years = [None] if evaluation == "random" else sorted(block["Left_Out_Year"].dropna().unique())
    fig, axes = plt.subplots(
        len(scalings), len(years), figsize=(5.5 * len(years) + 1, 4.5 * len(scalings)),
        squeeze=False, sharey=True
    )
    ticks = [fraction * 100 for fraction in TRAIN_FRACTIONS]
    for r, scaling in enumerate(scalings):
        for c, year in enumerate(years):
            ax = axes[r][c]
            panel = block[block["Scaling"] == scaling]
            if year is not None:
                panel = panel[panel["Left_Out_Year"] == year]
            for scenario in list(SCENARIOS) + [BASELINE_NAME]:
                line = panel[panel["Scenario"] == scenario].sort_values("Train_Fraction")
                if line.empty:
                    continue
                x = line["Train_Fraction"] * 100
                color = SCENARIO_COLORS[scenario]
                style = "--" if scenario == BASELINE_NAME else "-"
                ax.plot(x, line["NRMSE_Median"], style, marker="o", color=color, label=scenario)
                ax.fill_between(x, line["NRMSE_Q25"], line["NRMSE_Q75"], color=color, alpha=0.15)
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{fraction:.0%}" for fraction in TRAIN_FRACTIONS])
            title = f"{scaling} scaling" if year is None else f"{scaling} scaling, {int(year)} held out"
            ax.set_title(title)
            ax.set_xlabel("Share of rice training rows used")
            ax.set_ylabel("Median NRMSE (%)")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def plot_full_data_boxplot(completed: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    """
    Box plot of one metric on random splits with all rice training rows.

    Args:
        completed: Completed result rows.
        metric: "NRMSE_Percent" or "RMSE".
        ylabel: Axis label.
        filename: Destination PNG name under Results/Graphs.
    """
    block = completed[(completed["Evaluation"] == "random") & (completed["Train_Fraction"] == 1.0)]
    if block.empty:
        return
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=block, x="Scenario", y=metric, hue="Scaling",
        order=list(SCENARIOS) + [BASELINE_NAME], hue_order=ordered_scalings(block),
    )
    plt.title("Random 80/20 splits, all rice training rows")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close()


def plot_loyo_bars(completed: pd.DataFrame, filename: str) -> None:
    """
    Median LOYO NRMSE per held-out year with all rice training rows.

    Error bars span the middle 50% of repeats.

    Args:
        completed: Completed result rows.
        filename: Destination PNG name under Results/Graphs.
    """
    block = completed[(completed["Evaluation"] == "loyo") & (completed["Train_Fraction"] == 1.0)]
    if block.empty:
        return
    block = block.assign(Left_Out_Year=block["Left_Out_Year"].astype(int))
    scalings = ordered_scalings(block)
    fig, axes = plt.subplots(1, len(scalings), figsize=(9 * len(scalings), 6), squeeze=False, sharey=True)
    for ax, scaling in zip(axes[0], scalings):
        sns.barplot(
            data=block[block["Scaling"] == scaling], x="Left_Out_Year", y="NRMSE_Percent",
            hue="Scenario", hue_order=list(SCENARIOS) + [BASELINE_NAME], palette=SCENARIO_COLORS,
            estimator="median", errorbar=("pi", 50), ax=ax,
        )
        ax.set_title(f"Leave one year out, {scaling} scaling")
        ax.set_xlabel("Held-out year")
        ax.set_ylabel("Median NRMSE (%)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def plot_source_spread(completed: pd.DataFrame, filename: str) -> None:
    """
    LOYO NRMSE of warm runs split by source network, with all rice training rows.

    Args:
        completed: Completed result rows.
        filename: Destination PNG name under Results/Graphs.
    """
    block = completed[
        (completed["Evaluation"] == "loyo") & (completed["Train_Fraction"] == 1.0)
        & completed["Source_Model"].notna()
    ]
    if block.empty:
        return
    block = block.assign(
        Left_Out_Year=block["Left_Out_Year"].astype(int),
        Source_Model=block["Source_Model"].astype(int).astype(str),
    )
    scalings = ordered_scalings(block)
    warm = [name for name, (_, is_warm) in SCENARIOS.items() if is_warm]
    source_order = [str(k) for k in sorted(block["Source_Model"].astype(int).unique())]
    fig, axes = plt.subplots(
        len(scalings), len(warm), figsize=(8 * len(warm), 5 * len(scalings)), squeeze=False, sharey=True
    )
    for r, scaling in enumerate(scalings):
        for c, scenario in enumerate(warm):
            ax = axes[r][c]
            panel = block[(block["Scaling"] == scaling) & (block["Scenario"] == scenario)]
            sns.boxplot(
                data=panel, x="Left_Out_Year", y="NRMSE_Percent", hue="Source_Model",
                hue_order=source_order, ax=ax,
            )
            ax.set_title(f"{scenario}, {scaling} scaling")
            ax.set_xlabel("Held-out year")
            ax.set_ylabel("NRMSE (%)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(title="Source network", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def format_comparisons(comparisons: pd.DataFrame, a: str, b: str) -> str:
    """
    Lay out one comparison as a text table: rows are fractions, columns are evaluations.

    Each cell is "median difference in points | share of pairs A wins", plus
    the corrected t-test p-value for random splits.

    Args:
        comparisons: Output of compare_configurations.
        a: First configuration (the A side).
        b: Second configuration (the B side).

    Returns:
        Text tables, one per held-constant value.
    """
    block = comparisons[(comparisons["A"] == a) & (comparisons["B"] == b)]
    if block.empty:
        return ""

    def cell(row: pd.Series) -> str:
        text = f"{row['Median_Diff']:+.2f} | {row['A_Better_Share']:.0%}"
        if row["Evaluation"] == "random" and not np.isnan(row["Corrected_t_p"]):
            text += f" | p={row['Corrected_t_p']:.2g}"
        return text

    block = block.assign(
        Column=np.where(
            block["Evaluation"] == "random", "random",
            "LOYO " + block["Left_Out_Year"].fillna(0).astype(int).astype(str)
        ),
        Cell=block.apply(cell, axis=1),
    )
    lines = []
    for constant, group in block.groupby("Held_Constant"):
        table = group.pivot(index="Train_Fraction", columns="Column", values="Cell")
        table.index = [f"{fraction:.0%}" for fraction in table.index]
        lines.extend([f"{a} minus {b} ({constant})", table.to_string(), ""])
    return "\n".join(lines)


def write_stats_report(
    data: dict,
    scalings: Sequence[str],
    params: dict,
    source_metrics: dict,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    sources: pd.DataFrame
) -> str:
    """
    Write the plain-text digest of a run and return it.

    Args:
        data: Output of load_data.
        scalings: Scaling modes that were run.
        params: Hyperparameters selected by Optuna.
        source_metrics: Soybean holdout metrics.
        summary: Output of summarize_results.
        comparisons: Scenario and scaling comparisons.
        sources: Output of summarize_sources.

    Returns:
        The report text.
    """
    lines = [
        f"Best architecture: {params}",
        f"Shared feature scaling: {', '.join(scalings)}",
        f"Common encoded features: {data['n_common_features']}",
        f"Soybean rows: {data['soy_rows']} | Rice rows: {data['rice_rows']}",
        (
            f"Soybean source holdout (seed {REPRESENTATIVE_SEED}): "
            f"RMSE={source_metrics['RMSE']:.4f}, NRMSE={source_metrics['NRMSE_Percent']:.4f}%"
        ),
        (
            f"Grid: {NUM_RANDOM_SPLITS} random splits and {NUM_LOYO_REPEATS} LOYO repeats per year, "
            f"training fractions {list(TRAIN_FRACTIONS)}, {NUM_SOURCE_MODELS} soybean source networks"
        ),
        "",
        "Cells: median NRMSE difference in points (negative means the first one is better) | share of",
        "pairs where the first one is better | corrected resampled t-test p (random splits only).",
        "LOYO repeats reuse the same rows, so they get no p-value.",
        "",
        "--- Warm vs cold ---",
        format_comparisons(comparisons, "Warm_Full", "Cold_Full"),
        format_comparisons(comparisons, "Warm_Common", "Cold_Common"),
        "--- Full vs shared-only features ---",
        format_comparisons(comparisons, "Warm_Full", "Warm_Common"),
        format_comparisons(comparisons, "Cold_Full", "Cold_Common"),
    ]
    if set(SHARED_SCALING_MODES) <= set(scalings):
        lines.append("--- per_crop vs soybean scaling ---")
        for scenario in SCENARIOS:
            lines.append(format_comparisons(comparisons, f"per_crop:{scenario}", f"soybean:{scenario}"))

    full = summary[summary["Train_Fraction"] == 1.0].copy()
    full["Column"] = np.where(
        full["Evaluation"] == "random", "random",
        "LOYO " + full["Left_Out_Year"].fillna(0).astype(int).astype(str)
    )
    medians = full.pivot_table(index=["Scaling", "Scenario"], columns="Column", values="NRMSE_Median")
    lines.extend(["--- Median NRMSE (%) with all rice training rows ---", medians.round(2).to_string(), ""])

    spread = sources[sources["Train_Fraction"] == 1.0]
    if not spread.empty:
        spread = spread.assign(Column=np.where(
            spread["Evaluation"] == "random", "random",
            "LOYO " + spread["Left_Out_Year"].fillna(0).astype(int).astype(str)
        ))
        ranges = spread.groupby(["Scaling", "Scenario", "Column"])["NRMSE_Median"].agg(["min", "max"])
        ranges = (ranges["min"].round(1).astype(str) + " to " + ranges["max"].round(1).astype(str)).unstack()
        lines.extend([
            "--- Warm median NRMSE (%) across source networks, all rice training rows (lowest to highest) ---",
            ranges.to_string(), "",
        ])

    report = "\n".join(lines) + "\n"
    with open(f"{RESULTS_DIR}/Errors/statistical_tests_{TIMESTAMP}.txt", "w") as handle:
        handle.write(report)
    return report


def main() -> None:
    """
    Run the whole experiment grid and write every table, plot, and model.
    """
    started = time.time()
    scalings = parse_scaling_modes(SHARED_SCALING_SETTING)
    folder_creation()
    data = load_data()
    years = sorted(int(year) for year in np.unique(data['rice_years']))
    tasks = build_tasks(scalings, years)
    print(f"Shared feature scaling: {', '.join(scalings)}")
    print(f"Planned rice fits: {len(tasks)} across training fractions {list(TRAIN_FRACTIONS)}")

    save_json(f"{RESULTS_DIR}/Models/run_config_{TIMESTAMP}.json", {
        "shared_scaling": list(scalings),
        "num_random_splits": NUM_RANDOM_SPLITS,
        "num_loyo_repeats_per_year": NUM_LOYO_REPEATS,
        "num_source_models": NUM_SOURCE_MODELS,
        "train_fractions": list(TRAIN_FRACTIONS),
        "test_fraction": TEST_FRACTION,
        "optuna_trials": OPTUNA_TRIALS,
        "num_workers": NUM_WORKERS,
        "global_seed": GLOBAL_SEED,
        "head_epochs": HEAD_EPOCHS,
        "max_fine_tune_epochs": MAX_FINE_TUNE_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE,
        "planned_rice_fits": len(tasks),
    })

    best_params = optimize_hyperparameters(data, n_trials=OPTUNA_TRIALS)
    print(f"[{(time.time() - started) / 60:.1f} min] Optuna done: {best_params}")

    source_metrics = evaluate_source_holdout(data, best_params)
    print(f"Soybean holdout: RMSE={source_metrics['RMSE']:.2f}, NRMSE={source_metrics['NRMSE_Percent']:.2f}%")

    print(f"Training {NUM_SOURCE_MODELS} soybean source networks...")
    source_weights = [
        map_source_weights(weights, data, k) for k, weights in enumerate(train_source_models(data, best_params))
    ]
    print(f"[{(time.time() - started) / 60:.1f} min] Source networks done")

    state = {
        "rice_inputs": {mode: build_rice_inputs(data, mode) for mode in scalings},
        "y_raw": data['rice_y_raw'],
        "years": data['rice_years'],
        "params": best_params,
        "source_weights": source_weights,
        "results_dir": RESULTS_DIR,
        "timestamp": TIMESTAMP,
    }
    print("Running rice fits...")
    results, diagnostics = run_rice_tasks(tasks, state)
    print(f"[{(time.time() - started) / 60:.1f} min] Rice fits done")

    results = pd.concat(
        [results, results_frame(mean_baseline_rows(tasks, data['rice_y_raw'], data['rice_years']))],
        ignore_index=True,
    )
    results[results["Evaluation"] == "random"].to_csv(
        f"{RESULTS_DIR}/Errors/random_split_metrics_{TIMESTAMP}.csv", index=False
    )
    results[results["Evaluation"] == "loyo"].to_csv(f"{RESULTS_DIR}/Errors/loyo_results_{TIMESTAMP}.csv", index=False)
    status_counts = results.groupby(["Evaluation", "Status"]).size().to_dict()
    print(f"Result status counts: {status_counts}")

    completed = results[results["Status"] == "Completed"]
    summary = summarize_results(completed)
    summary.to_csv(f"{RESULTS_DIR}/Errors/summary_{TIMESTAMP}.csv", index=False)

    comparisons = [compare_configurations(completed, "Scaling", "Scenario", SCENARIO_PAIRS)]
    if set(SHARED_SCALING_MODES) <= set(scalings):
        labelled = completed.assign(Config=completed["Scaling"] + ":" + completed["Scenario"])
        scaling_pairs = [(f"per_crop:{scenario}", f"soybean:{scenario}") for scenario in SCENARIOS]
        comparisons.append(compare_configurations(labelled, "Scenario", "Config", scaling_pairs))
    comparisons = pd.concat(comparisons, ignore_index=True)
    comparisons.to_csv(f"{RESULTS_DIR}/Errors/comparisons_{TIMESTAMP}.csv", index=False)

    sources = summarize_sources(completed)
    sources.to_csv(f"{RESULTS_DIR}/Errors/source_model_spread_{TIMESTAMP}.csv", index=False)

    for item in diagnostics:
        name = f"{item['Scaling']}_{item['Scenario']}"
        plot_predicted_vs_actual(
            item["y_true"], item["y_pred"],
            title=f"Predicted vs Measured Yield ({item['Scenario']}, {item['Scaling']} scaling)",
            filename=f"NN_Scatter_{name}_{TIMESTAMP}.png",
        )
        print(f"  {name}: scatter + model -> {item['model_path']}")

    plot_learning_curves(summary, "random", f"NN_LearningCurve_Random_{TIMESTAMP}.png")
    plot_learning_curves(summary, "loyo", f"NN_LearningCurve_LOYO_{TIMESTAMP}.png")
    plot_full_data_boxplot(completed, "NRMSE_Percent", "Normalized RMSE (%)", f"NN_Boxplot_{TIMESTAMP}.png")
    plot_full_data_boxplot(completed, "RMSE", "RMSE", f"NN_Boxplot_RMSE_{TIMESTAMP}.png")
    plot_loyo_bars(completed, f"NN_LOYO_NRMSE_{TIMESTAMP}.png")
    plot_source_spread(completed, f"NN_SourceSpread_LOYO_{TIMESTAMP}.png")

    report = write_stats_report(data, scalings, best_params, source_metrics, summary, comparisons, sources)
    print("Pipeline completed.")
    print(report)
    print(f"[{(time.time() - started) / 60:.1f} min] All outputs written")


if __name__ == '__main__':
    main()
