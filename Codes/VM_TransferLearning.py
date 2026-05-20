from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
import seaborn as sns
from scipy.stats import ttest_ind
import warnings



# System Configuration
try:
    multiprocessing.set_start_method('spawn', force=True)
except: pass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

# Paths
PARENT_DIR = Path(__file__).parent
SOY_DATASET_FILE = str(PARENT_DIR.parent) + "/DATA/soybean_with_harvesting_only.csv"
RICE_DATASET_FILE = str(PARENT_DIR.parent) + "/DATA/rice_with_harvesting_only.csv"
RESULTS_DIR = str(PARENT_DIR.parent) + "/Results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Global Constants
NUM_ITERATIONS = 100
NUM_WORKERS = os.cpu_count()
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

def folder_creation() -> None :
    """
    Creates the necessary directory structure for storing results, models, and graphs.
    """
    for folder in ["Cleaned_Data", "Errors", "Models", "Graphs"] :
        os.makedirs(f"{RESULTS_DIR}/{folder}", exist_ok=True)

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


def clean_data(
    path_to_file: str,
    return_years: bool = False,
    dataset_label: str = "dataset"
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Cleans the input CSV by removing specific columns, handling missing values, and encoding categorical variables.

    Args:
        path_to_file: The system path to the CSV dataset.
        return_years: When True, also returns years aligned to cleaned rows.
        dataset_label: Human-readable crop or dataset name for error messages.

    Returns:
        A cleaned and pre-processed pandas DataFrame, optionally with aligned years.
    """
    raw_df = pd.read_csv(path_to_file).drop(columns=['Sl', 'GPS'], errors='ignore')
    year_series = resolve_year_series(raw_df, dataset_label) if return_years else None

    df = raw_df.dropna(axis=1, how='all').fillna(raw_df.median(numeric_only=True))
    df = pd.get_dummies(df, drop_first=True).dropna()
    file_name = Path(path_to_file).stem + "_cleaned.csv"
    cleaned_data = df.loc[:, (df != df.iloc[0]).any()]
    cleaned_data.to_csv(f"{RESULTS_DIR}/Cleaned_Data/{file_name}", index=False)

    if return_years:
        aligned_years = year_series.loc[cleaned_data.index].to_numpy(dtype=int)
        return cleaned_data, aligned_years

    return cleaned_data


def load_data() -> dict:
    """
    Robust data loading and feature alignment for Transfer Learning.
    Aligns common features between Soybean and Rice datasets.

    Returns:
        A dictionary containing scaled feature matrices, scaled targets, 
        raw targets, and the target scaler.
    """
    soy_p = SOY_DATASET_FILE
    rice_p = RICE_DATASET_FILE

    soy = clean_data(soy_p)
    rice, rice_years = clean_data(rice_p, return_years=True, dataset_label="Rice")
    
    common = sorted(list((set(soy.columns) & set(rice.columns)) - {'Yield'}))
    soy_only = sorted(list(set(soy.columns) - set(rice.columns)))
    rice_only = sorted(list(set(rice.columns) - set(soy.columns)))
    
    print(f"\n--- Column Comparison ---")
    print(f"Common Features ({len(common)}): {common[:5]}... (and more)")
    print(f"Soybean Only ({len(soy_only)}): {soy_only[:5]}...")
    print(f"Rice Only ({len(rice_only)}): {rice_only[:5]}...")
    print(f"------------------------\n")
    
    print(f"Features Identified: {len(common)} Common | Soy: {len(soy)} rows | Rice: {len(rice)} rows")

    scaler_soy_X = StandardScaler()
    scaler_rice_full_X = StandardScaler()

    soy_X_scaled = scaler_soy_X.fit_transform(soy[common])
    rice_X_comm_scaled = scaler_soy_X.transform(rice[common])

    rice_features_full = list(rice.drop(columns=['Yield']).columns)
    rice_X_full_scaled = scaler_rice_full_X.fit_transform(rice[rice_features_full])

    scaler_soy_y = StandardScaler()
    scaler_rice_y = StandardScaler()

    return {
        'soy_X': soy_X_scaled,
        'soy_y_z': scaler_soy_y.fit_transform(soy[['Yield']]).flatten(),
        'rice_X_full': rice_X_full_scaled,
        'rice_X_comm': rice_X_comm_scaled,
        'rice_y_z': scaler_rice_y.fit_transform(rice[['Yield']]).flatten(),
        'rice_y_raw': rice['Yield'].values,
        'rice_years': rice_years,
        'scaler_rice_y': scaler_rice_y,
        'input_dim_soy': len(common),
        'rice_features_full': rice_features_full,
        'common_features': common
    }

def build_and_train(X_train: np.ndarray, y_train: np.ndarray, input_dim:int, params:dict, weights:Optional[List[np.ndarray]]=None):
    """
    Constructs and trains a Neural Network. Supports Transfer Learning through 
    weight initialization and two-phase fine-tuning.

    Args:
        X_train: Training features.
        y_train: Training targets.
        input_dim: Number of input features.
        params: Dictionary containing 'n_layers', 'lr', and units per layer.
        weights: Optional pre-trained weights for transfer learning.

    Returns:
        A trained Keras Model object.
    """

    import tensorflow as tf

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for i in range(params['n_layers']):
        model.add(tf.keras.layers.Dense(params[f'n_units_l{i}'], activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    if weights:
        model.set_weights(weights)

        # Phase 1: Freeze Transfer Layers
        for layer in model.layers[:-1]: layer.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Phase 2: Fine-tune Entire Network
        for layer in model.layers: layer.trainable = True

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']/2), loss='mse')
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0, callbacks=[es])
    return model

def worker_task(
    seed: int, 
    X: np.ndarray, 
    y_z: np.ndarray, 
    dim: int, 
    params: dict, 
    weights: Optional[List[np.ndarray]], 
    scaler_y: StandardScaler, 
    y_raw: np.ndarray
) -> float:
    """
    A worker function for parallel processing to evaluate model performance over different random splits.

    Returns:
        Normalized Root Mean Squared Error (NRMSE) as a percentage.
    """
    try:
        Xt, Xv, yt, _ = train_test_split(X, y_z, test_size=0.2, random_state=seed)
        _, _, _, yv_raw = train_test_split(X, y_raw, test_size=0.2, random_state=seed)
        
        m = build_and_train(Xt, yt, dim, params, weights)
        pz = m.predict(Xv, verbose=0).reshape(-1, 1)
        praw = scaler_y.inverse_transform(pz).flatten()

        return (np.sqrt(mean_squared_error(yv_raw, praw)) / np.mean(yv_raw)) * 100
    except: 
        return np.nan


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


def evaluate_loyo_fold(
    iteration: int,
    scenario_name: str,
    left_out_year: int,
    X: np.ndarray,
    y_z: np.ndarray,
    y_raw: np.ndarray,
    years: np.ndarray,
    input_dim: int,
    params: dict,
    weights: Optional[List[np.ndarray]],
    scaler_y: StandardScaler
) -> Dict[str, Union[str, int, float]]:
    """
    Trains on all years except one and evaluates on the held-out year.

    Args:
        scenario_name: Name of the transfer learning scenario.
        left_out_year: Year excluded from training and used for testing.
        X: Feature matrix for the scenario.
        y_z: Standardized target values.
        y_raw: Raw target values.
        years: Year labels aligned to rows in X/y.
        input_dim: Number of input features.
        params: Hyperparameters selected by Optuna.
        weights: Optional transfer weights.
        scaler_y: Target scaler used to invert standardized predictions.

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
        model = build_and_train(X[train_mask], y_z[train_mask], input_dim, params, weights)
        pred_z = model.predict(X[test_mask], verbose=0).reshape(-1, 1)
        pred_raw = scaler_y.inverse_transform(pred_z).flatten()
        metrics = calculate_error_metrics(y_raw[test_mask], pred_raw)
        result.update(metrics)
    except Exception as exc:
        result["Status"] = "Failed"
        result["Error"] = str(exc)

    return result


def plot_loyo_results(loyo_df: pd.DataFrame) -> None:
    """
    Saves a year-by-year NRMSE comparison plot for completed LOYO folds.

    Args:
        loyo_df: DataFrame returned by run_loyo_evaluation.
    """
    completed_df = loyo_df[loyo_df["Status"] == "Completed"].copy()
    if completed_df.empty:
        print("LOYO plotting skipped: no completed folds.")
        return

    plt.figure(figsize=(12, 7))
    sns.barplot(data=completed_df, x="Left_Out_Year", y="NRMSE_Percent", hue="Scenario")
    plt.title("Leave-One-Year-Out Transfer Learning Evaluation (100 Iterations)")
    plt.xlabel("Held-Out Year")
    plt.ylabel("Normalized RMSE (%)")
    plt.grid(True, axis="y")
    plt.legend(title="Scenario")
    plt.tight_layout()

    loyo_plot_filename = f"NN_LOYO_NRMSE_{TIMESTAMP}.png"
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", loyo_plot_filename), dpi=300)
    plt.show()


def run_loyo_evaluation(
    data: Dict,
    scenarios: List[Tuple[str, np.ndarray, int, Optional[List[np.ndarray]]]],
    best_params: Dict[str, Union[int, float]]
) -> pd.DataFrame:
    """
    Runs leave-one-year-out evaluation after the random-split benchmark.

    Args:
        data: Loaded data dictionary containing rice features, targets, scaler, and years.
        scenarios: Transfer learning scenarios to evaluate.
        best_params: Hyperparameters selected for the neural network.

    Returns:
        DataFrame containing one row per scenario and held-out year.
    """
    years = np.asarray(data['rice_years'], dtype=int)
    unique_years = sorted(np.unique(years))

    if len(unique_years) < 2:
        raise ValueError("LOYO requires at least two years of target-crop data.")

    print(f"\nStarting LOYO Evaluation across {len(unique_years)} years: {unique_years}")
    rows = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futs = []
        for name, X, dim, weights in scenarios:
            for left_out_year in unique_years:
                for i in range(NUM_ITERATIONS):
                    futs.append(executor.submit(
                        evaluate_loyo_fold,
                        i,
                        name,
                        left_out_year,
                        X,
                        data['rice_y_z'],
                        data['rice_y_raw'],
                        years,
                        dim,
                        best_params,
                        weights,
                        data['scaler_rice_y']
                    ))
        
        print("Processing LOYO Iterations...")
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                res = f.result()
                rows.append(res)
            except Exception as e:
                print(f"Failure in 1 LOYO Process with exception: {e}")

    loyo_df = pd.DataFrame(rows)
    loyo_results_path = f"{RESULTS_DIR}/Errors/loyo_results_{TIMESTAMP}.csv"
    loyo_df.to_csv(loyo_results_path, index=False)
    plot_loyo_results(loyo_df)

    print("\n--- LOYO Performance by Scenario (NRMSE %) ---")
    completed_df = loyo_df[loyo_df["Status"] == "Completed"]
    if completed_df.empty:
        print("No completed LOYO folds.")
    else:
        for scenario_name, scenario_df in completed_df.groupby("Scenario"):
            print(f"{scenario_name:12}: median={scenario_df['NRMSE_Percent'].median():.2f}%")

    return loyo_df
    
def optimize_hyperparameters(
    data: Dict, 
    n_trials: int = 20, 
    n_jobs: int = NUM_WORKERS
) -> Dict[str, Union[int, float]]:
    """
    Performs hyperparameter optimization using Optuna to find the best 
    architecture and learning rate for the Soybean base model.

    Args:
        data: Dictionary containing 'soy_X', 'soy_y_z', and 'input_dim_soy'.
        n_trials: The number of optimization trials to run.
        n_jobs: Number of parallel jobs for optimization.

    Returns:
        A dictionary containing the best discovered hyperparameters 
        (n_layers, lr, and units per layer).
    """
    
    def objective(t: optuna.Trial) -> float:
        """
        Internal objective function for Optuna trial evaluation.
        """
        p = {
            'n_layers': t.suggest_int('n_layers', 1, 3), 
            'lr': t.suggest_float('lr', 1e-4, 1e-3, log=True)
        }
        
        for i in range(p['n_layers']):
            p[f'n_units_l{i}'] = t.suggest_int(f'n_units_l{i}', 32, 128)
            
        Xt, Xv, yt, yv = train_test_split(
            data['soy_X'], 
            data['soy_y_z'], 
            test_size=0.2
        )
        
        m = build_and_train(Xt, yt, data['input_dim_soy'], p)
        
        score = float(m.evaluate(Xv, yv, verbose=0))
        return score

    print("Starting Optuna Optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    import json
    os.makedirs(f"{RESULTS_DIR}/Models", exist_ok=True)
    with open(f"{RESULTS_DIR}/Models/optuna_best_params_{TIMESTAMP}.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    return study.best_params


if __name__ == '__main__':
    folder_creation()
    data = load_data()
    
    best_params = optimize_hyperparameters(data, n_trials=50)

    print("Training Base Soybean Model...")
    base_m = build_and_train(data['soy_X'], data['soy_y_z'], data['input_dim_soy'], best_params)
    w_soy = base_m.get_weights()

    # Construct Transfer Weights for Full-Feature Rice Model
    # We must map Soy features (sorted alphabetically) to their correct positions in the Rice Full set
    w0_rice_full = np.random.normal(scale=0.01, size=(data['rice_X_full'].shape[1], w_soy[0].shape[1]))
    
    rice_cols = data['rice_features_full']
    common_cols = data['common_features']
    
    # Map each common feature to its index in the full rice dataset
    for i, col_name in enumerate(common_cols):
        rice_idx = rice_cols.index(col_name)
        w0_rice_full[rice_idx, :] = w_soy[0][i, :]
        
    weights_full = [w0_rice_full] + w_soy[1:]

    scenarios = [
        ("Warm_Full", data['rice_X_full'], data['rice_X_full'].shape[1], weights_full),
        ("Cold_Full", data['rice_X_full'], data['rice_X_full'].shape[1], None),
        ("Warm_Common", data['rice_X_comm'], data['input_dim_soy'], w_soy),
        ("Cold_Common", data['rice_X_comm'], data['input_dim_soy'], None)
    ]

    final_metrics = {s[0]: [] for s in scenarios}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for name, X, dim, weights in scenarios:
            print(f"Processing Scenario: {name}")
            futs = [executor.submit(worker_task, i, X, data['rice_y_z'], dim, best_params, weights, data['scaler_rice_y'], data['rice_y_raw']) for i in range(NUM_ITERATIONS)]
            for f in tqdm(as_completed(futs), total=NUM_ITERATIONS):
                try :
                    res = f.result()
                    if not np.isnan(res): final_metrics[name].append(res)
                except :
                    print(f"Failure in 1 Process with exception: {f.exception()}")
                    pass

    combined_df = pd.DataFrame.from_dict(final_metrics, orient='index').transpose()
    combined_df.to_csv(f"{RESULTS_DIR}/Errors/final_results_{TIMESTAMP}.csv")
    loyo_df = run_loyo_evaluation(data, scenarios, best_params)

    ## Boxplot
    plt.figure(figsize=(12, 7))
    plt.boxplot([combined_df[col] for col in combined_df.columns],
                patch_artist=True, tick_labels=combined_df.columns)
    plt.title(f'NN Transfer Learning')
    plt.ylabel('Normalized RMSE (%)')
    plt.grid(True)
    medians = [combined_df[col].median() for col in combined_df.columns]
    for i, median in enumerate(medians, start=1):
        plt.text(i, median, f'{median:.2f}', ha='center', va='bottom', fontsize=10, color='blue')

    boxplot_filename = f"NN_Boxplot_{TIMESTAMP}.png"
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", boxplot_filename), dpi=300)
    plt.show()

    plt.figure(figsize=(12, 7))
    for col in combined_df.columns:
        sns.kdeplot(combined_df[col], label=col, fill=True, alpha=0.3)
    plt.title(f'Distribution of Normalized RMSE')
    plt.xlabel('Normalized RMSE (%)')
    plt.legend()
    plt.grid(True, axis='y')

    density_filename = f"NN_Density_{TIMESTAMP}.png"
    plt.savefig(os.path.join(f"{RESULTS_DIR}/Graphs", density_filename), dpi=300)
    plt.show()

    
    print("\n--- Final Performance Medians (NRMSE %) ---")
    for k, v in final_metrics.items():
        print(f"{k:12}: {np.median(v):.2f}%")
    
    t_warm_full_cold_full = ttest_ind(final_metrics["Warm_Full"], final_metrics["Cold_Full"])
    t_warm_common_cold_common = ttest_ind(final_metrics["Warm_Common"], final_metrics["Cold_Common"])
    t_warm_common_warm_full = ttest_ind(final_metrics["Warm_Common"], final_metrics["Warm_Full"])

    stats_output = (
        f"Best architecture: {best_params}\n\n"
        f"--- Statistical Test Results (Independent Two-Sample T-test on NRMSE) ---\n"
        f"Warm (Full) vs Cold (Full): t={t_warm_full_cold_full.statistic:.2f}, p={t_warm_full_cold_full.pvalue:.4e}\n"
        f"Warm (Common) vs Cold (Common): t={t_warm_common_cold_common.statistic:.2f}, p={t_warm_common_cold_common.pvalue:.4e}\n"
        f"Warm (Common) vs Warm (Full): t={t_warm_common_warm_full.statistic:.2f}, p={t_warm_common_warm_full.pvalue:.4e}\n"
    )

    print("Pipeline completed.")
    print(stats_output)

    with open(f"{RESULTS_DIR}/Errors/statistical_tests_{TIMESTAMP}.txt", "w") as f:
        f.write(stats_output)