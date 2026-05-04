from pathlib import Path
from typing import Dict, List, Optional, Union
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
SOY_DATASET_FILE = str(PARENT_DIR) + "/Data/Processed/Soybean/soybean_with_harvesting_only.csv"
RICE_DATASET_FILE = str(PARENT_DIR) + "/Data/Processed/Rice/rice_with_harvesting_only.csv"
RESULTS_DIR = str(PARENT_DIR) + "/Results/NN_TransferLearning"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Global Constants
NUM_ITERATIONS = 100
NUM_WORKERS = os.cpu_count()

def folder_creation() -> None :
    """
    Creates the necessary directory structure for storing results, models, and graphs.
    """
    for folder in ["Cleaned_Data", "Errors", "Models", "Graphs"] :
        os.makedirs(f"{RESULTS_DIR}/{folder}", exist_ok=True)

def clean_data(path_to_file: str) -> pd.DataFrame:
    """
    Cleans the input CSV by removing specific columns, handling missing values, and encoding categorical variables.

    Args:
        path_to_file: The system path to the CSV dataset.

    Returns:
        A cleaned and pre-processed pandas DataFrame.
    """
    df = pd.read_csv(path_to_file).drop(columns=['Sl', 'GPS'], errors='ignore')
    df = df.dropna(axis=1, how='all').fillna(df.median(numeric_only=True))
    df = pd.get_dummies(df, drop_first=True).dropna()
    file_name = path_to_file.replace(".csv", f"_cleaned.csv").split("/")[-1]
    cleaned_data = df.loc[:, (df != df.iloc[0]).any()]
    cleaned_data.to_csv(f"{RESULTS_DIR}/Cleaned_Data/{file_name}", index=False)
    return cleaned_data


def load_data() -> dict:
    """
    Robust data loading and feature alignment for Transfer Learning.
    Aligns common features between Soybean and Rice datasets.

    Returns:
        A dictionary containing scaled feature matrices, scaled targets, 
        raw targets, and the target scaler.
    """
    main_folder = os.path.dirname(os.path.realpath(__file__))
    soy_p = os.path.join(main_folder, SOY_DATASET_FILE)
    rice_p = os.path.join(main_folder, RICE_DATASET_FILE)

    soy = clean_data(soy_p)
    rice = clean_data(rice_p)
    
    common = sorted(list(set(soy.columns) & set(rice.columns) - {'Yield'}))
    
    print(f"Features Identified: {len(common)} Common | Soy: {len(soy)} rows | Rice: {len(rice)} rows")

    scaler_soy_X = StandardScaler()
    scaler_rice_full_X = StandardScaler()

    soy_X_scaled = scaler_soy_X.fit_transform(soy[common])
    rice_X_comm_scaled = scaler_soy_X.transform(rice[common])

    rice_X_full_scaled = scaler_rice_full_X.fit_transform(rice.drop(columns=['Yield']))

    scaler_soy_y = StandardScaler()
    scaler_rice_y = StandardScaler()

    return {
        'soy_X': soy_X_scaled,
        'soy_y_z': scaler_soy_y.fit_transform(soy[['Yield']]).flatten(),
        'rice_X_full': rice_X_full_scaled,
        'rice_X_comm': rice_X_comm_scaled,
        'rice_y_z': scaler_rice_y.fit_transform(rice[['Yield']]).flatten(),
        'rice_y_raw': rice['Yield'].values,
        'scaler_rice_y': scaler_rice_y,
        'input_dim_soy': len(common)
    }

def build_and_train(X_train: np.ndarray, y_train: np.ndarray, input_dim:int, params:dict, weights:Optional[list[np.ndarray]]=None):
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
    
    return study.best_params


if __name__ == '__main__':
    folder_creation()
    data = load_data()
    
    best_params = optimize_hyperparameters(data)

    print("Training Base Soybean Model...")
    base_m = build_and_train(data['soy_X'], data['soy_y_z'], data['input_dim_soy'], best_params)
    w_soy = base_m.get_weights()

    w0_rice_full = np.random.normal(scale=0.01, size=(data['rice_X_full'].shape[1], w_soy[0].shape[1]))
    w0_rice_full[:data['input_dim_soy'], :] = w_soy[0]
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
            for f in tqdm(as_completed(futs), total=100):
                try :
                    res = f.result()
                    if not np.isnan(res): final_metrics[name].append(res)
                except :
                    print(f"Failure in 1 Process with exception: {f.exception()}")
                    pass

    combined_df = pd.DataFrame.from_dict(final_metrics, orient='index').transpose()
    combined_df.to_csv(f"{RESULTS_DIR}/Errors/final_results_{TIMESTAMP}.csv")

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
    plt.savefig(os.path.join(F"{RESULTS_DIR}/Graphs", boxplot_filename), dpi=300)
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

    print("Pipeline completed.")
    print(f"Best architecture: {best_params}")
    print("\n--- Statistical Test Results (Independent Two-Sample T-test on NRMSE) ---")
    print(f"Warm (Full) vs Cold (Full): t={t_warm_full_cold_full.statistic:.2f}, p={t_warm_full_cold_full.pvalue:.4e}")
    print(f"Warm (Common) vs Cold (Common): t={t_warm_common_cold_common.statistic:.2f}, p={t_warm_common_cold_common.pvalue:.4e}")
    print(f"Warm (Common) vs Warm (Full): t={t_warm_common_warm_full.statistic:.2f}, p={t_warm_common_warm_full.pvalue:.4e}")