"""
Warm vs cold start on sparse rice data: accuracy by training size, and how fast each start settles.

For each random 80/20 rice split the 20% test rows stay fixed, and networks
are trained on a growing share of the 80% pool (TRAIN_FRACTIONS). A smaller
share is always a subset of a larger one. Every fit records the test error
after every epoch, so one run shows how accuracy grows with rice data and how
many epochs each starting point needs.

Starting points, on the full rice feature set and on the features shared with
soybean:

    Warm            Soybean source network weights, whole network fine-tuned
                    (the schedule of VM_TransferLearning.py).
    Cold            Random initialization.
    Cold_SmallInit  Random, except that rice-only input rows start at the same
                    small scale as in Warm (full feature set only). Separates
                    soybean knowledge from the effect of starting with
                    near-zero weights on the rice-only columns.
    Shuffled        Weights of a soybean network trained on shuffled soybean
                    yields. Same architecture, scaling, seed, and weight sizes
                    as Warm, but no real soybean knowledge.
    Frozen          Soybean weights with the middle hidden layers held fixed.
                    The first layer (so rice-only columns can still be used)
                    and the output layer train.

Train_Fraction 0 is a zero-shot row: each start is scored on the test rows
before any rice training. Its features are scaled on the pool rows and its
predictions are put into yield units with the pool's yield mean and spread,
so it shows where each start begins, not a usable model.

Warm, Frozen, and Shuffled fits on split i start from source network
i % NUM_SOURCE_MODELS. All fits on a split share the rows and the seed, so
starts are compared split by split.

Data loading, scaling, row selection, and statistics come from
VM_TransferLearning.py, so numbers are comparable with that pipeline.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Imported before NumPy and TensorFlow: it sets the per-process thread limits and the spawn start method.
import VM_TransferLearning as tl  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Experiment grid
NUM_SPLITS = 50
NUM_SOURCE_MODELS = 5
TRAIN_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
ZERO_SHOT = 0.0
MIN_TRAIN_ROWS = 20
CHECKPOINT_EVERY = 500
# Test error within this many NRMSE points of the final value counts as settled
SETTLED_TOLERANCE = 1.0
# Offsets the yield-shuffling seed away from the network seeds
SHUFFLE_SEED_OFFSET = 1000

# Comma-separated scaling modes for rice's shared features (see VM_TransferLearning.py)
SCALING_SETTING = os.environ.get("VM_SHARED_SCALING", "per_crop")
# Optional JSON with Optuna's best parameters from an earlier run; empty runs Optuna
PARAMS_FILE = os.environ.get("VM_BEST_PARAMS_FILE", "").strip()

FEATURE_SETS = ("full", "common")
STARTS = ("Warm", "Cold", "Cold_SmallInit", "Shuffled", "Frozen")
FULL_ONLY_STARTS = ("Cold_SmallInit",)
# Start -> which soybean source weights it uses (None means random)
START_SOURCE = {"Warm": "real", "Frozen": "real", "Shuffled": "shuffled", "Cold": None, "Cold_SmallInit": None}
BASELINE = tl.BASELINE_NAME
START_COLORS = {
    "Warm": "tab:blue",
    "Cold": "tab:orange",
    "Cold_SmallInit": "tab:red",
    "Shuffled": "tab:purple",
    "Frozen": "tab:green",
    BASELINE: "gray",
}
# Pairs compared split by split. Differences are always first minus second.
START_PAIRS = (
    ("Warm", "Cold"),
    ("Warm", "Cold_SmallInit"),
    ("Warm", "Shuffled"),
    ("Frozen", "Warm"),
    ("Frozen", "Cold"),
    ("Cold_SmallInit", "Cold"),
)
EPOCH_PLOT_FRACTIONS = (0.01, 0.05, 0.25, 1.0)
PREFIX = "convergence"
FIT_KEYS = ["Scaling", "Feature_Set", "Start", "Train_Fraction", "Split"]

# Set in each worker process by init_worker
_STATE: Dict = {}


def init_worker(state: dict) -> None:
    """
    Store read-only inputs in a worker process once, so tasks stay small.

    Args:
        state: Matrices, targets, parameters, and weights the tasks read.
    """
    global _STATE
    _STATE = state


def feature_label(feature_set: str) -> str:
    """
    Display name of a feature set.

    Args:
        feature_set: "full" or "common".

    Returns:
        "Full" or "Common".
    """
    return feature_set.capitalize()


def starts_for(feature_set: str) -> List[str]:
    """
    Starting points that apply to a feature set.

    Args:
        feature_set: "full" or "common".

    Returns:
        Start names. Cold_SmallInit needs rice-only columns, so it is full only.
    """
    return [start for start in STARTS if feature_set == "full" or start not in FULL_ONLY_STARTS]


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation that returns NaN instead of warning on constant input.

    Args:
        a: First vector.
        b: Second vector of the same length.

    Returns:
        Correlation coefficient, or NaN if either vector is constant.
    """
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_best_params(data: dict) -> dict:
    """
    Read Optuna's best parameters from PARAMS_FILE, or run Optuna when it is not set.

    Args:
        data: Output of tl.load_data.

    Returns:
        Hyperparameters with n_layers, lr, and n_units_l{i}.

    Raises:
        FileNotFoundError: If PARAMS_FILE is set but missing.
        KeyError: If the file lacks the expected parameters.
    """
    if not PARAMS_FILE:
        return tl.optimize_hyperparameters(data, n_trials=tl.OPTUNA_TRIALS)
    with open(PARAMS_FILE, encoding="utf-8") as handle:
        payload = json.load(handle)
    params = payload.get("best_params", payload)
    missing = [key for key in ["n_layers", "lr"] if key not in params]
    missing += [f"n_units_l{i}" for i in range(int(params.get("n_layers", 0))) if f"n_units_l{i}" not in params]
    if missing:
        raise KeyError(f"{PARAMS_FILE} is missing {missing}")
    print(f"Using parameters from {PARAMS_FILE}: {params}")
    return params


def train_source_task(task: dict) -> Tuple[str, int, List[np.ndarray], dict]:
    """
    Train one soybean source network on all soybean rows, with real or shuffled yields.

    Real and shuffled source k share the seed, so they start from the same
    weights and see rows in the same order. Only the yields differ.

    Args:
        task: {"Kind": "real" or "shuffled", "Index": source number}.

    Returns:
        Kind, source number, trained weights, and a summary row with the
        in-sample correlation against the real soybean yields.
    """
    state = _STATE
    kind, index = task["Kind"], task["Index"]
    seed = tl.GLOBAL_SEED + index
    y = state["soy_y_z"]
    if kind == "shuffled":
        y = np.random.default_rng(tl.GLOBAL_SEED + SHUFFLE_SEED_OFFSET + index).permutation(y)
    order = np.random.default_rng(seed).permutation(len(y))
    model = tl.build_and_train(
        state["soy_X"][order], y[order], state["soy_X"].shape[1], state["params"], seed=seed
    )
    prediction = model.predict(state["soy_X"], verbose=0).ravel()
    stem = "soybean_source" if kind == "real" else "soybean_shuffled_source"
    path = tl.save_trained_model(model, f"{PREFIX}_{stem}_{index}", state["results_dir"], state["timestamp"])
    info = {
        "Kind": kind,
        "Source_Model": index,
        "InSample_r_vs_real_yield": safe_pearson(prediction, state["soy_y_z"]),
        "Model_Path": path,
    }
    return kind, index, model.get_weights(), info


def train_sources(data: dict, params: dict, timestamp: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Train the real and shuffled soybean source networks in parallel and map them onto rice.

    Args:
        data: Output of tl.load_data.
        params: Hyperparameters.
        timestamp: Run timestamp for saved models.

    Returns:
        {(kind, index): {"common": weights, "full": weights}}, and one
        summary row per source network.
    """
    state = {
        "soy_X": data["soy_X"], "soy_y_z": data["soy_y_z"], "params": params,
        "results_dir": tl.RESULTS_DIR, "timestamp": timestamp,
    }
    tasks = [{"Kind": kind, "Index": k} for kind in ("real", "shuffled") for k in range(NUM_SOURCE_MODELS)]
    weights, rows = {}, []
    workers = max(1, min(len(tasks), tl.NUM_WORKERS))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(state,)) as executor:
        for future in as_completed([executor.submit(train_source_task, task) for task in tasks]):
            kind, index, source_weights, info = future.result()
            weights[(kind, index)] = tl.map_source_weights(source_weights, data, index)
            rows.append(info)
            print(f"  {kind} source {index}: in-sample r={info['InSample_r_vs_real_yield']:.3f} -> {info['Model_Path']}")
    return weights, pd.DataFrame(rows).sort_values(["Kind", "Source_Model"]).reset_index(drop=True)


def build_network(tf, input_dim: int, params: dict):
    """
    Build the tuned architecture (same layers as tl.build_and_train).

    Args:
        tf: The imported tensorflow module.
        input_dim: Number of input features.
        params: Hyperparameters.

    Returns:
        An uncompiled Keras Sequential model.
    """
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for i in range(params["n_layers"]):
        model.add(tf.keras.layers.Dense(params[f"n_units_l{i}"], activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    return model


def initial_weights(task: dict, state: dict, model) -> Optional[List[np.ndarray]]:
    """
    Starting weights for one fit.

    Args:
        task: Fit description (Start, Feature_Set, Source_Model, Split).
        state: Worker state with the mapped source weights and rice-only indices.
        model: Freshly built and seeded network, used for Cold_SmallInit.

    Returns:
        Weights to load, or None to keep the random initialization.
    """
    start = task["Start"]
    kind = START_SOURCE[start]
    if kind is not None:
        return state["weights"][(kind, task["Source_Model"])][task["Feature_Set"]]
    if start == "Cold_SmallInit":
        weights = model.get_weights()
        rows = state["rice_only_idx"]
        rng = np.random.default_rng(tl.GLOBAL_SEED + task["Split"])
        weights[0][rows, :] = rng.normal(scale=tl.RICE_ONLY_INIT_SCALE, size=(len(rows), weights[0].shape[1]))
        return weights
    return None


def fine_tune_mask(start: str, n_layers: int) -> List[bool]:
    """
    Which Dense layers train in the fine-tuning phase.

    Args:
        start: Starting point name.
        n_layers: Number of hidden layers.

    Returns:
        One flag per Dense layer (hidden layers, then output). Frozen keeps
        only the first hidden layer and the output layer trainable.
    """
    if start != "Frozen":
        return [True] * (n_layers + 1)
    return [i == 0 or i == n_layers for i in range(n_layers + 1)]


def epoch_callback(tf, on_epoch: Callable[[int, dict], None]):
    """
    Keras callback that hands each finished epoch to `on_epoch`.

    Args:
        tf: The imported tensorflow module.
        on_epoch: Called with the epoch index within the fit() call and its logs.

    Returns:
        A Keras Callback instance.
    """
    class _EpochTracker(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            on_epoch(epoch, logs or {})

    return _EpochTracker()


def train_and_track(
    X_train: np.ndarray,
    y_train_z: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    y_scaler,
    task: dict,
    state: dict
) -> Tuple[np.ndarray, List[dict]]:
    """
    Train one network with the shared schedule and score the test rows after every epoch.

    The schedule is tl.build_and_train's: HEAD_EPOCHS on the output layer,
    then fine-tuning at half the learning rate with early stopping on a 20%
    validation split of the training rows. Test scores are only recorded;
    they never steer training.

    Args:
        X_train: Scaled training features.
        y_train_z: Standardized training yield.
        X_test: Scaled test features.
        y_test_raw: Test yield in original units.
        y_scaler: Scaler that turns predictions back into yield units.
        task: Fit description.
        state: Worker state.

    Returns:
        Final test predictions in yield units (from the weights early
        stopping keeps), and one curve row per epoch, starting at epoch 0
        (before training).
    """
    import tensorflow as tf

    params = state["params"]
    tf.keras.utils.set_random_seed(int(task["Split"]))
    model = build_network(tf, X_train.shape[1], params)
    weights = initial_weights(task, state, model)
    if weights is not None:
        model.set_weights(weights)

    curve: List[dict] = []

    def record(phase: str, epoch: int, logs: dict) -> None:
        pred_raw = y_scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).ravel()
        curve.append({
            "Epoch": epoch, "Phase": phase,
            "Train_Loss": float(logs.get("loss", np.nan)), "Val_Loss": float(logs.get("val_loss", np.nan)),
            "Test_NRMSE": tl.calculate_error_metrics(y_test_raw, pred_raw)["NRMSE_Percent"],
        })

    record("start", 0, {})

    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]), loss="mse")
    model.fit(
        X_train, y_train_z, epochs=tl.HEAD_EPOCHS, batch_size=tl.BATCH_SIZE, verbose=0,
        callbacks=[epoch_callback(tf, lambda e, logs: record("head", e + 1, logs))],
    )

    for layer, trainable in zip(model.layers, fine_tune_mask(task["Start"], params["n_layers"])):
        layer.trainable = trainable
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"] / 2), loss="mse")
    # The tracker comes first so it scores each epoch before early stopping can restore weights.
    callbacks = [
        epoch_callback(tf, lambda e, logs: record("fine_tune", tl.HEAD_EPOCHS + e + 1, logs)),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=tl.EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),
    ]
    model.fit(
        X_train, y_train_z, validation_split=0.2, epochs=tl.MAX_FINE_TUNE_EPOCHS,
        batch_size=tl.BATCH_SIZE, verbose=0, callbacks=callbacks,
    )
    pred_raw = y_scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).ravel()
    return pred_raw, curve


def zero_shot(X_test: np.ndarray, y_scaler, task: dict, state: dict) -> np.ndarray:
    """
    Predict the test rows with a start's weights before any rice training.

    Args:
        X_test: Test features scaled on the pool rows.
        y_scaler: Yield scaler fit on the pool rows.
        task: Fit description.
        state: Worker state.

    Returns:
        Test predictions in yield units.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(task["Split"]))
    model = build_network(tf, X_test.shape[1], state["params"])
    weights = initial_weights(task, state, model)
    if weights is not None:
        model.set_weights(weights)
    return y_scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1)).ravel()


def curve_summary(curve: List[dict], final_nrmse: float) -> dict:
    """
    Condense an epoch curve into the numbers the report uses.

    Args:
        curve: Rows from train_and_track.
        final_nrmse: Test NRMSE of the kept model.

    Returns:
        Start, end-of-head, and kept-model epoch; epochs run; and the first
        epoch whose test error was within SETTLED_TOLERANCE of the final error.
    """
    frame = pd.DataFrame(curve)
    fine = frame[frame["Phase"] == "fine_tune"]
    head = frame[frame["Phase"] == "head"]
    settled = frame.loc[frame["Test_NRMSE"] <= final_nrmse + SETTLED_TOLERANCE, "Epoch"]
    return {
        "Start_NRMSE": float(frame.loc[frame["Phase"] == "start", "Test_NRMSE"].iloc[0]),
        "Head_NRMSE": float(head["Test_NRMSE"].iloc[-1]) if not head.empty else np.nan,
        "Kept_Epoch": int(fine.loc[fine["Val_Loss"].idxmin(), "Epoch"]) if fine["Val_Loss"].notna().any() else np.nan,
        "Epochs_Run": int(frame["Epoch"].max()),
        "Settled_Epoch": int(settled.iloc[0]) if not settled.empty else np.nan,
    }


def run_fit(task: dict) -> dict:
    """
    Run one fit (or zero-shot score) described by `task`.

    Args:
        task: Scaling, Feature_Set, Start, Scenario, Train_Fraction, Split, Source_Model.

    Returns:
        One result row. Completed training fits carry their epoch curve under "_curve".
    """
    state = _STATE
    feature_set = task["Feature_Set"]
    inputs = state["rice_inputs"][task["Scaling"]]
    X = inputs[f"X_{feature_set}"]
    fit_columns = inputs[f"{feature_set}_fit_idx"]
    y = state["y_raw"]
    fraction = task["Train_Fraction"]
    train_idx, test_idx = tl.select_rows("random", task["Split"], 1.0 if fraction == ZERO_SHOT else fraction, state["years"])

    row = dict(task)
    row.update({
        "Train_Rows": 0 if fraction == ZERO_SHOT else int(len(train_idx)), "Test_Rows": int(len(test_idx)),
        "RMSE": np.nan, "NRMSE_Percent": np.nan, "Pearson_r": np.nan,
        "Start_NRMSE": np.nan, "Head_NRMSE": np.nan, "Kept_Epoch": np.nan, "Epochs_Run": np.nan,
        "Settled_Epoch": np.nan, "Status": "Completed", "Error": "",
    })
    if fraction != ZERO_SHOT and len(train_idx) < MIN_TRAIN_ROWS:
        row.update(Status="Skipped", Error=f"Too few training rows ({len(train_idx)}).")
        return row

    try:
        Xt, Xv, yt_z, y_scaler = tl.prepare_split(X[train_idx], X[test_idx], y[train_idx], fit_columns)
        if fraction == ZERO_SHOT:
            pred_raw = zero_shot(Xv, y_scaler, task, state)
        else:
            pred_raw, curve = train_and_track(Xt, yt_z, Xv, y[test_idx], y_scaler, task, state)
        row.update(tl.calculate_error_metrics(y[test_idx], pred_raw))
        row["Pearson_r"] = safe_pearson(pred_raw, y[test_idx])
        if fraction == ZERO_SHOT:
            row["Start_NRMSE"] = row["NRMSE_Percent"]
        else:
            row.update(curve_summary(curve, row["NRMSE_Percent"]))
            row["_curve"] = curve
    except Exception as exc:
        row.update(Status="Failed", Error=str(exc))
    return row


def build_tasks(scalings: Sequence[str]) -> List[dict]:
    """
    List every fit in the grid, largest training shares first so long fits start early.

    Args:
        scalings: Shared-feature scaling modes to run.

    Returns:
        Task dictionaries for run_fit.
    """
    tasks = []
    for fraction in sorted(TRAIN_FRACTIONS + (ZERO_SHOT,), reverse=True):
        for scaling in scalings:
            for feature_set in FEATURE_SETS:
                for start in starts_for(feature_set):
                    for split in range(NUM_SPLITS):
                        tasks.append({
                            "Scaling": scaling,
                            "Feature_Set": feature_set,
                            "Start": start,
                            "Scenario": f"{start}_{feature_label(feature_set)}",
                            "Train_Fraction": float(fraction),
                            "Split": int(split),
                            "Source_Model": int(split % NUM_SOURCE_MODELS) if START_SOURCE[start] else None,
                        })
    return tasks


def run_fits(tasks: List[dict], state: dict, timestamp: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every fit in parallel, checkpointing result rows as they finish.

    Args:
        tasks: Output of build_tasks.
        state: Shared inputs handed to each worker once.
        timestamp: Run timestamp for the checkpoint file.

    Returns:
        Result rows and epoch curve rows (tagged with FIT_KEYS).
    """
    rows, curves = [], []
    checkpoint = f"{tl.RESULTS_DIR}/Errors/{PREFIX}_partial_results_{timestamp}.csv"
    with ProcessPoolExecutor(max_workers=tl.NUM_WORKERS, initializer=init_worker, initargs=(state,)) as executor:
        futures = [executor.submit(run_fit, task) for task in tasks]
        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
            try:
                row = future.result()
            except Exception as exc:
                print(f"Worker failed: {exc}")
                continue
            curve = row.pop("_curve", None)
            if curve is not None:
                keys = {key: row[key] for key in FIT_KEYS}
                curves.extend({**keys, **point} for point in curve)
            rows.append(row)
            if count % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    return pd.DataFrame(rows), pd.DataFrame(curves)


def baseline_rows(scalings: Sequence[str], y_raw: np.ndarray, years: np.ndarray) -> pd.DataFrame:
    """
    Score "predict the mean training yield" on the same rows as every fit.

    Args:
        scalings: Scaling modes run (the baseline is repeated per mode for plotting).
        y_raw: Rice yield in original units.
        years: Rice year per row.

    Returns:
        Rows with Start set to the baseline name and no feature set.
    """
    rows = []
    for fraction in (ZERO_SHOT,) + TRAIN_FRACTIONS:
        for split in range(NUM_SPLITS):
            train_idx, test_idx = tl.select_rows("random", split, 1.0 if fraction == ZERO_SHOT else fraction, years)
            prediction = np.full(len(test_idx), y_raw[train_idx].mean())
            for scaling in scalings:
                rows.append({
                    "Scaling": scaling, "Feature_Set": None, "Start": BASELINE, "Scenario": BASELINE,
                    "Train_Fraction": float(fraction), "Split": split, "Source_Model": None,
                    "Train_Rows": 0 if fraction == ZERO_SHOT else int(len(train_idx)), "Test_Rows": int(len(test_idx)),
                    **tl.calculate_error_metrics(y_raw[test_idx], prediction),
                    "Status": "Completed", "Error": "",
                })
    return pd.DataFrame(rows)


def summarize(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Medians and interquartile range per scaling, feature set, start, and training share.

    Args:
        completed: Completed rows, including the baseline.

    Returns:
        One row per configuration and training share.
    """
    grouped = completed.groupby(["Scaling", "Feature_Set", "Start", "Train_Fraction"], dropna=False)
    return grouped.agg(
        N=("NRMSE_Percent", "size"),
        Train_Rows=("Train_Rows", "median"),
        NRMSE_Median=("NRMSE_Percent", "median"),
        NRMSE_Q25=("NRMSE_Percent", lambda s: s.quantile(0.25)),
        NRMSE_Q75=("NRMSE_Percent", lambda s: s.quantile(0.75)),
        RMSE_Median=("RMSE", "median"),
        Pearson_r_Median=("Pearson_r", "median"),
        Start_NRMSE_Median=("Start_NRMSE", "median"),
        Head_NRMSE_Median=("Head_NRMSE", "median"),
        Kept_Epoch_Median=("Kept_Epoch", "median"),
        Settled_Epoch_Median=("Settled_Epoch", "median"),
        Epochs_Run_Median=("Epochs_Run", "median"),
    ).reset_index()


def compare_starts(completed: pd.DataFrame) -> pd.DataFrame:
    """
    Paired split-by-split comparisons of starting points at every training share.

    Zero-shot rows get medians and win shares but no p-values, since nothing
    was trained.

    Args:
        completed: Completed rows without the baseline.

    Returns:
        One row per scaling, feature set, pair, and training share, with the
        columns of tl.paired_statistics.
    """
    rows = []
    for (scaling, feature_set), block in completed.groupby(["Scaling", "Feature_Set"]):
        for a, b in START_PAIRS:
            left = block.loc[block["Start"] == a, ["Train_Fraction", "Split", "NRMSE_Percent", "Train_Rows", "Test_Rows"]]
            right = block.loc[block["Start"] == b, ["Train_Fraction", "Split", "NRMSE_Percent"]]
            if left.empty or right.empty:
                continue
            merged = left.merge(right, on=["Train_Fraction", "Split"], suffixes=("_A", "_B"))
            for fraction, group in merged.groupby("Train_Fraction"):
                rows.append({
                    "Scaling": scaling, "Feature_Set": feature_set, "A": a, "B": b, "Train_Fraction": fraction,
                    **tl.paired_statistics(group, "random" if fraction > ZERO_SHOT else "zero_shot"),
                })
    if not rows:
        return pd.DataFrame(columns=[
            "Scaling", "Feature_Set", "A", "B", "Train_Fraction",
            "N_Pairs", "Median_A", "Median_B", "Median_Diff", "A_Better_Share",
            "Wilcoxon_p", "Corrected_t_p",
        ])
    return pd.DataFrame(rows)


def data_efficiency(summary: pd.DataFrame) -> pd.DataFrame:
    """
    How many rows a cold network needs to match each start's median error.

    Cold's median error is made non-increasing in rows (running minimum), and
    log(rows) is interpolated linearly against error.

    Args:
        summary: Output of summarize.

    Returns:
        One row per scaling, feature set, start, and training share, with the
        rows cold needs and the ratio to the rows the start used. NaN means
        the start's error is outside cold's range.
    """
    rows = []
    trained = summary[summary["Train_Fraction"] > ZERO_SHOT]
    for (scaling, feature_set), block in trained.groupby(["Scaling", "Feature_Set"]):
        cold = block[block["Start"] == "Cold"].sort_values("Train_Rows")
        if cold.empty:
            continue
        cold_rows = cold["Train_Rows"].to_numpy(dtype=float)
        cold_error = np.minimum.accumulate(cold["NRMSE_Median"].to_numpy(dtype=float))
        for _, line in block[block["Start"] != "Cold"].iterrows():
            error = line["NRMSE_Median"]
            inside = cold_error.min() <= error <= cold_error.max()
            needed = float(np.exp(np.interp(-error, -cold_error, np.log(cold_rows)))) if inside else np.nan
            rows.append({
                "Scaling": scaling, "Feature_Set": feature_set, "Start": line["Start"],
                "Train_Fraction": line["Train_Fraction"], "Train_Rows": line["Train_Rows"],
                "NRMSE_Median": error, "Cold_Rows_Needed": needed,
                "Ratio": needed / line["Train_Rows"] if inside else np.nan,
            })
    return pd.DataFrame(rows)


def fraction_labels(fractions: Sequence[float]) -> List[str]:
    """
    Axis labels for training shares.

    Args:
        fractions: Training shares, including ZERO_SHOT.

    Returns:
        Labels such as "0%", "1%", "100%".
    """
    return [f"{fraction:.0%}" for fraction in fractions]


def plot_learning_curves(summary: pd.DataFrame, filename: str) -> None:
    """
    Median test NRMSE by training share, one panel per scaling and feature set.

    Shares are evenly spaced, with the zero-shot point first. Bands are the
    interquartile range. The dashed gray line is the mean-yield baseline.

    Args:
        summary: Output of summarize.
        filename: PNG name under Results/Graphs.
    """
    scalings = [mode for mode in tl.SHARED_SCALING_MODES if mode in set(summary["Scaling"])]
    fractions = (ZERO_SHOT,) + TRAIN_FRACTIONS
    position = {fraction: i for i, fraction in enumerate(fractions)}
    fig, axes = plt.subplots(len(scalings), len(FEATURE_SETS), figsize=(14, 5 * len(scalings)), squeeze=False)
    for r, scaling in enumerate(scalings):
        for c, feature_set in enumerate(FEATURE_SETS):
            ax = axes[r][c]
            panel = summary[summary["Scaling"] == scaling]
            for start in starts_for(feature_set) + [BASELINE]:
                selector = panel["Start"] == start
                if start != BASELINE:
                    selector &= panel["Feature_Set"] == feature_set
                line = panel[selector].sort_values("Train_Fraction")
                if line.empty:
                    continue
                x = line["Train_Fraction"].map(position)
                style = "--" if start == BASELINE else "-"
                ax.plot(x, line["NRMSE_Median"], style, marker="o", color=START_COLORS[start], label=start)
                ax.fill_between(x, line["NRMSE_Q25"], line["NRMSE_Q75"], color=START_COLORS[start], alpha=0.12)
            ax.set_xticks(range(len(fractions)))
            ax.set_xticklabels(fraction_labels(fractions))
            ax.set_title(f"{feature_label(feature_set)} features, {scaling} scaling")
            ax.set_xlabel("Share of the rice training pool used (0% = before any rice training)")
            ax.set_ylabel("Median test NRMSE (%)")
            ax.grid(True, alpha=0.3)
            # Zero-shot errors of random networks can be far above the rest; cap the axis so trained shares stay readable.
            trained = panel[panel["Train_Fraction"] > ZERO_SHOT]
            if not trained.empty:
                top = min(panel["NRMSE_Median"].max() + 2, trained["NRMSE_Q75"].max() + 15)
                ax.set_ylim(trained["NRMSE_Q25"].min() - 1, top)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(f"{tl.RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def filled_curves(epochs: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Epoch curves on a common epoch axis.

    After early stopping, a fit's curve is extended with its final (kept
    model) error, since that is what the fit would score from then on.

    Args:
        epochs: Epoch curve rows from run_fits.
        results: Completed fit rows with NRMSE_Percent.

    Returns:
        Long frame with FIT_KEYS, Epoch, and Test_NRMSE for every epoch up to
        HEAD_EPOCHS + MAX_FINE_TUNE_EPOCHS.
    """
    wide = epochs.pivot_table(index=FIT_KEYS, columns="Epoch", values="Test_NRMSE")
    wide = wide.reindex(columns=range(tl.HEAD_EPOCHS + tl.MAX_FINE_TUNE_EPOCHS + 1))
    final = results.set_index(FIT_KEYS)["NRMSE_Percent"].reindex(wide.index).to_numpy()
    values = wide.to_numpy()
    values = np.where(np.isnan(values), final[:, None], values)
    filled = pd.DataFrame(values, index=wide.index, columns=pd.Index(wide.columns, name="Epoch"))
    # melt rather than stack: pandas 3 rejects stack(dropna=False), which is how Grace 3.0.6 crashed.
    return filled.reset_index().melt(id_vars=list(filled.index.names), var_name="Epoch", value_name="Test_NRMSE")


def plot_epoch_curves(curves: pd.DataFrame, feature_set: str, filename: str) -> None:
    """
    Median test NRMSE after each epoch, for a few training shares.

    Args:
        curves: Output of filled_curves.
        feature_set: "full" or "common".
        filename: PNG name under Results/Graphs.
    """
    block = curves[(curves["Feature_Set"] == feature_set) & curves["Train_Fraction"].isin(EPOCH_PLOT_FRACTIONS)]
    if block.empty:
        return
    scalings = [mode for mode in tl.SHARED_SCALING_MODES if mode in set(block["Scaling"])]
    fractions = [fraction for fraction in EPOCH_PLOT_FRACTIONS if fraction in set(block["Train_Fraction"])]
    medians = block.groupby(["Scaling", "Start", "Train_Fraction", "Epoch"])["Test_NRMSE"].median().reset_index()
    fig, axes = plt.subplots(
        len(scalings), len(fractions), figsize=(4.8 * len(fractions), 4.2 * len(scalings)),
        squeeze=False, sharey="row"
    )
    for r, scaling in enumerate(scalings):
        for c, fraction in enumerate(fractions):
            ax = axes[r][c]
            panel = medians[(medians["Scaling"] == scaling) & (medians["Train_Fraction"] == fraction)]
            for start in starts_for(feature_set):
                line = panel[panel["Start"] == start]
                ax.plot(line["Epoch"], line["Test_NRMSE"], color=START_COLORS[start], label=start)
            ax.axvline(tl.HEAD_EPOCHS, color="black", linestyle=":", linewidth=1)
            trained = panel[panel["Epoch"] >= 1]["Test_NRMSE"]
            if not trained.empty:
                ax.set_ylim(trained.min() - 1, min(trained.max(), trained.min() + 25) + 1)
            ax.set_title(f"{fraction:.0%} of pool, {scaling} scaling")
            ax.set_xlabel("Epoch (dotted line: fine-tuning starts)")
            ax.set_ylabel("Median test NRMSE (%)")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{feature_label(feature_set)} features: test error after each epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(f"{tl.RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def plot_settled_epochs(completed: pd.DataFrame, filename: str) -> None:
    """
    Box plot of the first epoch within SETTLED_TOLERANCE of each fit's final error.

    Args:
        completed: Completed fit rows (trained shares only are drawn).
        filename: PNG name under Results/Graphs.
    """
    block = completed[(completed["Train_Fraction"] > ZERO_SHOT) & completed["Settled_Epoch"].notna()]
    if block.empty:
        return
    block = block.assign(Share=block["Train_Fraction"].map(lambda f: f"{f:.0%}"))
    scalings = [mode for mode in tl.SHARED_SCALING_MODES if mode in set(block["Scaling"])]
    fig, axes = plt.subplots(len(scalings), len(FEATURE_SETS), figsize=(16, 5 * len(scalings)), squeeze=False)
    for r, scaling in enumerate(scalings):
        for c, feature_set in enumerate(FEATURE_SETS):
            ax = axes[r][c]
            panel = block[(block["Scaling"] == scaling) & (block["Feature_Set"] == feature_set)]
            order = starts_for(feature_set)
            sns.boxplot(
                data=panel, x="Share", y="Settled_Epoch", hue="Start", hue_order=order,
                order=fraction_labels(TRAIN_FRACTIONS), palette=START_COLORS, ax=ax, fliersize=2,
            )
            ax.axhline(tl.HEAD_EPOCHS, color="black", linestyle=":", linewidth=1)
            ax.set_title(f"{feature_label(feature_set)} features, {scaling} scaling")
            ax.set_xlabel("Share of the rice training pool used")
            ax.set_ylabel(f"First epoch within {SETTLED_TOLERANCE:g} point of final error")
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(f"{tl.RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def comparison_table(comparisons: pd.DataFrame, a: str, b: str) -> str:
    """
    Text table for one pair: rows are training shares, columns are scaling and feature set.

    Each cell is "median difference | share of splits A wins | Wilcoxon p | corrected t p".

    Args:
        comparisons: Output of compare_starts.
        a: First start.
        b: Second start.

    Returns:
        The table, or "" if the pair was not run.
    """
    block = comparisons[(comparisons["A"] == a) & (comparisons["B"] == b)] if not comparisons.empty else comparisons
    if block.empty:
        return ""

    def cell(row: pd.Series) -> str:
        text = f"{row['Median_Diff']:+.2f} | {row['A_Better_Share']:.0%}"
        if not np.isnan(row["Wilcoxon_p"]):
            text += f" | W p={row['Wilcoxon_p']:.2g}"
        if not np.isnan(row["Corrected_t_p"]):
            text += f" | t p={row['Corrected_t_p']:.2g}"
        return text

    block = block.assign(
        Column=block["Feature_Set"].map(feature_label) + ", " + block["Scaling"],
        Cell=block.apply(cell, axis=1),
        Share=block["Train_Fraction"].map(lambda f: f"{f:.0%}"),
    )
    table = block.pivot(index="Share", columns="Column", values="Cell")
    table = table.reindex([label for label in fraction_labels((ZERO_SHOT,) + TRAIN_FRACTIONS) if label in table.index])
    return "\n".join([f"{a} minus {b}", table.to_string(), ""])


def write_report(
    data: dict,
    params: dict,
    scalings: Sequence[str],
    sources: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    efficiency: pd.DataFrame,
    timestamp: str
) -> str:
    """
    Write the plain-text digest of the run and return it.

    Args:
        data: Output of tl.load_data.
        params: Hyperparameters used.
        scalings: Scaling modes run.
        sources: Output of train_sources (summary rows).
        summary: Output of summarize.
        comparisons: Output of compare_starts.
        efficiency: Output of data_efficiency.
        timestamp: Run timestamp.

    Returns:
        The report text.
    """
    def pivot(values: str, starts: Optional[Sequence[str]] = None, digits: int = 2) -> str:
        block = summary if starts is None else summary[summary["Start"].isin(starts)]
        block = block.assign(
            Config=block["Feature_Set"].fillna("any").map(lambda f: f if f == "any" else feature_label(f))
            + " " + block["Start"],
            Share=block["Train_Fraction"].map(lambda f: f"{f:.0%}"),
        )
        table = block.pivot_table(index=["Scaling", "Config"], columns="Share", values=values)
        table = table[[label for label in fraction_labels((ZERO_SHOT,) + TRAIN_FRACTIONS) if label in table.columns]]
        return table.round(digits).to_string()

    lines = [
        f"Run {timestamp}",
        f"Architecture: {params}",
        f"Shared feature scaling: {', '.join(scalings)}",
        f"Soybean rows: {data['soy_rows']} | Rice rows: {data['rice_rows']} | Common features: {data['n_common_features']}",
        f"Grid: {NUM_SPLITS} random 80/20 splits, pool shares {list(TRAIN_FRACTIONS)} plus zero-shot, "
        f"{NUM_SOURCE_MODELS} real and {NUM_SOURCE_MODELS} shuffled soybean source networks",
        "",
        "--- Soybean source networks (in-sample correlation with the real soybean yields) ---",
        "Real sources should be clearly positive; shuffled sources should be near 0.",
        sources[["Kind", "Source_Model", "InSample_r_vs_real_yield"]].round(3).to_string(index=False),
        "",
        "--- Median test NRMSE (%) by share of the rice pool (0% = before any rice training) ---",
        pivot("NRMSE_Median"),
        "",
        "--- Median correlation between predicted and measured test yield ---",
        pivot("Pearson_r_Median", digits=3),
        "",
        "Cells: median NRMSE difference in points (negative means the first is better) | share of",
        "splits where the first is better | Wilcoxon signed-rank p | corrected resampled t-test p.",
        "Zero-shot rows get no p-values.",
        "",
    ]
    lines.extend(comparison_table(comparisons, a, b) for a, b in START_PAIRS)
    if not efficiency.empty:
        table = efficiency.assign(
            Config=efficiency["Feature_Set"].map(feature_label) + " " + efficiency["Start"],
            Share=efficiency["Train_Fraction"].map(lambda f: f"{f:.0%}"),
        ).pivot_table(index=["Scaling", "Config"], columns="Share", values="Ratio")
        table = table[[label for label in fraction_labels(TRAIN_FRACTIONS) if label in table.columns]]
        lines.extend([
            "--- Data efficiency: cold rows needed to match the start's error, as a multiple of the rows it used ---",
            "Above 1 means the start needs fewer rows than cold. Blank means outside cold's error range.",
            table.round(2).to_string(),
            "",
        ])
    lines.extend([
        f"--- Median first epoch within {SETTLED_TOLERANCE:g} NRMSE point of the final error "
        f"(epochs 1 to {tl.HEAD_EPOCHS} train only the output layer) ---",
        pivot("Settled_Epoch_Median", starts=STARTS, digits=1),
        "",
        "--- Median test NRMSE (%) after the output-layer phase, before fine-tuning ---",
        pivot("Head_NRMSE_Median", starts=STARTS),
        "",
    ])
    report = "\n".join(lines) + "\n"
    with open(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_report_{timestamp}.txt", "w", encoding="utf-8") as handle:
        handle.write(report)
    return report


def main() -> None:
    """
    Run the convergence experiment and write every table, plot, and model.
    """
    started = time.time()
    timestamp = tl.TIMESTAMP
    scalings = tl.parse_scaling_modes(SCALING_SETTING)
    tl.folder_creation()
    data = tl.load_data()
    tasks = build_tasks(scalings)
    print(f"Shared feature scaling: {', '.join(scalings)}")
    print(f"Planned fits: {len(tasks)} ({NUM_SPLITS} splits, shares {list(TRAIN_FRACTIONS)} plus zero-shot)")

    tl.save_json(f"{tl.RESULTS_DIR}/Models/{PREFIX}_run_config_{timestamp}.json", {
        "shared_scaling": list(scalings),
        "num_splits": NUM_SPLITS,
        "num_source_models": NUM_SOURCE_MODELS,
        "train_fractions": list(TRAIN_FRACTIONS),
        "starts": list(STARTS),
        "test_fraction": tl.TEST_FRACTION,
        "min_train_rows": MIN_TRAIN_ROWS,
        "settled_tolerance": SETTLED_TOLERANCE,
        "rice_only_init_scale": tl.RICE_ONLY_INIT_SCALE,
        "params_file": PARAMS_FILE or None,
        "head_epochs": tl.HEAD_EPOCHS,
        "max_fine_tune_epochs": tl.MAX_FINE_TUNE_EPOCHS,
        "early_stopping_patience": tl.EARLY_STOPPING_PATIENCE,
        "batch_size": tl.BATCH_SIZE,
        "num_workers": tl.NUM_WORKERS,
        "planned_fits": len(tasks),
    })

    params = load_best_params(data)
    print(f"[{(time.time() - started) / 60:.1f} min] Parameters: {params}")

    print(f"Training {NUM_SOURCE_MODELS} real and {NUM_SOURCE_MODELS} shuffled soybean source networks...")
    weights, sources = train_sources(data, params, timestamp)
    sources.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_sources_{timestamp}.csv", index=False)
    print(f"[{(time.time() - started) / 60:.1f} min] Source networks done")

    state = {
        "rice_inputs": {mode: tl.build_rice_inputs(data, mode) for mode in scalings},
        "y_raw": data["rice_y_raw"],
        "years": data["rice_years"],
        "rice_only_idx": data["rice_only_idx"],
        "params": params,
        "weights": weights,
    }
    print("Running rice fits...")
    results, epochs = run_fits(tasks, state, timestamp)
    print(f"[{(time.time() - started) / 60:.1f} min] Rice fits done")
    print("Status counts:", results.groupby("Status").size().to_dict())
    for error in results.loc[results["Status"] == "Failed", "Error"].unique()[:5]:
        print(f"  Failure: {error}")

    results = pd.concat([results, baseline_rows(scalings, data["rice_y_raw"], data["rice_years"])], ignore_index=True)
    results.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_results_{timestamp}.csv", index=False)
    epochs.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_epochs_{timestamp}.csv", index=False)

    completed = results[results["Status"] == "Completed"]
    summary = summarize(completed)
    summary.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_summary_{timestamp}.csv", index=False)
    comparisons = compare_starts(completed[completed["Start"] != BASELINE])
    comparisons.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_comparisons_{timestamp}.csv", index=False)
    efficiency = data_efficiency(summary[summary["Start"] != BASELINE])
    efficiency.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_data_efficiency_{timestamp}.csv", index=False)

    plot_learning_curves(summary, f"Convergence_LearningCurve_{timestamp}.png")
    trained = completed[(completed["Start"] != BASELINE) & (completed["Train_Fraction"] > ZERO_SHOT)]
    if not epochs.empty:
        curves = filled_curves(epochs, trained)
        for feature_set in FEATURE_SETS:
            plot_epoch_curves(curves, feature_set, f"Convergence_Epochs_{feature_label(feature_set)}_{timestamp}.png")
    plot_settled_epochs(trained, f"Convergence_SettledEpoch_{timestamp}.png")

    report = write_report(data, params, scalings, sources, summary, comparisons, efficiency, timestamp)
    print(report)
    print(f"[{(time.time() - started) / 60:.1f} min] All outputs written")


if __name__ == "__main__":
    main()
