"""
Follow-up transfer tests: alignment, overlap features, matched soybean rows,
soybean score as a rice input, joint crop-flag training, and small-init Full.

Drops harvest day of year in VM_TransferLearning.clean_data (pre-harvest).
Reuses that module's splits, training schedule, and paired statistics.

Methods (same rice rows within a split/fraction/year):

    Cold_Common / Warm_Common     per_crop shared features (reference)
    Cold_Quantile / Warm_Quantile shared features mapped to soybean quantiles
    Cold_Overlap / Warm_Overlap   shared columns whose rice values overlap soybean
    Warm_Matched                  Warm_Common weights from soybean grown in
                                  rice-like GDD / planting-day weather
    Cold_SoyScore                 Cold_Common plus the soybean net's rice score
    Joint_CropFlag                one net on soybean + rice, with a crop flag
    Cold_Full / Warm_Full         all rice columns, per_crop
    Cold_SmallInit                Cold_Full with tiny rice-only starting weights
    Cold_Full_SoyScore            Cold_Full plus the soybean score

Warm_* start from the matching soybean source. Joint has no soybean-weight copy.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import VM_TransferLearning as tl  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from typing import Dict, List, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

NUM_RANDOM_SPLITS = 30
NUM_LOYO_REPEATS = 10
TRAIN_FRACTIONS = (0.05, 0.25, 1.0)
MIN_TRAIN_ROWS = 20
CHECKPOINT_EVERY = 500
MIN_OVERLAP_FEATURES = 5
MIN_MATCHED_SOY_ROWS = 200
OVERLAP_INSIDE = 0.50
CLIMATE_FEATURES = ("GDD_Avg", "days_from_year_start_Planting")
SOY_PER_RICE = 4
MIN_JOINT_SOY = 200
PARAMS_FILE = os.environ.get("VM_BEST_PARAMS_FILE", "").strip()
PREFIX = "ideas"
PAIR_KEYS = ["Evaluation", "Left_Out_Year", "Train_Fraction", "Iteration"]

# name -> (group used in plots, needs soybean-style weights)
METHODS = (
    "Cold_Common",
    "Warm_Common",
    "Cold_Quantile",
    "Warm_Quantile",
    "Cold_Overlap",
    "Warm_Overlap",
    "Warm_Matched",
    "Cold_SoyScore",
    "Joint_CropFlag",
    "Cold_Full",
    "Warm_Full",
    "Cold_SmallInit",
    "Cold_Full_SoyScore",
)
METHOD_PAIRS = (
    ("Warm_Common", "Cold_Common"),
    ("Warm_Quantile", "Cold_Quantile"),
    ("Warm_Overlap", "Cold_Overlap"),
    ("Warm_Matched", "Warm_Common"),
    ("Warm_Matched", "Cold_Common"),
    ("Cold_SoyScore", "Cold_Common"),
    ("Joint_CropFlag", "Cold_Common"),
    ("Warm_Full", "Cold_Full"),
    ("Cold_SmallInit", "Cold_Full"),
    ("Cold_Full_SoyScore", "Cold_Full"),
    ("Warm_Quantile", "Warm_Common"),
    ("Warm_Overlap", "Warm_Common"),
)
METHOD_COLORS = {
    "Cold_Common": "tab:orange",
    "Warm_Common": "tab:blue",
    "Cold_Quantile": "tab:pink",
    "Warm_Quantile": "tab:purple",
    "Cold_Overlap": "olivedrab",
    "Warm_Overlap": "tab:green",
    "Warm_Matched": "tab:cyan",
    "Cold_SoyScore": "tab:brown",
    "Joint_CropFlag": "tab:red",
    "Cold_Full": "darkorange",
    "Warm_Full": "royalblue",
    "Cold_SmallInit": "firebrick",
    "Cold_Full_SoyScore": "sienna",
    tl.BASELINE_NAME: "gray",
}

_STATE: Dict = {}


def init_worker(state: dict) -> None:
    """Store read-only inputs in each spawned worker."""
    global _STATE
    _STATE = state


def load_best_params(data: dict) -> dict:
    """Read VM_BEST_PARAMS_FILE or run Optuna on soybean."""
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


def overlap_keep_mask(soy_X: np.ndarray, rice_X: np.ndarray) -> np.ndarray:
    """
    True for shared columns where at least OVERLAP_INSIDE of rice lies in soybean's 5th-95th percentile.

    If that keeps fewer than MIN_OVERLAP_FEATURES, the highest-overlap columns are kept instead.
    """
    n = soy_X.shape[1]
    inside = np.zeros(n, dtype=float)
    keep = np.zeros(n, dtype=bool)
    for j in range(n):
        lo, hi = np.quantile(soy_X[:, j], [0.05, 0.95])
        inside[j] = float(np.mean((rice_X[:, j] >= lo) & (rice_X[:, j] <= hi)) if hi > lo else 0.0)
        keep[j] = inside[j] >= OVERLAP_INSIDE
    if int(keep.sum()) < MIN_OVERLAP_FEATURES:
        order = np.argsort(-inside)
        keep[:] = False
        keep[order[:MIN_OVERLAP_FEATURES]] = True
    return keep


def climate_match_mask(soy_X: np.ndarray, rice_X: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, dict]:
    """
    Soybean rows closest to rice on GDD and planting day.

    A hard 5th-95th percentile box of rice contains no soybean rows, so the
    nearest 10% of soybean (at least MIN_MATCHED_SOY_ROWS) is kept instead.
    Distance is in soybean standard deviations from the rice median.

    Returns:
        Boolean mask over soybean rows, and a small summary (hard-box count,
        rows kept, median distance).
    """
    cols = [list(names).index(name) for name in CLIMATE_FEATURES if name in names]
    hard = np.ones(len(soy_X), dtype=bool)
    for j in cols:
        lo, hi = np.quantile(rice_X[:, j], [0.05, 0.95])
        hard &= (soy_X[:, j] >= lo) & (soy_X[:, j] <= hi)
    soy = soy_X[:, cols]
    scale = soy.std(axis=0)
    scale[scale == 0] = 1.0
    center = np.median((rice_X[:, cols] - soy.mean(axis=0)) / scale, axis=0)
    distance = np.linalg.norm((soy - soy.mean(axis=0)) / scale - center, axis=1)
    n_keep = max(MIN_MATCHED_SOY_ROWS, int(round(0.10 * len(soy_X))))
    chosen = np.zeros(len(soy_X), dtype=bool)
    chosen[np.argsort(distance)[:n_keep]] = True
    info = {
        "hard_box_rows": int(hard.sum()),
        "matched_rows": int(chosen.sum()),
        "median_distance_sd": float(np.median(distance[chosen])),
    }
    return chosen, info


def slice_weights(weights: List[np.ndarray], row_idx: np.ndarray) -> List[np.ndarray]:
    """Keep selected input rows of the first Dense kernel; leave later layers as they are."""
    first = np.array(weights[0], copy=True)[row_idx]
    return [first] + [np.array(w, copy=True) for w in weights[1:]]


def small_init_full(source_common: List[np.ndarray], n_full: int, common_idx: np.ndarray, seed: int) -> List[np.ndarray]:
    """Warm_Full-style first layer: soybean rows on shared columns, tiny random elsewhere."""
    rng = np.random.default_rng(tl.GLOBAL_SEED + seed)
    first = rng.normal(scale=tl.RICE_ONLY_INIT_SCALE, size=(n_full, source_common[0].shape[1]))
    first[common_idx, :] = source_common[0]
    return [first] + [np.array(w, copy=True) for w in source_common[1:]]


def train_source_job(job: dict) -> Tuple[str, List[np.ndarray], str]:
    """Train one soybean source described by job (Kind, X, y_z, Stem)."""
    state = _STATE
    seed = tl.GLOBAL_SEED
    order = np.random.default_rng(seed).permutation(len(job["y_z"]))
    model = tl.build_and_train(
        job["X"][order], job["y_z"][order], job["X"].shape[1], state["params"], seed=seed
    )
    path = tl.save_trained_model(model, f"{PREFIX}_{job['Stem']}", state["results_dir"], state["timestamp"])
    return job["Kind"], model.get_weights(), path


def predict_source(weights: List[np.ndarray], X: np.ndarray, params: dict) -> np.ndarray:
    """Score rows with a soybean-shaped network (no training)."""
    import tensorflow as tf

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(X.shape[1],))])
    for i in range(params["n_layers"]):
        model.add(tf.keras.layers.Dense(params[f"n_units_l{i}"], activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.set_weights(weights)
    return model.predict(X, verbose=0).ravel()


def joint_train_and_predict(
    rice_train_raw: np.ndarray,
    rice_y_train: np.ndarray,
    rice_test_raw: np.ndarray,
    soy_raw: np.ndarray,
    soy_y_raw: np.ndarray,
    params: dict,
    seed: int,
) -> np.ndarray:
    """
    Train one net on mixed soybean and rice rows with a crop flag (0 soybean, 1 rice).

    Features are z-scored on the mixed training rows so both crops share one scale.
    Soybean is subsampled to about SOY_PER_RICE times the rice training size.
    Returns rice test predictions in original yield units.
    """
    rng = np.random.default_rng(seed)
    n_keep = min(len(soy_raw), max(SOY_PER_RICE * len(rice_train_raw), MIN_JOINT_SOY))
    pick = rng.choice(len(soy_raw), size=n_keep, replace=False)
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler().fit(np.vstack([soy_raw[pick], rice_train_raw]))
    soy_z = x_scaler.transform(soy_raw[pick])
    rice_z = x_scaler.transform(rice_train_raw)
    rice_test_z = x_scaler.transform(rice_test_raw)
    y_scaler = StandardScaler().fit(np.concatenate([soy_y_raw[pick], rice_y_train]).reshape(-1, 1))
    soy_yz = y_scaler.transform(soy_y_raw[pick].reshape(-1, 1)).ravel()
    rice_yz = y_scaler.transform(rice_y_train.reshape(-1, 1)).ravel()
    X = np.vstack([
        np.column_stack([soy_z, np.zeros(n_keep)]),
        np.column_stack([rice_z, np.ones(len(rice_z))]),
    ])
    y = np.concatenate([soy_yz, rice_yz])
    order = rng.permutation(len(y))
    model = tl.build_and_train(X[order], y[order], X.shape[1], params, seed=seed)
    pred_z = model.predict(np.column_stack([rice_test_z, np.ones(len(rice_test_z))]), verbose=0).ravel()
    return y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()


def append_score(X: np.ndarray, score: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Add the soybean score as a last column for the given rows."""
    return np.column_stack([X, score[idx].reshape(-1, 1)])


def method_inputs(method: str, state: dict, train_idx: np.ndarray, test_idx: np.ndarray):
    """
    Feature matrices, optional transfer weights, and per-split scale columns for one method.

    Returns:
        X_train, X_test, weights or None, fit_columns, and 'joint' | 'mlp'.
    """
    comm = state["rice_comm_raw"]
    full = state["rice_full_raw"]
    overlap = state["overlap_idx"]
    common_idx = state["common_idx"]
    n_full = full.shape[1]

    def per_crop(Xtr, Xte):
        return Xtr, Xte, np.arange(Xtr.shape[1])

    if method in ("Cold_Common", "Warm_Common", "Warm_Matched"):
        Xtr, Xte, fit = per_crop(comm[train_idx], comm[test_idx])
        weights = None
        if method == "Warm_Common":
            weights = state["w_std"]
        elif method == "Warm_Matched":
            weights = state["w_matched"]
        return Xtr, Xte, weights, fit, "mlp"

    if method in ("Cold_Quantile", "Warm_Quantile"):
        Xtr, Xte = state["rice_comm_qt"][train_idx], state["rice_comm_qt"][test_idx]
        weights = state["w_qt"] if method.startswith("Warm") else None
        return Xtr, Xte, weights, None, "mlp"

    if method in ("Cold_Overlap", "Warm_Overlap"):
        Xtr, Xte, fit = per_crop(comm[train_idx][:, overlap], comm[test_idx][:, overlap])
        weights = state["w_overlap"] if method.startswith("Warm") else None
        return Xtr, Xte, weights, fit, "mlp"

    if method == "Cold_SoyScore":
        Xtr = append_score(comm[train_idx], state["soy_score"], train_idx)
        Xte = append_score(comm[test_idx], state["soy_score"], test_idx)
        return Xtr, Xte, None, np.arange(Xtr.shape[1]), "mlp"

    if method == "Joint_CropFlag":
        Xtr, Xte, fit = per_crop(comm[train_idx], comm[test_idx])
        return Xtr, Xte, None, fit, "joint"

    if method in ("Cold_Full", "Warm_Full", "Cold_SmallInit"):
        Xtr, Xte = full[train_idx], full[test_idx]
        fit = np.arange(n_full)
        if method == "Warm_Full":
            return Xtr, Xte, small_init_full(state["w_std"], n_full, common_idx, seed=0), fit, "mlp"
        if method == "Cold_SmallInit":
            return Xtr, Xte, "smallinit", fit, "mlp"
        return Xtr, Xte, None, fit, "mlp"

    if method == "Cold_Full_SoyScore":
        Xtr = append_score(full[train_idx], state["soy_score"], train_idx)
        Xte = append_score(full[test_idx], state["soy_score"], test_idx)
        return Xtr, Xte, None, np.arange(Xtr.shape[1]), "mlp"

    raise ValueError(f"Unknown method {method}")


def cold_smallinit_weights(state: dict, seed: int) -> List[np.ndarray]:
    """Random full-network weights with rice-only input rows at RICE_ONLY_INIT_SCALE."""
    import tensorflow as tf

    n_full = state["rice_full_raw"].shape[1]
    params = state["params"]
    tf.keras.utils.set_random_seed(int(seed))
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(n_full,))])
    for i in range(params["n_layers"]):
        model.add(tf.keras.layers.Dense(params[f"n_units_l{i}"], activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    weights = model.get_weights()
    rng = np.random.default_rng(tl.GLOBAL_SEED + int(seed))
    rows = state["rice_only_idx"]
    weights[0][rows, :] = rng.normal(scale=tl.RICE_ONLY_INIT_SCALE, size=(len(rows), weights[0].shape[1]))
    return weights


def run_fit(task: dict) -> dict:
    """Train and score one method on one split."""
    state = _STATE
    method = task["Method"]
    train_idx, test_idx = tl.select_rows(
        task["Evaluation"], task["Iteration"], task["Train_Fraction"],
        state["years"], task["Left_Out_Year"],
    )
    row = dict(task)
    row.update({
        "Train_Rows": int(len(train_idx)), "Test_Rows": int(len(test_idx)),
        "RMSE": np.nan, "NRMSE_Percent": np.nan, "Status": "Completed", "Error": "",
    })
    if method == "Warm_Matched" and state["w_matched"] is None:
        row.update(Status="Skipped", Error="Too few climate-matched soybean rows.")
        return row
    if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) < tl.MIN_TEST_ROWS:
        row.update(Status="Skipped", Error=f"Too few rows (train={len(train_idx)}, test={len(test_idx)}).")
        return row
    try:
        Xtr, Xte, weights, fit_columns, kind = method_inputs(method, state, train_idx, test_idx)
        y = state["y_raw"]
        Xt, Xv, yt_z, y_scaler = tl.prepare_split(Xtr, Xte, y[train_idx], fit_columns)
        seed = int(task["Iteration"])
        if kind == "joint":
            pred_raw = joint_train_and_predict(
                state["rice_comm_raw"][train_idx], y[train_idx],
                state["rice_comm_raw"][test_idx],
                state["soy_raw"], state["soy_y_raw"], state["params"], seed,
            )
        else:
            if weights == "smallinit":
                weights = cold_smallinit_weights(state, seed)
            model = tl.build_and_train(Xt, yt_z, Xt.shape[1], state["params"], weights, seed=seed)
            pred_z = model.predict(Xv, verbose=0).ravel()
            pred_raw = y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()
        row.update(tl.calculate_error_metrics(y[test_idx], pred_raw))
    except Exception as exc:
        row.update(Status="Failed", Error=str(exc))
    return row


def build_tasks(years: Sequence[int], include_matched: bool) -> List[dict]:
    """Every method x evaluation x fraction x repeat, largest fractions first."""
    methods = [m for m in METHODS if include_matched or m != "Warm_Matched"]
    tasks = []
    for fraction in sorted(TRAIN_FRACTIONS, reverse=True):
        for method in methods:
            splits = [("random", None, i) for i in range(NUM_RANDOM_SPLITS)]
            splits += [("loyo", int(year), i) for year in years for i in range(NUM_LOYO_REPEATS)]
            for evaluation, year, iteration in splits:
                tasks.append({
                    "Evaluation": evaluation,
                    "Method": method,
                    "Train_Fraction": float(fraction),
                    "Iteration": int(iteration),
                    "Left_Out_Year": year,
                })
    return tasks


def run_tasks(tasks: List[dict], state: dict, timestamp: str) -> pd.DataFrame:
    """Run fits in a spawn pool and checkpoint every CHECKPOINT_EVERY completions."""
    rows = []
    checkpoint = f"{tl.RESULTS_DIR}/Errors/{PREFIX}_partial_results_{timestamp}.csv"
    with ProcessPoolExecutor(max_workers=tl.NUM_WORKERS, initializer=init_worker, initargs=(state,)) as executor:
        futures = [executor.submit(run_fit, task) for task in tasks]
        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"Worker failed: {exc}")
                continue
            if count % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    return pd.DataFrame(rows)


def baseline_rows(tasks: List[dict], y_raw: np.ndarray, years: np.ndarray) -> pd.DataFrame:
    """Mean training yield, one row per unique split (taken from Cold_Common tasks)."""
    rows = []
    seen = set()
    for task in tasks:
        if task["Method"] != "Cold_Common":
            continue
        key = (task["Evaluation"], task["Left_Out_Year"], task["Train_Fraction"], task["Iteration"])
        if key in seen:
            continue
        seen.add(key)
        train_idx, test_idx = tl.select_rows(
            task["Evaluation"], task["Iteration"], task["Train_Fraction"], years, task["Left_Out_Year"]
        )
        prediction = np.full(len(test_idx), y_raw[train_idx].mean())
        rows.append({
            **{k: task[k] for k in task},
            "Method": tl.BASELINE_NAME,
            "Train_Rows": int(len(train_idx)), "Test_Rows": int(len(test_idx)),
            **tl.calculate_error_metrics(y_raw[test_idx], prediction),
            "Status": "Completed", "Error": "",
        })
    return pd.DataFrame(rows)


def summarize(completed: pd.DataFrame) -> pd.DataFrame:
    """Median NRMSE and RMSE per method, evaluation, fraction, and year."""
    keys = ["Evaluation", "Method", "Train_Fraction", "Left_Out_Year"]
    return completed.groupby(keys, dropna=False).agg(
        N=("NRMSE_Percent", "size"),
        Train_Rows=("Train_Rows", "median"),
        NRMSE_Median=("NRMSE_Percent", "median"),
        NRMSE_Q25=("NRMSE_Percent", lambda s: s.quantile(0.25)),
        NRMSE_Q75=("NRMSE_Percent", lambda s: s.quantile(0.75)),
        RMSE_Median=("RMSE", "median"),
    ).reset_index()


def compare_methods(completed: pd.DataFrame) -> pd.DataFrame:
    """Split-by-split NRMSE comparisons for METHOD_PAIRS."""
    rows = []
    block = completed[completed["Method"] != tl.BASELINE_NAME]
    for a, b in METHOD_PAIRS:
        left = block.loc[block["Method"] == a, PAIR_KEYS + ["NRMSE_Percent", "Train_Rows", "Test_Rows"]]
        right = block.loc[block["Method"] == b, PAIR_KEYS + ["NRMSE_Percent"]]
        if left.empty or right.empty:
            continue
        merged = left.merge(right, on=PAIR_KEYS, suffixes=("_A", "_B"))
        for (evaluation, year, fraction), group in merged.groupby(
            ["Evaluation", "Left_Out_Year", "Train_Fraction"], dropna=False
        ):
            rows.append({
                "Evaluation": evaluation, "Left_Out_Year": year, "Train_Fraction": fraction,
                "A": a, "B": b, **tl.paired_statistics(group, evaluation),
            })
    return pd.DataFrame(rows)


def plot_curves(summary: pd.DataFrame, evaluation: str, filename: str) -> None:
    """Median NRMSE vs training share, one panel per evaluation slice."""
    block = summary[summary["Evaluation"] == evaluation]
    if block.empty:
        return
    years = [None] if evaluation == "random" else sorted(block["Left_Out_Year"].dropna().unique())
    fig, axes = plt.subplots(1, max(len(years), 1), figsize=(5.5 * max(len(years), 1) + 1, 5.5), squeeze=False, sharey=True)
    ticks = [f * 100 for f in TRAIN_FRACTIONS]
    for c, year in enumerate(years):
        ax = axes[0][c]
        panel = block if year is None else block[block["Left_Out_Year"] == year]
        for method in list(METHODS) + [tl.BASELINE_NAME]:
            line = panel[panel["Method"] == method].sort_values("Train_Fraction")
            if line.empty:
                continue
            style = "--" if method == tl.BASELINE_NAME else "-"
            ax.plot(line["Train_Fraction"] * 100, line["NRMSE_Median"], style, marker="o",
                    color=METHOD_COLORS.get(method, "black"), label=method)
            ax.fill_between(line["Train_Fraction"] * 100, line["NRMSE_Q25"], line["NRMSE_Q75"],
                            color=METHOD_COLORS.get(method, "black"), alpha=0.08)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{f:.0%}" for f in TRAIN_FRACTIONS])
        ax.set_title("Random 80/20" if year is None else f"LOYO {int(year)} held out")
        ax.set_xlabel("Share of rice training rows")
        ax.set_ylabel("Median NRMSE (%)")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(f"{tl.RESULTS_DIR}/Graphs", filename), dpi=300)
    plt.close(fig)


def format_pair(comparisons: pd.DataFrame, a: str, b: str) -> str:
    """Text table of one method pair."""
    block = comparisons[(comparisons["A"] == a) & (comparisons["B"] == b)] if not comparisons.empty else comparisons
    if block.empty:
        return ""

    def cell(row: pd.Series) -> str:
        text = f"{row['Median_Diff']:+.2f} | {row['A_Better_Share']:.0%}"
        if row["Evaluation"] == "random" and not np.isnan(row.get("Corrected_t_p", np.nan)):
            text += f" | p={row['Corrected_t_p']:.2g}"
        return text

    block = block.assign(
        Column=np.where(
            block["Evaluation"] == "random", "random",
            "LOYO " + block["Left_Out_Year"].fillna(0).astype(int).astype(str),
        ),
        Cell=block.apply(cell, axis=1),
        Share=block["Train_Fraction"].map(lambda f: f"{f:.0%}"),
    )
    table = block.pivot(index="Share", columns="Column", values="Cell")
    return "\n".join([f"{a} minus {b}", table.to_string(), ""])


def write_report(data: dict, params: dict, meta: dict, summary: pd.DataFrame, comparisons: pd.DataFrame, timestamp: str) -> str:
    """Write the digest of the ideas run."""
    full = summary.copy()
    full["Column"] = np.where(
        full["Evaluation"] == "random", "random",
        "LOYO " + full["Left_Out_Year"].fillna(0).astype(int).astype(str),
    )
    lines = [
        f"Run {timestamp}",
        f"Architecture: {params}",
        f"Soybean rows: {data['soy_rows']} | Rice rows: {data['rice_rows']} | Common: {data['n_common_features']}",
        f"Overlap features kept: {meta['n_overlap']} / {data['n_common_features']} -> {meta['overlap_names']}",
        f"Climate-matched soybean rows: {meta['n_matched']} "
        f"(rice 5-95% box contained {meta['matched_hard_box_rows']}; "
        f"kept nearest, median distance {meta['matched_median_distance_sd']:.2f} soybean SD)",
        "Harvest-timing columns dropped in clean_data (days_from_year_start_harvest).",
        f"Grid: {NUM_RANDOM_SPLITS} random splits, {NUM_LOYO_REPEATS} LOYO repeats/year, fractions {list(TRAIN_FRACTIONS)}",
        "",
        "Cells: median NRMSE difference (negative => first better) | share first wins | corrected t p on random splits.",
        "",
    ]
    for a, b in METHOD_PAIRS:
        lines.append(format_pair(comparisons, a, b))
    for fraction in TRAIN_FRACTIONS:
        slice_ = full[full["Train_Fraction"] == fraction]
        table = slice_.pivot_table(index="Method", columns="Column", values="NRMSE_Median")
        lines.extend([f"--- Median NRMSE (%) at {fraction:.0%} of rice training ---", table.round(2).to_string(), ""])
    report = "\n".join(lines) + "\n"
    with open(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_report_{timestamp}.txt", "w", encoding="utf-8") as handle:
        handle.write(report)
    return report


def main() -> None:
    """Run every follow-up method and write tables, plots, and source models."""
    started = time.time()
    timestamp = tl.TIMESTAMP
    tl.folder_creation()
    data = tl.load_data()
    years = sorted(int(y) for y in np.unique(data["rice_years"]))
    common = data["common_features"]
    soy_raw, rice_raw = data["soy_X_raw"], data["rice_X_comm_raw"]

    keep = overlap_keep_mask(soy_raw, rice_raw)
    overlap_idx = np.flatnonzero(keep)
    overlap_names = [common[i] for i in overlap_idx]
    matched, match_info = climate_match_mask(soy_raw, rice_raw, common)
    n_matched = int(matched.sum())
    print(f"Overlap columns ({len(overlap_names)}): {overlap_names}")
    print(
        f"Climate-matched soybean rows: {n_matched} / {len(soy_raw)} "
        f"(hard rice box had {match_info['hard_box_rows']}; "
        f"median distance {match_info['median_distance_sd']:.2f} soybean SD)"
    )

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(soy_raw)),
        subsample=len(soy_raw),
        random_state=tl.GLOBAL_SEED,
    )
    soy_qt = qt.fit_transform(soy_raw)
    rice_qt = qt.transform(rice_raw)

    params = load_best_params(data)
    print(f"[{(time.time() - started) / 60:.1f} min] Parameters: {params}")

    jobs = [
        {"Kind": "std", "Stem": "source_std", "X": data["soy_X"], "y_z": data["soy_y_z"]},
        {"Kind": "qt", "Stem": "source_quantile", "X": soy_qt, "y_z": data["soy_y_z"]},
        {"Kind": "overlap", "Stem": "source_overlap", "X": data["soy_X"][:, overlap_idx], "y_z": data["soy_y_z"]},
    ]
    include_matched = n_matched >= MIN_MATCHED_SOY_ROWS
    if include_matched:
        jobs.append({
            "Kind": "matched", "Stem": "source_matched",
            "X": data["soy_X"][matched], "y_z": data["soy_y_z"][matched],
        })
    else:
        print(f"Skipping Warm_Matched (need {MIN_MATCHED_SOY_ROWS} soybean rows, have {n_matched}).")

    src_state = {"params": params, "results_dir": tl.RESULTS_DIR, "timestamp": timestamp}
    sources: Dict[str, List[np.ndarray]] = {}
    workers = max(1, min(len(jobs), tl.NUM_WORKERS))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(src_state,)) as executor:
        for future in as_completed([executor.submit(train_source_job, job) for job in jobs]):
            kind, weights, path = future.result()
            sources[kind] = weights
            print(f"  {kind} -> {path}")

    soy_score = predict_source(sources["std"], data["scaler_soy_X"].transform(rice_raw), params)
    meta = {
        "n_overlap": int(len(overlap_names)),
        "overlap_names": overlap_names,
        "overlap_inside": OVERLAP_INSIDE,
        "n_matched": n_matched,
        "matched_hard_box_rows": match_info["hard_box_rows"],
        "matched_median_distance_sd": match_info["median_distance_sd"],
        "climate_features": [n for n in CLIMATE_FEATURES if n in common],
        "params_file": PARAMS_FILE or None,
        "dropped_harvest_timing": True,
    }
    tasks = build_tasks(years, include_matched)
    tl.save_json(f"{tl.RESULTS_DIR}/Models/{PREFIX}_run_config_{timestamp}.json", {
        **meta,
        "methods": [m for m in METHODS if include_matched or m != "Warm_Matched"],
        "num_random_splits": NUM_RANDOM_SPLITS,
        "num_loyo_repeats": NUM_LOYO_REPEATS,
        "train_fractions": list(TRAIN_FRACTIONS),
        "planned_fits": len(tasks),
        "architecture": params,
    })
    print(f"Planned fits: {len(tasks)}")

    state = {
        "params": params,
        "years": data["rice_years"],
        "y_raw": data["rice_y_raw"],
        "rice_comm_raw": rice_raw,
        "rice_full_raw": data["rice_X_full_raw"],
        "rice_comm_qt": rice_qt,
        "overlap_idx": overlap_idx,
        "common_idx": data["common_idx"],
        "rice_only_idx": data["rice_only_idx"],
        "soy_raw": data["soy_X_raw"],
        "soy_y_raw": data["soy_y_raw"],
        "soy_score": soy_score,
        "w_std": sources["std"],
        "w_qt": sources["qt"],
        "w_overlap": sources["overlap"],
        "w_matched": sources.get("matched"),
    }
    results = run_tasks(tasks, state, timestamp)
    print(f"[{(time.time() - started) / 60:.1f} min] Fits done")
    print("Status:", results.groupby("Status").size().to_dict())
    for err in results.loc[results["Status"] == "Failed", "Error"].unique()[:8]:
        print(f"  Failure: {err}")

    results = pd.concat([results, baseline_rows(tasks, data["rice_y_raw"], data["rice_years"])], ignore_index=True)
    results.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_results_{timestamp}.csv", index=False)
    completed = results[results["Status"] == "Completed"]
    summary = summarize(completed)
    summary.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_summary_{timestamp}.csv", index=False)
    comparisons = compare_methods(completed)
    comparisons.to_csv(f"{tl.RESULTS_DIR}/Errors/{PREFIX}_comparisons_{timestamp}.csv", index=False)
    plot_curves(summary, "random", f"Ideas_LearningCurve_Random_{timestamp}.png")
    plot_curves(summary, "loyo", f"Ideas_LearningCurve_LOYO_{timestamp}.png")
    report = write_report(data, params, meta, summary, comparisons, timestamp)
    print(report)
    print(f"[{(time.time() - started) / 60:.1f} min] All outputs written")


if __name__ == "__main__":
    main()
