# Results

Outputs of `Codes/VM_TransferLearning.py`. Every file from one run ends with the same `YYYYMMDD_HHMMSS` timestamp.

## Scenarios

| Scenario | Rice features | Starting weights |
| :--- | :--- | :--- |
| `Warm_Full` | All rice features | Soybean first layer matched by feature name; features soybean doesn't have start small and random |
| `Cold_Full` | All rice features | Random |
| `Warm_Common` | Features shared with soybean | Soybean network as trained |
| `Cold_Common` | Features shared with soybean | Random |

## Files

### `Errors/`
- `final_results_*.csv`, `final_rmse_results_*.csv`: NRMSE and RMSE per scenario over 100 random 80/20 splits.
- `random_split_metrics_*.csv`: the same splits in long format (scenario, iteration, RMSE, NRMSE). Failed splits appear as empty rows.
- `random_split_nrmse_description_*.csv`, `random_split_rmse_description_*.csv`: summary statistics per scenario.
- `loyo_results_*.csv`: leave-one-year-out rows (scenario x held-out year x 100 iterations) with fold sizes and status.
- `loyo_error_description_*.csv`: LOYO summary statistics per scenario and held-out year.
- `source_holdout_*.json`: the soybean source network scored on one 80/20 holdout.
- `statistical_tests_*.txt`: chosen architecture, feature and row counts, the soybean holdout, random-split t-tests, and LOYO median NRMSE by held-out year. LOYO gets no t-test because iterations within a year use the same rows and differ only in weight initialization.

### `Graphs/`
- `NN_Boxplot_*.png`, `NN_Boxplot_RMSE_*.png`: random-split NRMSE and RMSE by scenario.
- `NN_Density_*.png`: NRMSE distributions.
- `NN_LOYO_NRMSE_*.png`, `NN_LOYO_RMSE_*.png`: LOYO error by held-out year.
- `NN_Scatter_<Scenario>_*.png`: predicted vs. measured rice yield.

### `Models/`
- `optuna_best_params_*.json`: `best_params`, `best_value` (soybean validation MSE on standardized yield), and `n_trials_run`.
- `soybean_source_*.keras`: source network trained on all soybean rows. Its weights seed the warm scenarios.
- `rice_<scenario>_*.keras`: one model per scenario, refit on a fixed split (seed 0). These go with the scatter plots; they are not the best of the 100 splits. Saved as `.h5` on older TensorFlow.

### `Cleaned_Data/`
- `*_cleaned.csv`: encoded soybean and rice tables used for training.
- `feature_inventory_*.json`: row counts, yield summaries, and the shared, soybean-only, and rice-only feature lists.
- `common_features_*.csv`, `soy_only_features_*.csv`, `rice_only_features_*.csv`, `yield_description_*.csv`: the same information as flat files.

## What goes into the model

- Year and calendar-date columns are removed before encoding. The crop year is still used to build the LOYO folds. Day-of-year columns such as `days_from_year_start_Planting` stay.
- Harvest weight and harvest moisture are removed. Yield is harvested grain weight adjusted to a standard moisture, so either column would give the model part of its answer. Runs before September 2026 kept them in the rice full feature set.
- Rows with missing or non-positive yield are dropped, so `soy_rows` and `rice_rows` in the stats file can be lower than the raw row counts.
Runs before September 2026 (for example `20260520_135303`) only wrote NRMSE for the random splits, a flat Optuna JSON, and a single LOYO plot.

## Metrics

| Metric | Meaning |
| :--- | :--- |
| RMSE | Root mean squared error, in yield units. |
| NRMSE | RMSE divided by mean observed yield, as a percentage. Comparable across crops. |
| p-value | Independent two-sample t-test between two scenarios' error distributions. |

Re-run `Codes/VM_TransferLearning.py` to regenerate everything here after changing the data or the model.
