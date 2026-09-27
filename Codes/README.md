# Codes

The scripts run in this order: weather extraction, weather aggregation, and transfer learning.

## `VM_DataExtractionNasa_v3.py`

Pulls daily NASA POWER weather for every plot straight from NASA's public S3 Zarr stores (`xarray` + `s3fs`, no API key). Variables: `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PRECTOTCORR`, and `ALLSKY_SFC_SW_DWN`. Writes one sheet per plot to an Excel workbook.

## `VM_WeatherAggregator.py`

Reads those workbooks and summarizes each plot's season: total rain, days with heavy rain or wind, average humidity, days above 80 °F, solar radiation totals, and growing degree days using crop-specific thresholds.

## `VM_TransferLearning.py`

Soybean to rice transfer learning. Optuna picks the architecture on soybean, a source network is trained on all soybean rows, and its weights seed the rice networks. Four scenarios are compared: warm or cold start, on all rice features or only the features rice shares with soybean. Each scenario is scored on 100 random 80/20 splits and on leave-one-year-out (LOYO) folds over the rice years.

**Inputs.** `DATA/soybean_with_harvesting_only.csv` and `DATA/rice_with_harvesting_only.csv` at the repository root (not included in the repository). On Linux the folder name is case-sensitive, so it must be uppercase `DATA/`.

**Outputs.** RMSE and NRMSE tables, a soybean holdout score, the feature inventory, predicted vs. measured plots, saved Keras models, and t-tests. See [`../Results/README.md`](../Results/README.md).

**What the model does not see.**
- Year and calendar-date columns. The year is still used to build the LOYO folds.
- Harvest weight and harvest moisture, since yield is calculated from them.
- Rows with missing or non-positive yield. These are dropped before imputation.

**Scaling.** Shared features are always in soybean units (scaler fit on soybean only), so transferred weights seee crops, so those dummies are never shared.

**Training.** Warm and cold networks follow the same schedule: output layer only for 10 epochs, then all layers at half the learning rate with early stopping. Differences between them come from the units they were trained on. Rice-only features and rice yield are scaled on each split's training rows.

**Encoding.** Categoricals are one-hot encoded without `drop_first`. Station and variety names differ between th the starting weights.

**Seeds.** Every split, LOYO repeat, and Optuna trial is seeded. `GLOBAL_SEED = 42` covers the sampler, the source network, and the weights for rice-only features. Optuna trials run in threads, so their order can still vary between runs.

**Parallelism.** The worker pool size is `SLURM_CPUS_PER_TASK` when set, otherwise every local core. Each worker is its own TensorFlow process limited to one op thread, so plan on 2 to 4 GB of memory per worker. Plots use the `Agg` backend and never open a window. For Grace, use [`../slurm/`](../slurm/README.md).
