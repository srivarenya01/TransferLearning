# Codes

The scripts run in this order: weather extraction, weather aggregation, and transfer learning.

## `VM_DataExtractionNasa_v3.py`

Pulls daily NASA POWER weather for every plot straight from NASA's public S3 Zarr stores (`xarray` + `s3fs`, no API key). Variables: `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PRECTOTCORR`, and `ALLSKY_SFC_SW_DWN`. Writes one sheet per plot to an Excel workbook.

## `VM_WeatherAggregator.py`

Reads those workbooks and summarizes each plot's season: total rain, days with heavy rain or wind, average humidity, days above 80 °F, solar radiation totals, and growing degree days using crop-specific thresholds.

## `VM_TransferLearning.py`

Soybean to rice transfer learning. Optuna picks the architecture on soybean, five soybean source networks are trained on all soybean rows with different seeds, and their weights seed the rice networks. Four scenarios are compared: warm or cold start, on all rice features or only the features rice shares with soybean.

**Experiment grid.** One run covers every combination of:

| Factor | Levels | Constant |
| :--- | :--- | :--- |
| Shared-feature scaling | `soybean`, `per_crop` (see Scaling) | `VM_SHARED_SCALING` |
| Rice training size | 5%, 10%, 25%, 50%, 100% of the training rows | `TRAIN_FRACTIONS` |
| Random 80/20 splits | 50 | `NUM_RANDOM_SPLITS` |
| Leave-one-year-out (LOYO) | each rice year held out, 25 repeats per year | `NUM_LOYO_REPEATS` |
| Soybean source networks | 5; warm repeat `i` starts from source `i % 5` | `NUM_SOURCE_MODELS` |

That is about 5,000 rice fits. A training size keeps the same test rows and uses a subset of the rows of every larger size. Fits with the same repeat index share the split, the training rows, and the seed, so warm vs cold, full vs shared-only, and `per_crop` vs `soybean` are compared split by split. A mean-yield baseline (predict the average training yield) is scored on the same rows.

**Statistics.** Random splits get a Wilcoxon signed-rank test and the Nadeau and Bengio corrected resampled t-test, which allows for training sets that overlap between splits. LOYO repeats reuse the same rows and differ only in initialization and source network, so they get medians and win shares without p-values.

**Inputs.** `DATA/soybean_with_harvesting_only.csv` and `DATA/rice_with_harvesting_only.csv` at the repository root (not included in the repository). On Linux the folder name is case-sensitive, so it must be uppercase `DATA/`.

**Outputs.** Per-fit RMSE and NRMSE tables, summaries, split-by-split comparisons, learning curves, a soybean holdout score, the feature inventory, predicted vs. measured plots, and saved Keras models. See [`../Results/README.md`](../Results/README.md).

**What the model does not see.**
- Year and calendar-date columns. The year is still used to build the LOYO folds.
- Harvest weight and harvest moisture, since yield is calculated from them.
- Rows with missing or non-positive yield. These are dropped before imputation.

**Scaling.** Soybean features are scaled on soybean rows. Rice-only features and rice yield are scaled on each split's training rows. Rice's shared features are scaled one of two ways:

| Mode | Rice shared features | Meaning |
| :--- | :--- | :--- |
| `soybean` | Soybean's scaler | Transferred weights see the units they were trained on. Rice values far from soybean's range stay far from 0. |
| `per_crop` | Each split's rice training rows | Both crops are centered at 0. A value means "relative to that crop's usual season", not the same absolute weather. |

`VM_SHARED_SCALING` lists the modes to run, comma-separated (default `soybean,per_crop`). The modes are printed at start-up and written to `statistical_tests_*.txt` and `Models/run_config_*.json`.

**Training.** Warm and cold networks follow the same schedule: output layer only for 10 epochs, then all layers at half the learning rate with early stopping. Differences between them come from the starting weights.

**Encoding.** Categoricals are one-hot encoded without `drop_first`. Station and variety names differ between the crops, so those dummies are never shared.

**Seeds.** Every split, LOYO repeat, and Optuna trial is seeded. Source network `k` uses seed `42 + k`, so source 0 matches the single source network of earlier runs. Optuna trials run in threads, so the chosen architecture can still vary between runs.

**Parallelism.** The worker pool size is `SLURM_CPUS_PER_TASK` when set, otherwise every local core. Each worker is its own TensorFlow process limited to one op thread, so plan on 2 to 4 GB of memory per worker. Plots use the `Agg` backend and never open a window. For Grace, use [`../slurm/`](../slurm/README.md).

Harvest day of year (`days_from_year_start_harvest`) is dropped with harvest weight and moisture. It is only known at harvest.

## `VM_Convergence.py`

Sparse-data and epoch-curve companion to `VM_TransferLearning.py`. It reuses that script's cleaning, scaling, splits, training schedule, and statistics so the numbers are comparable.

For each of 50 random 80/20 rice splits the 20% test rows stay fixed. Networks train on 1%, 2%, 5%, 10%, 25%, 50%, and 100% of the 80% pool (nested subsets) and are also scored at 0% (the starting weights, no rice training). Test NRMSE is recorded after every epoch, which shows both how much rice data each start needs and how many epochs it takes to settle.

| Start | Weights | What it rules out |
| :--- | :--- | :--- |
| Warm | Soybean source network |  |
| Cold | Random |  |
| Cold_SmallInit | Random, rice-only input rows at the same small scale as Warm (full features only) | That Warm's small-data gain is only the quiet rice-only start |
| Shuffled | Soybean network trained on shuffled soybean yields | That any pretrained start of that size would help |
| Frozen | Soybean weights; middle hidden layers stay fixed | That fine-tuning the whole network is necessary when rice data is scarce |

Warm, Frozen, and Shuffled split `i` start from source `i % 5`. Cold_SmallInit is omitted on the shared-only feature set, which has no rice-only columns. Default scaling is `per_crop` (`VM_SHARED_SCALING`). Set `VM_BEST_PARAMS_FILE` to an Optuna JSON from an earlier run to skip the search.

**Outputs.** `convergence_results_*.csv`, `convergence_epochs_*.csv`, `convergence_summary_*.csv`, `convergence_comparisons_*.csv`, `convergence_data_efficiency_*.csv`, `convergence_report_*.txt`, plus learning-curve, per-epoch, and settled-epoch plots. See [`../Results/README.md`](../Results/README.md). On Grace: `sbatch slurm/run_convergence.slurm`.

## `VM_TransferIdeas.py`

Follow-up methods that try to make soybean useful, on the same rice splits. Harvest day of year is dropped. Default grid is 30 random splits, 10 LOYO repeats per year, and training shares 5%, 25%, and 100%. One soybean source per method (seed 42). Optional `VM_BEST_PARAMS_FILE` skips Optuna.

| Method | What it changes |
| :--- | :--- |
| `Warm_Common` / `Cold_Common` | Shared features, `per_crop` scaling (the reference) |
| `Warm_Quantile` / `Cold_Quantile` | Shared features mapped onto soybean's distribution |
| `Warm_Overlap` / `Cold_Overlap` | Only shared columns whose rice values sit inside soybean's range |
| `Warm_Matched` | Soybean trained on the 10% of soybean rows closest to rice in GDD and planting day (a strict overlap box contains none) |
| `Cold_SoyScore` | Cold_Common plus the soybean network's score on that rice row |
| `Joint_CropFlag` | One network on soybean and rice together, with a crop flag |
| `Warm_Full` / `Cold_Full` / `Cold_SmallInit` | All rice columns; SmallInit is random with tiny rice-only weights |
| `Cold_Full_SoyScore` | Cold_Full plus the soybean score |

**Outputs.** `ideas_results_*.csv`, `ideas_summary_*.csv`, `ideas_comparisons_*.csv`, `ideas_report_*.txt`, and `Ideas_LearningCurve_*.png`. On Grace: `sbatch slurm/run_transfer_ideas.slurm`.

## `VM_EnrichCommonFeatures.py`

Builds thicker *shared* feature tables for a transfer sensitivity: SSURGO topsoil at each unique lat/lon, NASA POWER daily weather for each planting–harvest window, and identical stress indices for both crops (flood / heavy rain / dry spells / heat / frost / shared-base GDD at 10 °C / VPD proxy). Also adds harvest-day POWER snapshots so soybean gets rice’s `Temperature_H`-style columns. Writes `Data/VM_SOYBEAN_TESTING_STUFF.csv` and `Data/VM_RICE_TESTING_STUFF.csv` (gitignored; scp to Grace `DATA/`). Cache under `Data/enrichment_cache/`.

Point the main pipeline at those files with `VM_SOY_CSV` / `VM_RICE_CSV`, or `sbatch slurm/run_enriched_transfer.slurm`.

## `VM_SmokeTransferEnriched.py`

Tiny local sklearn Warm vs Cold check on the TESTING_STUFF tables (TensorFlow optional). Not a paper result — only a laptop sanity check before Grace.
