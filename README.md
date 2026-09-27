# Cross-Crop Yield Prediction via Transfer Learning

Predicting rice yield with a neural network pre-trained on soybean. Soybean has more trial data, rice has less, and both are paired with NASA POWER weather. The question is whether starting the rice network from soybean weights (warm start) beats training it from scratch (cold start).

## Repository layout

- [`Codes/`](Codes/README.md): weather extraction, weather aggregation, and the transfer-learning pipeline.
- [`Results/`](Results/README.md): error tables, plots, saved models, and the feature inventory from each run.
- [`slurm/`](slurm/README.md): batch job for TAMU HPRC Grace.

## Getting started

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

The weather extraction step also needs `xarray` and `s3fs`.

## Running the pipeline

The scripts live in `Codes/` and run in this order:

1. `VM_DataExtractionNasa_v3.py` downloads daily NASA POWER weather for each plot.
2. `VM_WeatherAggregator.py` turns the daily weather into season totals and counts.
3. `VM_TransferLearning.py` trains and evaluates the four scenarios from `DATA/soybean_with_harvesting_only.csv` and `DATA/rice_with_harvesting_only.csv`.

A full training run (Optuna search, 400 random-split fits, 1,200 leave-one-year-out fits) needs a cluster node. The worker pool follows `SLURM_CPUS_PER_TASK`. On Grace, run `sbatch slurm/run_transfer_learning.slurm` from the repository root; it sets up the venv and runs everything on one node. See [`slurm/README.md`](slurm/README.md).

## Tools

| Purpose | Libraries |
| :--- | :--- |
| Neural networks | TensorFlow, Keras |
| Hyperparameter search | Optuna |
| Statistics and plots | SciPy, Matplotlib, Seaborn |
| Data handling | Pandas, Xarray, S3FS |
| Weather source | NASA POWER |
