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
4. `VM_Convergence.py` (optional) holds the rice test rows fixed and grows the training share from 0% to 100%, scoring Warm vs Cold (and three controls) after every epoch.
5. `VM_TransferIdeas.py` (optional) tries quantile alignment, overlap features, climate-matched soybean, a soybean score feature, and a joint crop-flag network.

A full training run (Optuna search, five soybean source networks, and about 5,000 rice fits across two scaling modes, five training sizes, random splits, and leave-one-year-out) needs a cluster node. The worker pool follows `SLURM_CPUS_PER_TASK`. On Grace, run `sbatch slurm/run_transfer_learning.slurm` from the repository root; it sets up the venv and runs everything on one node. For the sparse-data / epoch-curve run, `sbatch slurm/run_convergence.slurm`. For the follow-up transfer methods, `sbatch slurm/run_transfer_ideas.slurm`. See [`slurm/README.md`](slurm/README.md).

## Tools

| Purpose | Libraries |
| :--- | :--- |
| Neural networks | TensorFlow, Keras |
| Hyperparameter search | Optuna |
| Statistics and plots | SciPy, Matplotlib, Seaborn |
| Data handling | Pandas, Xarray, S3FS |
| Weather source | NASA POWER |
