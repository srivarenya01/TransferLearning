# Slurm job for TAMU HPRC Grace

`run_transfer_learning.slurm` runs the whole `Codes/VM_TransferLearning.py` pipeline on one Grace compute node: the Optuna search, the soybean holdout, five soybean source networks, and about 5,000 rice fits (2 scaling modes x 4 scenarios x 5 training sizes x 50 random splits and 3 held-out years x 25 repeats). It sets up and maintains its own virtual environment, so one `sbatch` is enough.

`run_convergence.slurm` is a thin wrapper around that job. It runs `Codes/VM_Convergence.py` instead: 50 random 80/20 rice splits, a growing share of the 80% pool (1% through 100%, plus a zero-shot score before any rice training), and five starting points (Warm, Cold, Cold_SmallInit, Shuffled, Frozen). Default scaling is `per_crop` only, about 3,600 rice fits. Submit it the same way, from the repository root.

## Before the first submit

1. Clone the repository into scratch. Home has a small quota, and the venv and results live next to the code.
   ```bash
   cd /scratch/user/$USER
   git clone https://github.com/srivarenya01/TransferLearning.git
   cd TransferLearning
   ```
2. Upload the two input files to `DATA/` (uppercase) at the repository root. They are not in the repository.
3. Optional: check that the Python module still exists with `module spider Python/3.11.3`. If HPRC has retired it, pick another from `module spider Python` and pass it as `PYTHON_MODULES` (see Overrides).

## Submit

Submit from the repository root. The job finds everything relative to `$SLURM_SUBMIT_DIR`, and Slurm writes logs to `slurm/logs/`, which has to exist before submitting (git keeps it through `.gitkeep`).

```bash
sbatch slurm/run_transfer_learning.slurm
sbatch slurm/run_convergence.slurm                # sparse-data / epoch-curve run
sbatch slurm/run_transfer_ideas.slurm             # quantile, overlap, matched, score, joint
# Enriched soil + shared stress CSVs (scp DATA/VM_*_TESTING_STUFF.csv first):
sbatch --export=ALL,VM_BEST_PARAMS_FILE=Results/Models/optuna_best_params_20260928_002442.json \
    slurm/run_enriched_transfer.slurm
squeue -u $USER                                   # job state
tail -f slurm/logs/vm_transfer_learning_<jobid>.out
tail -f slurm/logs/vm_convergence_<jobid>.out
tail -f slurm/logs/vm_enriched_transfer_<jobid>.out
seff <jobid>                                      # CPU and memory use once it finishes
```

`VM_SOY_CSV` / `VM_RICE_CSV` override the default harvesting-only inputs (also used by `run_enriched_transfer.slurm`).

Results go to `Results/` (see `Results/README.md`).

## What the job does

1. Checks that `Codes/$VM_SCRIPT` (default `VM_TransferLearning.py`), `requirements.txt`, and both `DATA/` files exist, and stops right away if any are missing.
2. Runs `module purge`, then loads `GCCcore/12.3.0 Python/3.11.3` and `WebProxy`. Grace compute nodes have no internet access; `WebProxy` is how HPRC lets `pip` reach PyPI from a job.
3. Creates the venv at `$SCRATCH/venvs/vm_transfer_learning` if it doesn't exist, and runs `pip install -r requirements.txt`. It stores a hash of `requirements.txt` and the module list in the venv, so later submits skip the install unless either changes. Then it imports every package and prints its version, so a broken install stops the job before any training.
4. Limits TensorFlow, OpenMP, and BLAS to one thread per worker and starts the pipeline. With `SLURM_CPUS_PER_TASK=48` that means 48 worker processes and 48 Optuna threads.

## Resources

| Setting | Value | Reason |
| :--- | :--- | :--- |
| Partition | `short` | 2 hour limit, usually the shortest wait for one node. |
| Time | `02:00:00` | The target is about 1 hour; the rest is headroom so a slow run isn't killed. |
| Node | 1 node, 48 cores | A full standard Grace node (two 24-core Cascade Lake CPUs). |
| Memory | `360G` | All the usable RAM on a 384 GB node. The pipeline needs roughly 2 to 4 GB per worker, so 100 to 200 GB in total. |

The first submit spends a few extra minutes on `pip install`, mostly TensorFlow. Later submits reuse the venv.

**How long it takes.** Optuna takes about 8 minutes: it runs its 50 trials as threads in the main process, and Python's GIL limits how much those threads gain from 48 cores. The rice fits come next. Earlier runs took about 20 minutes for 1,600 full-size fits. The transfer-learning grid has about 5,000 fits, but most of them train on 5% to 50% of the rows, so expect roughly 30 to 45 minutes, and about 45 to 60 minutes for the whole job. The convergence job is about 3,600 fits with a test score after every epoch; expect a similar 45 to 60 minutes. The log prints elapsed minutes after each stage. Finished rows are saved to `Results/Errors/partial_results_*.csv` or `convergence_partial_results_*.csv` every 500 fits, so a run that hits the time limit still leaves its results. If that happens, resubmit on `medium`:

```bash
sbatch --partition=medium --time=06:00:00 slurm/run_transfer_learning.slurm
sbatch --partition=medium --time=06:00:00 slurm/run_convergence.slurm
sbatch --partition=medium --time=06:00:00 slurm/run_transfer_ideas.slurm
```

## Overrides

Pass these with `--export=ALL,...` when submitting:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `VENV_DIR` | `$SCRATCH/venvs/vm_transfer_learning` | Where the venv lives. |
| `PYTHON_MODULES` | `GCCcore/12.3.0 Python/3.11.3` | Lmod modules that provide `python3`. |
| `FORCE_REINSTALL` | `0` | Set to `1` to delete and rebuild the venv. |
| `VM_SHARED_SCALING` | `soybean,per_crop` (`per_crop` for `run_convergence.slurm`) | Comma-separated scaling modes for rice's shared features (see `Codes/README.md`). Any other value stops the job before setup. |
| `VM_SCRIPT` | `VM_TransferLearning.py` | Python file under `Codes/` to run. `run_convergence.slurm` sets `VM_Convergence.py`. |
| `VM_BEST_PARAMS_FILE` | empty | Path to an Optuna `best_params` JSON from an earlier run. Used only by `VM_Convergence.py`; empty runs Optuna. |

```bash
sbatch --export=ALL,VM_SHARED_SCALING=per_crop slurm/run_transfer_learning.slurm   # one mode only, about half the fits
sbatch --export=ALL,VM_BEST_PARAMS_FILE=Results/Models/optuna_best_params_20260928_002442.json slurm/run_convergence.slurm
sbatch --export=ALL,FORCE_REINSTALL=1 slurm/run_transfer_learning.slurm
sbatch --export=ALL,PYTHON_MODULES="GCCcore/13.2.0 Python/3.11.5" slurm/run_transfer_learning.slurm
```

The job is charged to your default allocation. Uncomment the `--mail-*` lines in the script for email notifications.

## Cleanup

```bash
rm -rf $SCRATCH/venvs/vm_transfer_learning     # the next submit rebuilds it
rm -f slurm/logs/*.out slurm/logs/*.err
```