# Slurm job for TAMU HPRC Grace

`run_transfer_learning.slurm` runs the whole `Codes/VM_TransferLearning.py` pipeline on one Grace compute node: the Optuna search, the soybean holdout, 4 scenarios x 100 random splits, and LOYO (3 years x 4 scenarios x 100 iterations). It sets up and maintains its own virtual environment, so one `sbatch` is enough.

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
squeue -u $USER                                   # job state
tail -f slurm/logs/vm_transfer_learning_<jobid>.out
seff <jobid>                                      # CPU and memory use once it finishes
```

Results go to `Results/` (see `Results/README.md`).

## What the job does

1. Checks that `Codes/VM_TransferLearning.py`, `requirements.txt`, and both `DATA/` files exist, and stops right away if any are missing.
2. Runs `module purge`, then loads `GCCcore/12.3.0 Python/3.11.3` and `WebProxy`. Grace compute nodes have no internet access; `WebProxy` is how HPRC lets `pip` reach PyPI from a job.
3. Creates the venv at `$SCRATCH/venvs/vm_transfer_learning` if it doesn't exist, and runs `pip install -r requirements.txt`. It stores a hash of `requirements.txt` and the module list in the venv, so later submits skip the install unless either changes. Then it prints the package versions and stops if matplotlib is older than 3.9 (needed for the boxplot `tick_labels` argument).
4. Limits TensorFlow, OpenMP, and BLAS to one thread per worker and starts the pipeline. With `SLURM_CPUS_PER_TASK=48` that means 48 worker processes and 48 Optuna threads.

## Resources

| Setting | Value | Reason |
| :--- | :--- | :--- |
| Partition | `short` | 2 hour limit, usually the shortest wait for one node. |
| Time | `02:00:00` | The target is about 1 hour; the rest is headroom so a slow run isn't killed. |
| Node | 1 node, 48 cores | A full standard Grace node (two 24-core Cascade Lake CPUs). |
| Memory | `360G` | All the usable RAM on a 384 GB node. The pipeline needs roughly 2 to 4 GB per worker, so 100 to 200 GB in total. |

The first submit spends a few extra minutes on `pip install`, mostly TensorFlow. Later submits reuse the venv.

**How long it takes.** The run is about 1,650 small network fits. With 48 running at once, the random-split and LOYO stages should take around 20 to 40 minutes. Optuna runs its 50 trials as threads in the main process, and Python's GIL limits how much those threads gain from 48 cores, so expect it to be the slowest stage for the work it does. The log prints the start time and the total duration. If a run hits the 2 hour limit, resubmit on `medium`:

```bash
sbatch --partition=medium --time=06:00:00 slurm/run_transfer_learning.slurm
```

## Overrides

Pass these with `--export=ALL,...` when submitting:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `VENV_DIR` | `$SCRATCH/venvs/vm_transfer_learning` | Where the venv lives. |
| `PYTHON_MODULES` | `GCCcore/12.3.0 Python/3.11.3` | Lmod modules that provide `python3`. |
| `FORCE_REINSTALL` | `0` | Set to `1` to delete and rebuild the venv. |

```bash
sbatch --export=ALL,FORCE_REINSTALL=1 slurm/run_transfer_learning.slurm
sbatch --export=ALL,PYTHON_MODULES="GCCcore/13.2.0 Python/3.11.5" slurm/run_transfer_learning.slurm
```

The job is charged to your default allocation. Uncomment the `--mail-*` lines in the script for email notifications.

## Cleanup

```bash
rm -rf $SCRATCH/venvs/vm_transfer_learning     # the next submit rebuilds it
rm -f slurm/logs/*.out slurm/logs/*.err
```