#!/bin/bash
#
# Transfer-learning grid on enriched TESTING_STUFF_v2 CSVs
# (SSURGO soil + shared season stress + GDD-stage S1–S4 + high-radiation).
#
# Submit from the repository root after scp'ing:
#     DATA/VM_SOYBEAN_TESTING_STUFF_v2.csv
#     DATA/VM_RICE_TESTING_STUFF_v2.csv
#
# Fresh Optuna (do NOT pass VM_BEST_PARAMS_FILE — soy input dim changed again):
#     sbatch slurm/run_enriched_transfer.slurm
#
# Reuses slurm/run_transfer_learning.slurm for modules, venv, and workers.

#SBATCH --job-name=vm_enriched_transfer
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=srivarenya@tamu.edu

export VM_SCRIPT="${VM_SCRIPT:-VM_TransferLearning.py}"
export VM_SOY_CSV="${VM_SOY_CSV:-DATA/VM_SOYBEAN_TESTING_STUFF_v2.csv}"
export VM_RICE_CSV="${VM_RICE_CSV:-DATA/VM_RICE_TESTING_STUFF_v2.csv}"
# Both scalings so the TL search is comparable to the main grid.
export VM_SHARED_SCALING="${VM_SHARED_SCALING:-soybean,per_crop}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${REPO_DIR}/slurm/run_transfer_learning.slurm"
