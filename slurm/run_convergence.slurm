#!/bin/bash
#
# Warm vs cold convergence on sparse rice data (Codes/VM_Convergence.py) on TAMU HPRC Grace.
#
# Submit from the repository root:
#     cd /scratch/user/$USER/<repo>
#     sbatch slurm/run_convergence.slurm
#
# Reuses slurm/run_transfer_learning.slurm for modules, venv, and data checks.
# Overridable at submit time (in addition to that script's variables):
#     VM_SHARED_SCALING     default per_crop (the mode that won or tied)
#     VM_BEST_PARAMS_FILE   optional JSON with Optuna best_params; empty runs Optuna
#
# See slurm/README.md for details.

#SBATCH --job-name=vm_convergence
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

export VM_SCRIPT="${VM_SCRIPT:-VM_Convergence.py}"
export VM_SHARED_SCALING="${VM_SHARED_SCALING:-per_crop}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${REPO_DIR}/slurm/run_transfer_learning.slurm"
