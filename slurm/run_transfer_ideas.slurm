#!/bin/bash
#
# Follow-up transfer methods (Codes/VM_TransferIdeas.py) on TAMU HPRC Grace.
#
# Submit from the repository root:
#     sbatch slurm/run_transfer_ideas.slurm
#
# Reuses slurm/run_transfer_learning.slurm for modules, venv, and data checks.
# VM_BEST_PARAMS_FILE can point at an earlier Optuna JSON to skip the search.

#SBATCH --job-name=vm_transfer_ideas
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

export VM_SCRIPT="${VM_SCRIPT:-VM_TransferIdeas.py}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${REPO_DIR}/slurm/run_transfer_learning.slurm"
