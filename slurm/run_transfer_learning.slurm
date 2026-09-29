#!/bin/bash
#
# Soybean -> rice transfer-learning pipeline (Codes/VM_TransferLearning.py) on TAMU HPRC Grace.
#
# Submit from the repository root (the directory that contains Codes/, DATA/, slurm/):
#     cd /scratch/user/$USER/<repo>
#     sbatch slurm/run_transfer_learning.slurm
#
# Overridable at submit time, e.g. `sbatch --export=ALL,VENV_DIR=/scratch/user/$USER/tl_venv slurm/...`:
#     VENV_DIR        Virtual environment location   (default: $SCRATCH/venvs/vm_transfer_learning)
#     PYTHON_MODULES  Lmod modules providing Python   (default: "GCCcore/12.3.0 Python/3.11.3")
#     FORCE_REINSTALL Set to 1 to rebuild the venv from scratch
#     VM_SHARED_SCALING  Comma-separated scaling modes to run for rice's shared features
#                        (default: soybean,per_crop)
#     VM_SCRIPT       Python script under Codes/ to run   (default: VM_TransferLearning.py;
#                     slurm/run_convergence.slurm sets VM_Convergence.py)
#
# See slurm/README.md for details.

#SBATCH --job-name=vm_transfer_learning
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

# Lmod references unset variables internally, so `set -u` is intentionally not used.
set -eo pipefail

VENV_DIR="${VENV_DIR:-${SCRATCH}/venvs/vm_transfer_learning}"
PYTHON_MODULES="${PYTHON_MODULES:-GCCcore/12.3.0 Python/3.11.3}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
export VM_SHARED_SCALING="${VM_SHARED_SCALING:-soybean,per_crop}"

VM_SCRIPT="${VM_SCRIPT:-VM_TransferLearning.py}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="${REPO_DIR}/Codes/${VM_SCRIPT}"
REQUIREMENTS="${REPO_DIR}/requirements.txt"
REQUIRED_DATA=(
    "${REPO_DIR}/DATA/soybean_with_harvesting_only.csv"
    "${REPO_DIR}/DATA/rice_with_harvesting_only.csv"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

fail() { log "ERROR: $*" >&2; exit 1; }

preflight() {
    [[ "${VM_SCRIPT}" != */* ]] || fail "VM_SCRIPT must be a file name under Codes/, got '${VM_SCRIPT}'."
    [[ -f "${SCRIPT_PATH}" ]] || fail "Missing ${SCRIPT_PATH}. Submit this job from the repository root."
    [[ -f "${REQUIREMENTS}" ]] || fail "Missing ${REQUIREMENTS}."
    local f
    for f in "${REQUIRED_DATA[@]}"; do
        [[ -s "${f}" ]] || fail "Missing or empty input file: ${f}"
    done
    [[ -n "${SCRATCH}" ]] || fail "\$SCRATCH is not set; set VENV_DIR explicitly."
    local mode modes
    IFS=',' read -ra modes <<< "${VM_SHARED_SCALING}"
    (( ${#modes[@]} > 0 )) || fail "VM_SHARED_SCALING is empty."
    for mode in "${modes[@]}"; do
        case "${mode}" in
            soybean|per_crop) ;;
            *) fail "VM_SHARED_SCALING entries must be soybean or per_crop, got '${mode}'." ;;
        esac
    done
}

load_modules() {
    module purge
    # shellcheck disable=SC2086
    module load ${PYTHON_MODULES}
    # Grace compute nodes have no direct internet; WebProxy is required for pip.
    module load WebProxy
    module list 2>&1
}

# Rebuilds the venv only when it is missing, broken, or requirements.txt changed.
setup_venv() {
    local stamp="${VENV_DIR}/.requirements.sha256"
    local want_hash
    want_hash="$( (sha256sum "${REQUIREMENTS}"; echo "${PYTHON_MODULES}") | sha256sum | cut -d' ' -f1)"

    if [[ "${FORCE_REINSTALL}" == "1" && -d "${VENV_DIR}" ]]; then
        log "FORCE_REINSTALL=1: removing ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
    fi

    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        log "Creating venv at ${VENV_DIR}"
        mkdir -p "$(dirname "${VENV_DIR}")"
        python3 -m venv "${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${want_hash}" ]]; then
        log "Venv is up to date with requirements.txt; skipping pip install"
    else
        log "Installing requirements into ${VENV_DIR}"
        export PIP_CACHE_DIR="${SCRATCH}/.cache/pip"
        python -m pip install --upgrade pip
        python -m pip install -r "${REQUIREMENTS}"
        echo "${want_hash}" > "${stamp}"
    fi

    python - <<'PY'
import sys
import matplotlib, numpy, optuna, pandas, scipy, seaborn, sklearn, tensorflow as tf
print(f"python {sys.version.split()[0]} | tensorflow {tf.__version__} | numpy {numpy.__version__} | "
      f"pandas {pandas.__version__} | sklearn {sklearn.__version__} | optuna {optuna.__version__} | "
      f"matplotlib {matplotlib.__version__} | scipy {scipy.__version__} | seaborn {seaborn.__version__}")
PY
}

main() {
    log "Job ${SLURM_JOB_ID:-local} on $(hostname) | cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} | repo=${REPO_DIR}"
    preflight
    load_modules
    setup_venv

    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=2
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    cd "${REPO_DIR}"
    log "Starting Codes/${VM_SCRIPT} with ${SLURM_CPUS_PER_TASK:-$(nproc)} workers | shared scaling=${VM_SHARED_SCALING}"
    local start=${SECONDS}
    python "${SCRIPT_PATH}"
    log "Finished in $(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s. Outputs: ${REPO_DIR}/Results"
}

main "$@"
