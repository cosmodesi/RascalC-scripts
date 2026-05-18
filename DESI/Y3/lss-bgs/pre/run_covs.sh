#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-LSS-BGS
# Edit this array to match CAMPAIGN; inspect with: CAMPAIGN=<name> python run_covs.py --list-cases
#SBATCH --array=0-17 # nonkp_bright_faint_priority
##SBATCH --array=0-13 # nonkp_bright_compare or legacy
##SBATCH --array=0-29 # pip_priority
##SBATCH --array=0-19 # pip_20p7

CAMPAIGN=${CAMPAIGN:-nonkp_bright_faint_priority}

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

args=(--campaign "$CAMPAIGN")
if [ -n "${COMPMD:-}" ]; then args+=(--compmd "$COMPMD"); fi
if [ -n "${VERSION:-}" ]; then args+=(--version "$VERSION"); fi
if [ -n "${DEFAULT_N_LOOPS:-}" ]; then args+=(--default-n-loops "$DEFAULT_N_LOOPS"); fi

python -u run_covs.py "$SLURM_ARRAY_TASK_ID" "${args[@]}"
