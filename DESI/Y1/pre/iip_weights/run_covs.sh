#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-Y1-v1.5-unblinded-iip-weights
#SBATCH --array=8-11 # only ELG
##SBATCH --array=0-5,8-11,14-17 # full job array: excluding LRG and ELG full ranges

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# GSL needed by the C++ code should already be loaded in cosmodesi
# module load gsl

# OpenMP settings
# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=256

# Hopefully let numpy use all threads
# export NUMEXPR_MAX_THREADS=256
# Limit OpenBLAS thread usage (for jackknife assignment, error otherwise)
# export OPENBLAS_NUM_THREADS=1

python -u run_covs.py $SLURM_ARRAY_TASK_ID