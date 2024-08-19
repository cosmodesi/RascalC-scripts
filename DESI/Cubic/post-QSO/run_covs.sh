#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=RascalC-cubic-QSO-post
#SBATCH --array=1-4 # all HODs

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

python -u run_covs.py $SLURM_ARRAY_TASK_ID