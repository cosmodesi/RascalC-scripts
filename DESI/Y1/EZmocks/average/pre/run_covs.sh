#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=RascalC-Y1-EZmocks-avg
#SBATCH --array=0-4,8-11,16-17 # excluding LRG NGC z0.8-1.1, which has already been run
##SBATCH --array=0-5,8-11,16-17 # full job array: excluding LRG and ELG full ranges, and BGS due to different directory structure

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