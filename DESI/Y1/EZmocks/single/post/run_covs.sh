#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=RascalC-Y1-EZmocks-recon-single
#SBATCH --array=20,25,34,44,49,50,52,58,59 # 10 mocks, SGC and NGC, no LRG but ELG and BGS
##SBATCH --array=0-59 # full job array: 10 mocks, SGC and NGC, LRG, ELG and BGS

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