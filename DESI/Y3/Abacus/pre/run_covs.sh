#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-Y3-Abacus-v2
##SBATCH --array=0-25 # full job array
##SBATCH --array=0-15,22-25 # only -21.5 and -21.35 z0.1-0.4 and BGS_ANY-02 for BGS, and all other tracers
#SBATCH --array=24,25 # BGS_ANY-02

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_covs.py $SLURM_ARRAY_TASK_ID