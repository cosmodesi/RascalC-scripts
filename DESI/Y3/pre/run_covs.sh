#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-Y3-v1.1-BAO-unblinded
##SBATCH --array=0-23 # full job array
##SBATCH --array=0-11,14,15,22,23 # only -21.35 z0.1-0.4 for BGS, and all other tracers
#SBATCH --array=14 # relaunch BGS_BRIGHT-21.35 z0.1-0.4 SGC

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_covs.py $SLURM_ARRAY_TASK_ID