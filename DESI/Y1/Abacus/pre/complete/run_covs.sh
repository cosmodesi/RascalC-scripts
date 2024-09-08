#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-Y1-Abacus-v4_2-complete
#SBATCH --array=4,5,10,11 # only higher-z ranges of LRG and ELG
##SBATCH --array=0-5,8-11,14-17 # full job array: excluding LRG and ELG full ranges

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_covs.py $SLURM_ARRAY_TASK_ID