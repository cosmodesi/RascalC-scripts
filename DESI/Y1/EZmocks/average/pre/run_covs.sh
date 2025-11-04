#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=RascalC-Y1-EZmocks-avg
#SBATCH --array=0-9,12-13 # full job array: excluding BGS due to different directory structure

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_covs.py $SLURM_ARRAY_TASK_ID