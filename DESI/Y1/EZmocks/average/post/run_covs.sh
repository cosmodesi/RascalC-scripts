#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=RascalC-Y1-EZmocks-recon
#SBATCH --array=10,11 # LRG+ELG
##SBATCH --array=0-11,13-14 # full job array: excluding BGS due to different directory structure

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_covs.py $SLURM_ARRAY_TASK_ID