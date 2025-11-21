#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-Y3-HOLI
##SBATCH --array=0-23 # full job array
##SBATCH --array=0-11,14,15,22,23 # only -21.35 z0.1-0.4 for BGS, and all other tracers
##SBATCH --array=0-5 # only LRG
#SBATCH --array=0 # re-run the badly converged case

# load cosmodesi test environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

python -u run_covs.py $SLURM_ARRAY_TASK_ID