#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 01:00:00
#SBATCH -q regular
#SBATCH --job-name=RascalC-Y5-data-recon

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering

srun -N 1 -n 4 python -u run_recon.py
