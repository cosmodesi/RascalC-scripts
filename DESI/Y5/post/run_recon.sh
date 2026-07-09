#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 04:00:00
#SBATCH -q regular
#SBATCH --time=4:00:00
#SBATCH --job-name=RascalC-Y5-data-recon

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering

srun -N 1 -n 4 python -u run_recon.py
