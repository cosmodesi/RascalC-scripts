#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --time=8:00:00
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --job-name=LRG+ELG-separate-xi-jack-recon

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_xi_jack.py
