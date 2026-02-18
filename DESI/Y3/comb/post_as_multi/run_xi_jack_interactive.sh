#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

srun --ntasks=1 --constraint=gpu --time=4:00:00 --gpus=4 --qos=interactive --account=desi_g --job-name=LRG+ELG-separate-xi-jack-recon python -u run_xi_jack.py