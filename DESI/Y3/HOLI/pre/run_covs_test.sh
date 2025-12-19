#!/bin/bash

# load cosmodesi 2025_05 (old test) environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_05

for i in {{0..9},22,23}; do
    echo ID $i
    python -u run_covs.py -t $i
done