#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# for i in {{0..9},14,15,22,23}; do
for i in {14,15}; do
    echo ID $i
    python -u run_covs.py -t $i
done