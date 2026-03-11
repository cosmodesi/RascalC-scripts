#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# for i in {0..11}; do # full list of job IDs
for i in {2,3}; do
    echo ID $i
    python -u run_covs.py -t $i
done