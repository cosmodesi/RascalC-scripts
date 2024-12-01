#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

for i in {24,25}; do
    echo ID $i
    python -u run_covs.py -t $i
done