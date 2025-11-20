#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

for i in {0..5}; do
    echo ID $i
    python -u run_covs.py -t $i
done