#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# for i in {0..3}; do # full job array
for i in {2,3}; do # BGS_BRIGHT-21.35 0.1-0.4
    echo ID $i
    python -u run_covs.py -t $i
done