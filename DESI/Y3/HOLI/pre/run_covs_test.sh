#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering # temporarily use custom desi-clustering

# for i in {{0..9},14,15,22,23}; do
for i in {{0..9},22,23}; do # skip BGS_BRIGHT-21.35 z0.1-0.4 for now
    echo ID $i
    python -u run_covs.py -t $i
done