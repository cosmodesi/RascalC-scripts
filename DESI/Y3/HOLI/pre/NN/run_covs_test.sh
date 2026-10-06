#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# module unload desi-clustering # temporarily use custom desi-clustering

for i in {8..11}; do # ELG
    echo ID $i
    python -u run_covs.py -t $i
done