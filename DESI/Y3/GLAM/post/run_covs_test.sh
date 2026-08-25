#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# module unload desi-clustering # use locally installed desi-clustering if uncommented, otherwise use the global one from cosmodesi environment

for i in {0..13}; do
    echo ID $i
    python -u run_covs.py -t $i
done
