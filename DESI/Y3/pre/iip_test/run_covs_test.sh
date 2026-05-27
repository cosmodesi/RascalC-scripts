#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering # use the user-installed version with the latest changes

for i in {6..9}; do
    echo ID $i
    python -u run_covs.py -t $i # the test process is killed while reading the full catalog, probably runs out of memory with the tighter limit on NERSC login nodes
done