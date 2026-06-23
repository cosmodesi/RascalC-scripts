#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH="${PYTHONPATH}:$HOME/Repos/desi-y3-kp/"
export PYTHONPATH="${PYTHONPATH}:$HOME/Repos/LSS/py/"
export DESICFS="${DESICFS:-/dvs_ro/cfs/cdirs/desi}"

for i in 0 1; do
    echo ID $i
    python -u run_covs.py -t $i
done
