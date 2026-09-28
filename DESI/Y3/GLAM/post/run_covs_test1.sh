#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# module unload desi-clustering # use locally installed desi-clustering if uncommented, otherwise use the global one from cosmodesi environment

for i in {0,1,3,4,10..15}; do # BGS_BRIGHT-21.35 0.1-0.4, LRG2, ELG2, LRG3+ELG1, QSO 0.8-2.1
    echo ID $i
    python -u run_covs.py -t $i --mock_id 151
done
