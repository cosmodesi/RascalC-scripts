#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Only valid IDs for current run_covs.py config:
# id=0 -> SGC, id=1 -> NGC  (since tracers/zs lists have length 1)

for i in 0 1; do
  echo "ID $i"
  python -u run_covs.py -t "$i"
done