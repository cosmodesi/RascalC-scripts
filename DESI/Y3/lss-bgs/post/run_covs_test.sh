#!/bin/bash

CAMPAIGN=${CAMPAIGN:-nonkp_bright_faint_priority}

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

args=(--campaign "$CAMPAIGN")
if [ -n "${COMPMD:-}" ]; then args+=(--compmd "$COMPMD"); fi
if [ -n "${VERSION:-}" ]; then args+=(--version "$VERSION"); fi
if [ -n "${DEFAULT_N_LOOPS:-}" ]; then args+=(--default-n-loops "$DEFAULT_N_LOOPS"); fi

python -u run_covs.py --list-cases "${args[@]}"

ncase=$(python - <<'INNER_PY'
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()))
from bgs_cases import array_length
print(array_length(os.environ.get('CAMPAIGN', 'nonkp_bright_faint_priority'), compmd=os.environ.get('COMPMD') or None, version=os.environ.get('VERSION') or None))
INNER_PY
)

for ((i=0; i<ncase; i++)); do
  echo "ID $i"
  python -u run_covs.py -t "$i" "${args[@]}"
done
