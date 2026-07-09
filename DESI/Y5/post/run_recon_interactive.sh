#!/bin/bash
# Interactive version of run_recon.sh
# Usage: salloc --account desi_g -C gpu&hbm80g -N 1 --gpus 4 -t 04:00:00 -q interactive
# Then: bash run_recon_interactive.sh

set -e
SECONDS=0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering

JOB_FLAGS="-N 1 -n 4"

srun $JOB_FLAGS python -u run_recon.py --tracer BGS_BRIGHT-21.35
srun $JOB_FLAGS python -u run_recon.py --tracer LRG
srun $JOB_FLAGS python -u run_recon.py --tracer ELG_LOPnotqso
srun $JOB_FLAGS python -u run_recon.py --tracer QSO

echo " "
if (( $SECONDS > 3600 )); then
    let "hours=SECONDS/3600"
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)"
elif (( $SECONDS > 60 )); then
    let "minutes=(SECONDS%3600)/60"
    let "seconds=(SECONDS%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $SECONDS seconds"
fi
