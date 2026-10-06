#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH --job-name=RascalC-Y3-HOLI-recon

set -e
SECONDS=0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# module unload desi-clustering # use locally installed desi-clustering if uncommented, otherwise use the global one from cosmodesi environment

JOB_FLAGS="-N 1 -n 4"

for MOCK_ID in {0..4} ; do
    for TRACER in BGS_BRIGHT-21.35 LRG ELG_LOPnotqso LRG+ELG_LOPnotqso QSO ; do
        srun $JOB_FLAGS python -u run_recon.py --tracer $TRACER --mock_id $MOCK_ID
    done
done

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
