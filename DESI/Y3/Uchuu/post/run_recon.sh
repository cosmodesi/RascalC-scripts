#!/bin/bash
#SBATCH --account desi_g
#SBATCH -C gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus 4
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH --job-name=RascalC-Y3-Uchuu-recon

set -e
SECONDS=0

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# module unload desi-clustering # use locally installed desi-clustering if uncommented, otherwise use the global one from cosmodesi environment

JOB_FLAGS="-N 1 -n 4"

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
