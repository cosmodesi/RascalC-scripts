#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-highzQSO
#SBATCH --array=0,1 # SGC (0) and NGC (1)

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH="${PYTHONPATH}:$HOME/Repos/desi-y3-kp/"
export PYTHONPATH="${PYTHONPATH}:$HOME/Repos/LSS/py/"
# Point to DESI catalog root; override if using a local copy (e.g. pscratch test area)
export DESICFS="${DESICFS:-/dvs_ro/cfs/cdirs/desi}"

python -u run_covs.py $SLURM_ARRAY_TASK_ID
