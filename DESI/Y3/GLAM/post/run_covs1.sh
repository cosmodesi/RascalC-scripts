#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-Y3-GLAM-post
##SBATCH --array=0,1,4,5,10-15 # BGS_BRIGHT-21.35 0.1-0.4, LRG2, ELG2, LRG3+ELG1, QSO 0.8-2.1
#SBATCH --array=0 # BGS_BRIGHT-21.35 0.1-0.4 SGC

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# module unload desi-clustering # use locally installed desi-clustering if uncommented, otherwise use the global one from cosmodesi environment

python -u run_covs.py $SLURM_ARRAY_TASK_ID --mock_id 151
