#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # 128 hyperthreads = 64 physical cores
#SBATCH --job-name=RascalC-LSS-BGS-nonkp_meeting_20260521_bright_faint
# Choose/edit job array to match the campaign; double-check with `python run_covs.py --list-cases --campaign <name>`
##SBATCH --array=0-17 # nonkp_bright_faint_priority
##SBATCH --array=0-13 # nonkp_bright_compare or legacy
#SBATCH --array=0-5 # nonkp_meeting_20260521_bright_faint or nonkp_meeting_20260521_bright20p7
##SBATCH --array=0-29 # pip_priority
##SBATCH --array=0-19 # pip_20p7

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

python -u run_covs.py $SLURM_ARRAY_TASK_ID --campaign nonkp_meeting_20260521_bright_faint
