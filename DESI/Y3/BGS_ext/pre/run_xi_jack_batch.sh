#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --time=4:00:00
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --account=desi_g
#SBATCH --job-name=Y3-BGS-ext-xi-jack
#SBATCH --array=0-1

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

CONF=PIP
if (( $SLURM_ARRAY_TASK_ID == 1 )); then
    CONF=nonKP
fi

LSSCODE=$HOME
BASEDIR=/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/$CONF/
BASE_OUTDIR=loa-v1/LSScats/v2/$CONF/

python $LSSCODE/LSS/scripts/xirunpc.py --gpu --tracer BGS_BRIGHT-21.35 --nran 1 --region NGC SGC --corr_type smu --njack 60 --weight_type default_FKP --basedir $BASEDIR --outdir $BASE_OUTDIR/xi