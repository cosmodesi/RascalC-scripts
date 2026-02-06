#!/bin/bash

# load cosmodesi environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

LSSCODE=$HOME
BASEDIR=/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/PIP/
BASE_OUTDIR=loa-v1/LSScats/v2/PIP/

srun -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python $LSSCODE/LSS/scripts/xirunpc.py --gpu --tracer BGS_ANY-21.35 --region NGC SGC --corr_type smu --njack 60 --weight_type default_FKP --basedir $BASEDIR --outdir $BASE_OUTDIR/xi