#!/bin/bash

# load cosmodesi environment, 2025_05 (old test) version
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_05

LSSCODE=$HOME
BASEDIR=/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/holi_v1/altmtl451/loa-v1/mock451/LSScats/
BASE_OUTDIR=altmtl451

srun -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python $LSSCODE/LSS/scripts/xirunpc.py --gpu --tracer LRG --survey DA2 --verspec loa-v1 --version test --region NGC SGC --corr_type smu --njack 60 --weight_type default_FKP --basedir $BASEDIR --outdir $BASE_OUTDIR/xi
srun -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python $LSSCODE/LSS/scripts/xirunpc.py --gpu --tracer ELG_LOPnotqso --survey DA2 --verspec loa-v1 --version test --region NGC SGC --corr_type smu --njack 60 --weight_type default_FKP --basedir $BASEDIR --outdir $BASE_OUTDIR/xi
srun -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python $LSSCODE/LSS/scripts/xirunpc.py --gpu --tracer QSO --survey DA2 --verspec loa-v1 --version test --region NGC SGC --corr_type smu --njack 60 --weight_type default_FKP --basedir $BASEDIR --outdir $BASE_OUTDIR/xi