#!/bin/bash

# load cosmodesi environment, test version
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh test

LSSCODE=$HOME

srun -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g python $LSSCODE/LSS/scripts/xirunpc.py --gpu --tracer LRG --survey DA2 --verspec loa-v1 --version test --region NGC SGC --corr_type smu --njack 60 --weight_type default_FKP --basedir /dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/mocks/Glam-Uchuu_Y3/altmtl11/loa-v1/mock11/LSScats/ --outdir xi