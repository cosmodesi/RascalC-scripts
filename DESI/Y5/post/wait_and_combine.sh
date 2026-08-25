#!/bin/bash
# Poll job 55803114 (task _1) until it leaves the queue, then run the combine + post-process script.
JOBID=55803390_1
cd /global/homes/q/qinxunli/dev/RascalC-scripts/DESI/Y5/post

while squeue -j "$JOBID" -h 2>/dev/null | grep -q .; do
    sleep 60
done

echo "Job $JOBID finished at $(date). Final state:"
sacct -j "$JOBID" --format=JobID,State,ExitCode,Elapsed --noheader

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -u combine_and_postprocess.py
