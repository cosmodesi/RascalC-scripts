# interactive launch, NOT a batch script
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun --account=desi_g --constraint=gpu --qos=interactive --time=4:00:00 --nodes=1 --gpus=4 --job-name=allcounts-Abacus-v4_2-ffa-jack python -u run_xi_jack.py