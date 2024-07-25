# interactive launch, NOT a batch script
srun --account=desi_g --constraint=gpu --qos=interactive --time=4:00:00 --nodes=1 --gpus=4 --job-name=allcounts-EZmock-altmtl1-jack python -u run_xi_jack.py