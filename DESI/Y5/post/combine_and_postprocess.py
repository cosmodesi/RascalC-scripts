### Combine raw covariance matrices from two independent RascalC runs (same tracer/region/settings)
### and post-process the combined set. Written to salvage BGS_BRIGHT-21.35 NGC z0.1-0.4 (task _1 of
### array 55740796), which failed post-processing convergence with only 1536 loops - combining with
### a second independent 1536-loop run effectively doubles the statistics to 3072 loops.
import os
from RascalC.raw_covariance_matrices import cat_raw_covariance_matrices
from RascalC.post_process.legendre_mix_jackknife import post_process_legendre_mix_jackknife

basedir = os.path.join(os.environ['PSCRATCH'], 'dr3', 'rascalc')
outdir_base = os.path.join('data-dr3-matterhorn-v2-v0-bao', 'BGS_BRIGHT-21.35_NGC_z0.1-0.4')

run_dirs = [
    os.path.join(basedir, 'outdirs', outdir_base + '_1'),  # original failed run (05:51, job 55740796_1)
    os.path.join(basedir, 'outdirs', outdir_base),          # new run (job 55803114_1)
]
combined_dir = os.path.join(basedir, 'outdirs', outdir_base + '_combined')
os.makedirs(combined_dir, exist_ok=True)

n = 45       # radial bins
max_l = 4    # max multipole
m = 100      # mu bins used for jackknife weights
njack = 60
skip_r_bins_post = 5
skip_l_post = 0

print(f"Combining raw covariance matrices from: {run_dirs}")
cat_raw_covariance_matrices(
    n=n, mstr=f"l{max_l}",
    input_roots=run_dirs, ns_samples=[None, None],
    output_root=combined_dir,
)

jackknife_file = os.path.join(combined_dir, 'xi_jack', f'xi_jack_n{n}_m{m}_j{njack}_11.dat')
weight_dir = os.path.join(combined_dir, 'weights')

results = post_process_legendre_mix_jackknife(
    jackknife_file, weight_dir, combined_dir, m, max_l, combined_dir,
    skip_r_bins=skip_r_bins_post, skip_l=skip_l_post,
)
print("Done. Results written to", combined_dir)
