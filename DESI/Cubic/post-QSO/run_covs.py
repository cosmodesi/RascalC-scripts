### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2024).
import sys, os
import numpy as np
from astropy.table import Table, vstack
from pycorr import TwoPointCorrelationFunction
from RascalC.pycorr_utils.utils import fix_bad_bins_pycorr
from RascalC import run_cov

def prevent_override(filename: str, max_num: int = 10) -> str: # append _{number} to filename to prevent override
    for i in range(max_num+1):
        trial_name = filename + ("_" + str(i)) * bool(i) # will be filename for i=0
        if not os.path.exists(trial_name): return trial_name
    print(f"Could not prevent override of {filename}, aborting.")
    sys.exit(1)

# Mode settings

mode = "legendre_accumulated"
max_l = 4 # maximum (even) multipole index

periodic_boxsize = 2000 # aperiodic if None (or 0)

# Covariance matrix binning
nbin = 50 # number of radial bins for output cov
mbin = None # number of angular (mu) bins to use for projections, None means to keep the original number from pycorr files
skip_nbin_pre = 0 # number of first radial bins to exclude before running the C++ code
skip_nbin_post = 5 # number of first radial bins to exclude at post-processing, in addition to the above
skip_l_post = 0 # number of higher (even) multipoles to exclude at post-processing

# Input correlation function binning
nbin_cf = 100 # number of radial bins for input 2PCF
mbin_cf = 10 # number of angular (mu) bins for input 2PCF

# Settings related to time and convergence

nthread = 256 # number of OMP threads to use
n_loops = 1024 # number of integration loops per filename
loops_per_sample = 64 # number of loops to collapse into one subsample
N2 = 5 # number of secondary cells/particles per primary cell
N3 = 10 # number of third cells/particles per secondary cell/particle
N4 = 20 # number of fourth cells/particles per third cell/particle

# Settings for filenames; many are decided by the first command-line argument
tlabels = ["QSO"] # tracer labels for filenames
hod_label = sys.argv[1] # HOD label (number)
z_ref = 1.4 # reference redshift

# Output and temporary directories

outdir_base = "_".join(tlabels) + f"-HOD{hod_label}"
outdir = prevent_override(os.path.join("outdirs", outdir_base)) # output file directory
tmpdir = os.path.join("tmpdirs", outdir_base) # directory to write intermediate files, kept in a different subdirectory for easy deletion, almost no need to worry about not overwriting there

# Form correlation function labels
assert len(tlabels) in (1, 2), "Only 1 and 2 tracers are supported"
corlabels = [tlabels[0]]
if len(tlabels) == 2: corlabels += ["_".join(tlabels), tlabels[1]] # cross-correlation comes between the auto-correlatons

# Filenames for saved pycorr counts
pycorr_filenames = [[f"/global/cfs/cdirs/desi/cosmosim/AbacusHOD_mocks/varietyHOD/CubicBox/{corlabel}/z{z_ref:.3f}/jmena/Xi/Post/pycorr_format/Xi_QSO_hod{hod_label}_ph{i:03d}.gcat_shift_MultiGrid_mesh512_smooth30_recsym_f0.920_b2.07.npy" for i in range(25)] for corlabel in corlabels]

# Filenames for randoms
random_filenames = [[f"/global/cfs/cdirs/desi/cosmosim/AbacusHOD_mocks/varietyHOD/CubicBox/{tlabel}/z{z_ref:.3f}/jmena/recon_catalogs/AbacusSummit_base_c000_ph000/QSO_hod{hod_label}_randoms_20x_ph000_shift_MultiGrid_mesh512_smooth30_recsym_f0.920_b2.07.fits"] for tlabel in tlabels] # shifted randoms from one realization – can also use uniform (not shifted) ones but that is less accurate

# Load pycorr counts
pycorr_allcounts = [0] * len(pycorr_filenames)
input_xis = [0] * len(pycorr_filenames)
ndata = [None] * 2
for c, pycorr_filenames_group in enumerate(pycorr_filenames):
    cumulative_ndata = 0
    for pycorr_filename in pycorr_filenames_group:
        these_counts = fix_bad_bins_pycorr(TwoPointCorrelationFunction.load(pycorr_filename)) # load and attempt to fix faulty bins using symmetry
        cumulative_ndata += these_counts.D1D2.size1 # accumulate number of data
        # reshape for covariance
        if mbin: assert these_counts.shape[1] % (2 * mbin) == 0, "Angular rebinning is not possible"
        pycorr_allcounts[c] += these_counts[::these_counts.shape[0] // nbin][skip_nbin_pre:, ::these_counts.shape[1] // 2 // mbin if mbin else 1].wrap()
        # reshape for input correlation function
        if mbin_cf: assert these_counts.shape[1] % (2 * mbin_cf) == 0, "Angular rebinning is not possible"
        input_xis[c] += these_counts[::these_counts.shape[0] // nbin_cf, ::these_counts.shape[1] // 2 // mbin_cf if mbin_cf else 1].wrap()
    if c % 2 == 0: ndata[c // 2] = cumulative_ndata / len(pycorr_filenames_group) # set the average number of data based on auto-correlatons
# add None's for missing counts
ncorr_max = 3 # maximum number of correlations
pycorr_allcounts += [None] * (ncorr_max - len(pycorr_filenames))
input_xis += [None] * (ncorr_max - len(pycorr_filenames))

# Load random catalogs
ntracers_max = 2 # maximum number of tracers
randoms_positions = [None] * ntracers_max
randoms_weights = [None] * ntracers_max
for t in range(len(tlabels)):
    # read randoms
    random_catalog = vstack([Table.read(random_filename) for random_filename in random_filenames[t]])
    # unit weights - make sure it is consistent with the counts!
    randoms_weights[t] = np.ones(len(random_catalog))
    # format randoms positions
    randoms_positions[t] = np.column_stack([random_catalog["x"], random_catalog["y"], random_catalog["z"]])

# Run the main code, post-processing and extra convergence check
results = run_cov(mode = mode, max_l = max_l, boxsize = periodic_boxsize,
                  nthread = nthread, N2 = N2, N3 = N3, N4 = N4, n_loops = n_loops, loops_per_sample = loops_per_sample,
                  pycorr_allcounts_11 = pycorr_allcounts[0], pycorr_allcounts_12 = pycorr_allcounts[1], pycorr_allcounts_22 = pycorr_allcounts[2],
                  xi_table_11 = input_xis[0], xi_table_12 = input_xis[1], xi_table_22 = input_xis[2],
                  no_data_galaxies1 = ndata[0], no_data_galaxies2 = ndata[1],
                  randoms_positions1 = randoms_positions[0], randoms_weights1 = randoms_weights[0],
                  randoms_positions2 = randoms_positions[1], randoms_weights2 = randoms_weights[1],
                  normalize_wcounts = True,
                  out_dir = outdir, tmp_dir = tmpdir,
                  skip_s_bins = skip_nbin_post, skip_l = skip_l_post)
