### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2025-2026).
import sys, os
import numpy as np
from astropy.table import vstack
import lsstypes
from clustering_statistics.tools import get_stats_fn, get_catalog_fn
from pycorr import KMeansSubsampler
from LSS.common_tools import read_hdf5_blosc
from LSS.tabulated_cosmo import TabulatedDESI
from RascalC.lsstypes_utils.utils import reshape_lsstypes
from RascalC import run_cov
import argparse

parser = argparse.ArgumentParser(description = "Main RascalC computation script for DESI Y3 GLAM mocks pre-recon single-tracer")
parser.add_argument("id", type = int, help = "number of the task in the array, encoding tracer, redshift bin and region (SGC/NGC)")
parser.add_argument("-t", "--test", action = "store_true", help = "test the input files, abort before the main computation")
args = parser.parse_args()

def preserve(filename: str, max_num: int = 10) -> None: # if the file/directory exists, rename it with a numeric suffix
    if not os.path.exists(filename): return
    for i in range(max_num+1):
        trial_name = filename + ("_" + str(i))
        if not os.path.exists(trial_name):
            os.rename(filename, trial_name)
            print(f"Found existing {filename}, renamed into {trial_name}.")
            return
    raise RuntimeError(f"Could not back up {filename}, aborting.")

def read_catalog(filename: str, z_min: float = -np.inf, z_max: float = np.inf, FKP_weight: bool = True):
    catalog = read_hdf5_blosc(filename)
    if FKP_weight: catalog["WEIGHT"] *= catalog["WEIGHT_FKP"] # apply FKP weight multiplicatively
    catalog.keep_columns(["RA", "DEC", "Z", "WEIGHT"]) # discard everything else
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range
    return catalog[filtering] # filtered catalog

# Mode settings

mode = "legendre_projected"
max_l = 4 # maximum (even) multipole index

njack = 60 # if 0 jackknife is turned off

periodic_boxsize = None # aperiodic if None (or 0)

# Covariance matrix binning
# nbin = 50 # number of radial bins for output cov
r_step = 4 # step in radial bins for output cov
mbin = None # number of angular (mu) bins to use for projections, None means to keep the original number from counts files
skip_nbin_pre = 0 # number of first radial bins to exclude before running the C++ code
skip_nbin_post = 5 # number of first radial bins to exclude at post-processing, in addition to the above
skip_l_post = 0 # number of higher (even) multipoles to exclude at post-processing

# Input correlation function binning
# nbin_cf = 100 # number of radial bins for input 2PCF
r_step_cf = 2 # step in radial bins for input 2PCF
mbin_cf = 10 # number of angular (mu) bins for input 2PCF

# Settings related to time and convergence

nthread = 64 # number of OMP threads to use
loops_per_sample = 64 # number of loops to collapse into one subsample
N2 = 5 # number of secondary cells/particles per primary cell
N3 = 10 # number of third cells/particles per secondary cell/particle
N4 = 20 # number of fourth cells/particles per third cell/particle

# Settings for filenames
version = 'glam-uchuu-v2-altmtl'

id = args.id # SLURM_JOB_ID to decide what this one has to do
reg = "NGC" if id%2 else "SGC" # region for filenames

id //= 2 # extracted all needed info from parity, move on
tracers = ['LRG'] * 3 + ['ELG_LOPnotqso'] * 2 + ['LRG+ELG_LOPnotqso', 'BGS_BRIGHT-21.5'] + ['BGS_BRIGHT-21.35'] * 2 + ['BGS_BRIGHT-20.2'] * 2 + ['QSO']
zs = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1), (0.8, 1.1), (1.1, 1.6), (0.8, 1.1), (0.1, 0.4), (0.1, 0.4), (0.25, 0.4), (0.1, 0.25), (0.1, 0.4), (0.8, 2.1)]
# need 2 * 12 = 24 jobs in this array

tlabels = [tracers[id]] # tracer labels for filenames
z_range = tuple(zs[id]) # for redshift cut and filenames
z_min, z_max = z_range
nrandoms = {'LRG': 4, 'ELG_LOPnotqso': 5, 'LRG+ELG_LOPnotqso': 5, 'BGS_BRIGHT-21.35': 1, 'QSO': 4}[tlabels[0]]

if tlabels[0] == 'BGS_BRIGHT-20.2': N3 *= 2; N4 *= 4 # due to convergence issues

# set the number of integration loops based on tracer, z range and region
n_loops = {'LRG': {(0.4, 0.6): {'SGC': 1536,
                                'NGC': 1536},
                   (0.6, 0.8): {'SGC': 1536,
                                'NGC': 1024},
                   (0.8, 1.1): {'SGC': 1024,
                                'NGC': 768}},
           'ELG_LOPnotqso': {(0.8, 1.1): {'SGC': 768,
                                          'NGC': 512},
                             (1.1, 1.6): {'SGC': 512,
                                          'NGC': 384}},
           'LRG+ELG_LOPnotqso': {(0.8, 1.1): {'SGC': 768,
                                              'NGC': 512}},
           'BGS_BRIGHT-21.5': {(0.1, 0.4): {'SGC': 2048,
                                            'NGC': 1024}},
           'BGS_BRIGHT-21.35': {(0.1, 0.4): {'SGC': 3072,
                                             'NGC': 1024},
                                (0.25, 0.4): {'SGC': 4096,
                                              'NGC': 1536}},
           'BGS_BRIGHT-20.2': {(0.1, 0.25): {'SGC': 4096,
                                             'NGC': 2048},
                               (0.1, 0.4): {'SGC': 1024,
                                            'NGC': 512}},
           'QSO': {(0.8, 2.1): {'SGC': 256,
                                'NGC': 256}}}[tlabels[0]][z_range][reg]

assert n_loops % nthread == 0, f"Number of integration loops ({n_loops}) must be divisible by the number of threads ({nthread})"
assert n_loops % loops_per_sample == 0, f"Number of integration loops ({n_loops}) must be divisible by the number of loops per sample ({loops_per_sample})"

# Output and temporary directories

mock_id = 150

outdir_base = os.path.join(version, f"mock{mock_id}", "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}")
outdir = os.path.join("outdirs", outdir_base) # output file directory
tmpdir = os.path.join("tmpdirs", outdir_base) # directory to write intermediate files, kept in a different subdirectory for easy deletion, almost no need to worry about not overwriting there

# Form correlation function labels
assert len(tlabels) in (1, 2), "Only 1 and 2 tracers are supported"
corlabels = [tlabels[0]]
if len(tlabels) == 2: corlabels += ["_".join(tlabels), tlabels[1]] # cross-correlation comes between the auto-correlatons

# Filenames for saved counts
allcounts_filenames = [[get_stats_fn(version=version, imock=mock_id, tracer=corlabel, region=reg, zrange=z_range, stats_dir='/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics', project='full_shape/base', kind='particle2_correlation', jackknife=dict(nsplits=njack))] for corlabel in corlabels]
# will probably need to adjust the number of randoms when run beyond LRG
print("allcounts filenames:", allcounts_filenames)

# Filenames for randoms and galaxy catalogs
random_filenames = [get_catalog_fn(version=version, imock=mock_id, tracer=tlabel, region=reg, kind='randoms', nran=nrandoms) for tlabel in tlabels]
print("Random filenames:", random_filenames)
data_ref_filenames = [get_catalog_fn(version=version, imock=mock_id, tracer=tlabel, region=reg, kind='data') for tlabel in tlabels] # for jackknife reference and the number of galaxies
print("Data filenames:", data_ref_filenames)

# Load counts and correlations
ncorr_max = 3 # maximum number of correlations
allcounts = [None] * ncorr_max
input_xis = [None] * ncorr_max
for c, allcounts_filename in enumerate(allcounts_filenames):
    these_counts = lsstypes.read(allcounts_filename)
    allcounts[c] = reshape_lsstypes(these_counts, r_step=r_step, n_mu=mbin, skip_r_bins=skip_nbin_pre) # reshape for covariance
    input_xis[c] = reshape_lsstypes(these_counts, r_step=r_step_cf, n_mu=mbin_cf) # reshape for input correlation function
del these_counts # free up memory

# Load randoms and galaxy catalogs
cosmology = TabulatedDESI() # for conversion from RA,DEC,Z to Cartesian
ntracers_max = 2 # maximum number of tracers
randoms_positions = [None] * ntracers_max
randoms_weights = [None] * ntracers_max
randoms_samples = [None] * ntracers_max
ndata = [None] * 2
for t in range(len(tlabels)):
    # read randoms with redshift cut
    random_catalog = vstack([read_catalog(random_filename, z_min=z_min, z_max=z_max) for random_filename in random_filenames[t]])
    randoms_weights[t] = random_catalog["WEIGHT"]
    data_catalog = read_catalog(data_ref_filenames[t], z_min=z_min, z_max=z_max)
    ndata[t] = np.sum(data_catalog["WEIGHT"])**2 / np.sum(data_catalog["WEIGHT"]**2) # probably better than just len(data_catalog) because insensitive to zero-weight objects and less sensitive to low-weight objects
    # create jackknives
    if njack:
        subsampler = KMeansSubsampler('angular', positions = [data_catalog["RA"], data_catalog["DEC"], data_catalog["Z"]], position_type = 'rdd', dtype='f8', nsamples = njack, nside = 512, random_state = 42)
        randoms_samples[t] = subsampler.label(positions = [random_catalog["RA"], random_catalog["DEC"], random_catalog["Z"]], position_type = 'rdd')
    # compute comoving distance
    randoms_positions[t] = [random_catalog["RA"], random_catalog["DEC"], cosmology.comoving_radial_distance(random_catalog["Z"])]
del random_catalog, data_catalog # free up memory

if args.test: sys.exit(0) # exit with an ok status in a test run
preserve(outdir) # rename the directory if it exists to prevent overwriting, but avoid doing this for a test run and in cases when the script fails at an earlier stage

# Run the main code, post-processing and extra convergence check
results = run_cov(mode = mode, max_l = max_l, boxsize = periodic_boxsize,
                  nthread = nthread, N2 = N2, N3 = N3, N4 = N4, n_loops = n_loops, loops_per_sample = loops_per_sample,
                  allcounts_11 = allcounts[0], allcounts_12 = allcounts[1], allcounts_22 = allcounts[2],
                  xi_table_11 = input_xis[0], xi_table_12 = input_xis[1], xi_table_22 = input_xis[2],
                  no_data_galaxies1 = ndata[0], no_data_galaxies2 = ndata[1],
                  position_type = "rdd",
                  randoms_positions1 = randoms_positions[0], randoms_weights1 = randoms_weights[0], randoms_samples1 = randoms_samples[0],
                  randoms_positions2 = randoms_positions[1], randoms_weights2 = randoms_weights[1], randoms_samples2 = randoms_samples[1],
                  normalize_wcounts = True,
                  out_dir = outdir, tmp_dir = tmpdir,
                  skip_s_bins = skip_nbin_post, skip_l = skip_l_post)
