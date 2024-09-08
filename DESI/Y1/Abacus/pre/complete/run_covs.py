### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2024).
import sys, os
import numpy as np
from astropy.table import Table, vstack
import desi_y1_files.file_manager as desi_y1_file_manager
from pycorr import TwoPointCorrelationFunction, KMeansSubsampler
from LSS.tabulated_cosmo import TabulatedDESI
from RascalC.pycorr_utils.utils import fix_bad_bins_pycorr
from RascalC import run_cov

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
    catalog = Table.read(filename)
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
nbin = 50 # number of radial bins for output cov
mbin = None # number of angular (mu) bins to use for projections, None means to keep the original number from pycorr files
skip_nbin_pre = 0 # number of first radial bins to exclude before running the C++ code
skip_nbin_post = 5 # number of first radial bins to exclude at post-processing, in addition to the above
skip_l_post = 0 # number of higher (even) multipoles to exclude at post-processing

# Input correlation function binning
nbin_cf = 100 # number of radial bins for input 2PCF
mbin_cf = 10 # number of angular (mu) bins for input 2PCF

# Settings related to time and convergence

nthread = 64 # number of OMP threads to use
n_loops = 1024 # number of integration loops per filename
loops_per_sample = 64 # number of loops to collapse into one subsample
N2 = 5 # number of secondary cells/particles per primary cell
N3 = 10 # number of third cells/particles per secondary cell/particle
N4 = 20 # number of fourth cells/particles per third cell/particle

# Settings for filenames; many are decided by the first command-line argument
version = "v4_2"
fa = "complete" # fiber assignment method

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only mount works faster, and we don't need to write

fm = desi_y1_file_manager.get_abacus_file_manager()

id = int(sys.argv[1]) # SLURM_JOB_ID to decide what this one has to do
mock_id = 0 # mock number, starting from 0
# not very obvious; made for compatibility with the previous ordering where LRG runs had IDs 0 through 19 with each mock's S/NGC next to each other

reg = "NGC" if id%2 else "SGC" # region for filenames
# known cases where more loops are needed consistently
if id in (4,): n_loops *= 2
elif id in (1, 3, 15): n_loops *= 3
elif id in (2, 8, 14): n_loops *= 4
elif id in (0,): n_loops *= 6
elif id in (17,): n_loops //= 2 # QSO NGC converge well and take rather long time

id //= 2 # extracted all needed info from parity, move on
tracers = ['LRG'] * 4 + ['ELG_LOPnotqso'] * 3 + ['BGS_BRIGHT-21.5', 'QSO']
zs = [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1], [0.4, 1.1], [0.8, 1.1], [1.1, 1.6], [0.8, 1.6], [0.1, 0.4], [0.8, 2.1]]
# need 2 * 9 = 18 jobs in this array

tlabels = [tracers[id]] # tracer labels for filenames
z_range = tuple(zs[id]) # for redshift cut and filenames
z_min, z_max = z_range
nrandoms = desi_y1_file_manager.list_nran[tlabels[0]]

if tlabels[0].startswith("BGS"): version = "v1" # BGS not available beyond v1 (yet)

if nrandoms >= 8: n_loops //= 2 # to keep closer to the old runtime & convergence level, when LRG and ELG had only 4 randoms

all_mocks_setup = {"region": reg, "version": version, "fa": fa}
single_mock_setup = all_mocks_setup | {"imock": mock_id}
xi_setup = desi_y1_file_manager.get_baseline_2pt_setup(tlabels[0], z_range)
xi_setup.update({"zrange": z_range, "cut": None}) # specify z_range and no cut

# Output and temporary directories

outdir_base = os.path.join(version, fa, "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}")
outdir = os.path.join("outdirs", outdir_base) # output file directory
tmpdir = os.path.join("tmpdirs", outdir_base) # directory to write intermediate files, kept in a different subdirectory for easy deletion, almost no need to worry about not overwriting there
preserve(outdir) # rename the directory if it exists to prevent overwriting

# Form correlation function labels
assert len(tlabels) in (1, 2), "Only 1 and 2 tracers are supported"
corlabels = [tlabels[0]]
if len(tlabels) == 2: corlabels += ["_".join(tlabels), tlabels[1]] # cross-correlation comes between the auto-correlatons

# Filenames for saved pycorr counts
input_xi_filenames = [[f.filepath for f in fm.select(id = 'correlation_abacus_y1', tracer = corlabel, njack = 0, **all_mocks_setup, **xi_setup)] for corlabel in corlabels]
print("input xi filenames:", input_xi_filenames)
counts_jack_filenames = [[f.filepath for f in fm.select(id = 'correlation_abacus_y1', tracer = corlabel, njack = 60, **single_mock_setup, **xi_setup)] for corlabel in corlabels]
print("counts/jack filenames:", counts_jack_filenames)

# Filenames for randoms and galaxy catalogs
random_filenames = [[f.filepath for f in fm.select(id = 'catalog_randoms_abacus_y1', tracer = tlabel, iran = range(nrandoms), **single_mock_setup)] for tlabel in tlabels]
print("Random filenames:", random_filenames)
if njack:
    data_ref_filenames = [fm.select(id = 'catalog_data_abacus_y1', tracer = tlabel, **single_mock_setup)[0].filepath for tlabel in tlabels] # only for jackknife reference, could be used for determining the number of galaxies but not in this case
    print("Data filenames:", data_ref_filenames)

# Load pycorr counts
pycorr_allcounts = [0] * len(counts_jack_filenames)
for c, pycorr_filenames_group in enumerate(counts_jack_filenames):
    cumulative_ndata = 0
    for pycorr_filename in pycorr_filenames_group:
        these_counts = fix_bad_bins_pycorr(TwoPointCorrelationFunction.load(pycorr_filename)) # load and attempt to fix faulty bins using symmetry
        # reshape for covariance
        if mbin: assert these_counts.shape[1] % (2 * mbin) == 0, "Angular rebinning is not possible"
        pycorr_allcounts[c] += these_counts[::these_counts.shape[0] // nbin][skip_nbin_pre:, ::these_counts.shape[1] // 2 // mbin if mbin else 1].wrap()
# Load pycorr correlation functions
input_xis = [0] * len(input_xi_filenames)
ndata = [None] * 2
for c, pycorr_filenames_group in enumerate(input_xi_filenames):
    cumulative_ndata = 0
    for pycorr_filename in pycorr_filenames_group:
        these_counts = fix_bad_bins_pycorr(TwoPointCorrelationFunction.load(pycorr_filename)) # load and attempt to fix faulty bins using symmetry
        cumulative_ndata += these_counts.D1D2.size1 # accumulate number of data
        # reshape for input correlation function
        if mbin_cf: assert these_counts.shape[1] % (2 * mbin_cf) == 0, "Angular rebinning is not possible"
        input_xis[c] += these_counts[::these_counts.shape[0] // nbin_cf, ::these_counts.shape[1] // 2 // mbin_cf if mbin_cf else 1].wrap()
    if c % 2 == 0: ndata[c // 2] = cumulative_ndata / len(pycorr_filenames_group) # set the average number of data based on auto-correlatons
# add None's for missing counts/correlations
ncorr_max = 3 # maximum number of correlations
pycorr_allcounts += [None] * (ncorr_max - len(counts_jack_filenames))
input_xis += [None] * (ncorr_max - len(input_xi_filenames))

# Load randoms and galaxy catalogs
cosmology = TabulatedDESI() # for conversion from RA,DEC,Z to Cartesian
ntracers_max = 2 # maximum number of tracers
randoms_positions = [None] * ntracers_max
randoms_weights = [None] * ntracers_max
randoms_samples = [None] * ntracers_max
for t in range(len(tlabels)):
    # read randoms with redshift cut
    random_catalog = vstack([read_catalog(random_filename, z_min = z_min, z_max = z_max) for random_filename in random_filenames[t]])
    randoms_weights[t] = random_catalog["WEIGHT"]
    # create jackknives
    if njack:
        data_catalog = read_catalog(data_ref_filenames[t], z_min = z_min, z_max = z_max)
        subsampler = KMeansSubsampler('angular', positions = np.array([data_catalog["RA"], data_catalog["DEC"], data_catalog["Z"]], dtype = np.float64), position_type = 'rdd', nsamples = njack, nside = 512, random_state = 42)
        randoms_samples[t] = subsampler.label(positions = np.array([random_catalog["RA"], random_catalog["DEC"], random_catalog["Z"]], dtype = np.float64), position_type = 'rdd')
    # compute comoving distance
    randoms_positions[t] = [random_catalog["RA"], random_catalog["DEC"], cosmology.comoving_radial_distance(random_catalog["Z"])]

# Run the main code, post-processing and extra convergence check
results = run_cov(mode = mode, max_l = max_l, boxsize = periodic_boxsize,
                  nthread = nthread, N2 = N2, N3 = N3, N4 = N4, n_loops = n_loops, loops_per_sample = loops_per_sample,
                  pycorr_allcounts_11 = pycorr_allcounts[0], pycorr_allcounts_12 = pycorr_allcounts[1], pycorr_allcounts_22 = pycorr_allcounts[2],
                  xi_table_11 = input_xis[0], xi_table_12 = input_xis[1], xi_table_22 = input_xis[2],
                  no_data_galaxies1 = ndata[0], no_data_galaxies2 = ndata[1],
                  position_type = "rdd",
                  randoms_positions1 = randoms_positions[0], randoms_weights1 = randoms_weights[0], randoms_samples1 = randoms_samples[0],
                  randoms_positions2 = randoms_positions[1], randoms_weights2 = randoms_weights[1], randoms_samples2 = randoms_samples[1],
                  normalize_wcounts = True,
                  out_dir = outdir, tmp_dir = tmpdir,
                  skip_s_bins = skip_nbin_post, skip_l = skip_l_post)
