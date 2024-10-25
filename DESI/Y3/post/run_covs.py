### Python script for running RascalC in DESI setup (Michael Rashkovetskyi, 2024).
import sys, os
import numpy as np
from astropy.table import Table, vstack
import desi_y3_files.file_manager as desi_y3_file_manager
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
loops_per_sample = 64 # number of loops to collapse into one subsample
N2 = 5 # number of secondary cells/particles per primary cell
N3 = 10 # number of third cells/particles per secondary cell/particle
N4 = 20 # number of fourth cells/particles per third cell/particle

# Settings for filenames
verspec = 'kibo-v1'
version = "v1"
conf = "BAO/blinded"

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only path 

fm = desi_y3_file_manager.get_data_file_manager(conf, verspec)

id = int(sys.argv[1]) # SLURM_JOB_ID to decide what this one has to do
reg = "NGC" if id%2 else "SGC" # region for filenames

id //= 2 # extracted all needed info from parity, move on
tracers = ['LRG'] * 3 + ['ELG_LOPnotqso'] * 2 + ['BGS_BRIGHT-21.5', 'BGS_BRIGHT-21.35', 'BGS_BRIGHT-20.2', 'QSO']
zs = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1), (0.8, 1.1), (1.1, 1.6), (0.1, 0.4), (0.1, 0.4), (0.1, 0.4), (0.8, 2.1)]
# need 2 * 9 = 18 jobs in this array

tlabels = [tracers[id]] # tracer labels for filenames
z_range = tuple(zs[id]) # for redshift cut and filenames
z_min, z_max = z_range
nrandoms = desi_y3_file_manager.list_nran[tlabels[0]]

if nrandoms >= 8: nrandoms //= 2 # to keep closer to the old runtime & convergence level, when LRG and ELG had only 4 randoms
if tlabels[0].startswith("BGS"): nrandoms = 1 # override 1 random catalog for any BGS
if tlabels[0] == 'BGS_BRIGHT-20.2': N3 *= 2; N4 *= 2

# set the number of integration loops based on tracer, z range and region
n_loops = {'LRG': {(0.4, 0.6): {'SGC': 2048,
                                'NGC': 2048},
                   (0.6, 0.8): {'SGC': 2048,
                                'NGC': 1536},
                   (0.8, 1.1): {'SGC': 1024,
                                'NGC': 512}},
           'ELG_LOPnotqso': {(0.8, 1.1): {'SGC': 768,
                                          'NGC': 512},
                             (1.1, 1.6): {'SGC': 512,
                                          'NGC': 384}},
           'BGS_BRIGHT-21.5': {(0.1, 0.4): {'SGC': 3072,
                                            'NGC': 1536}},
           'BGS_BRIGHT-21.35': {(0.1, 0.4): {'SGC': 3072,
                                             'NGC': 1536}},
           'BGS_BRIGHT-20.2': {(0.1, 0.4): {'SGC': 1024,
                                            'NGC': 768}},
           'QSO': {(0.8, 2.1): {'SGC': 256,
                                'NGC': 256}}}[tlabels[0]][z_range][reg]

assert n_loops % nthread == 0, f"Number of integration loops ({n_loops}) must be divisible by the number of threads ({nthread})"
assert n_loops % loops_per_sample == 0, f"Number of integration loops ({n_loops}) must be divisible by the number of loops per sample ({loops_per_sample})"

common_setup = {"region": reg, "version": version}
xi_setup = desi_y3_file_manager.get_baseline_2pt_setup(tlabels[0], z_range, recon = True)
xi_setup.update({"zrange": z_range, "cut": None, "njack": njack}) # specify z_range, no cut and jackknives
recon_setup = desi_y3_file_manager.get_baseline_recon_setup(tlabels[0], z_range)
recon_setup.pop("zrange")

recon_spec = 'recon_sm{smoothing_radius:.0f}_{algorithm}_{mode}'.format_map(xi_setup) # recon specifier string
recon_spec += '' if (zr := xi_setup['recon_zrange']) is None else '_z{zrange[0]:.1f}-{zrange[1]:.1f}'.format(zrange = zr)
recon_spec += '' if (w := xi_setup['recon_weighting']) == 'default' else '_{}'.format(w)

# Output and temporary directories

outdir_base = os.path.join(version, conf, recon_spec, "_".join(tlabels + [reg]) + f"_z{z_min}-{z_max}")
outdir = os.path.join("outdirs", outdir_base) # output file directory
tmpdir = os.path.join("tmpdirs", outdir_base) # directory to write intermediate files, kept in a different subdirectory for easy deletion, almost no need to worry about not overwriting there
preserve(outdir) # rename the directory if it exists to prevent overwriting

# Form correlation function labels
assert len(tlabels) in (1, 2), "Only 1 and 2 tracers are supported"
corlabels = [tlabels[0]]
if len(tlabels) == 2: corlabels += ["_".join(tlabels), tlabels[1]] # cross-correlation comes between the auto-correlatons

# Filenames for saved pycorr counts
pycorr_filenames = [[f.filepath for f in fm.select(id = 'correlation_recon_y1', tracer = corlabel, **common_setup, **xi_setup)] for corlabel in corlabels]
print("pycorr filenames:", pycorr_filenames)

# Filenames for randoms and galaxy catalogs
random_filenames = [[f.filepath for f in fm.select(id = 'catalog_randoms_recon_y1', tracer = tlabel, iran = range(nrandoms), **common_setup, **recon_setup)] for tlabel in tlabels]
print("Random filenames:", random_filenames)
if njack:
    data_ref_filenames = [fm.select(id = 'catalog_data_recon_y1', tracer = tlabel, **common_setup, **recon_setup)[0].filepath for tlabel in tlabels] # only for jackknife reference, could be used for determining the number of galaxies but not in this case
    print("Data filenames:", data_ref_filenames)

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
        subsampler = KMeansSubsampler('angular', positions = [data_catalog["RA"], data_catalog["DEC"], data_catalog["Z"]], position_type = 'rdd', nsamples = njack, nside = 512, random_state = 42)
        randoms_samples[t] = subsampler.label(positions = [random_catalog["RA"], random_catalog["DEC"], random_catalog["Z"]], position_type = 'rdd')
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
