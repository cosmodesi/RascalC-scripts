# This is the custom script to compute the 2PCF with jackknives, using the pre-saved separation of the combined tracer into the original tracers (from label_tracers.py)

import os
import numpy as np
import numpy.typing as npt
import logging
from astropy.table import Table, vstack
import desi_y3_files.file_manager as desi_y3_file_manager
from RascalC.pre_process import get_subsampler_xirunpc
from RascalC.utils import tracer1_corr, tracer2_corr
from pycorr import TwoPointCorrelationFunction, setup_logging, KMeansSubsampler
from LSS.tabulated_cosmo import TabulatedDESI


def get_rdd_positions(catalog: Table | None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] | None: # utility function to format positions from a catalog, handling the None case intended for the second tracer in auto-correlations
    if catalog is None: return None
    return (catalog["RA"], catalog["DEC"], catalog["comov_dist"])


def get_weights(catalog: Table | None) -> npt.NDArray[np.float64] | None: # utility function to format weights from a catalog, handling the None case intended for the second tracer in auto-correlations
    if catalog is None: return None
    return catalog["WEIGHT"]


def get_samples(catalog: Table | None) -> npt.NDArray[np.float64] | None: # utility function to format samples from a catalog, handling the None case intended for the second tracer in auto-correlations
    if catalog is None: return None
    return catalog["JACK"]


def prepare_catalog(filename: str, z_min: float = -np.inf, z_max: float = np.inf, jack_sampler: KMeansSubsampler | None = None, FKP_weight: bool = True, distinct_tracer_id: int = 1) -> Table:
    catalog: Table = Table.read(filename)
    if FKP_weight: catalog["WEIGHT"] *= catalog["WEIGHT_FKP"] # apply FKP weight multiplicatively
    catalog.keep_columns(["RA", "DEC", "Z", "WEIGHT"]) # discard everything else, including TARGETID (which is no longer needed)
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range
    catalog = catalog[filtering] # filtered catalog
    for key in catalog.keys():
        if catalog[key].dtype != float: # ensure all columns are float(64) for pycorr
            catalog[key] = catalog[key].astype(float)
    catalog["TRACERID"] = (np.load(os.path.basename(filename).replace(".fits", ".npz"))["TRACERID"][filtering] == distinct_tracer_id) # load and add TRACERID to keep track of which separate tracer each object belongs to. Convert to binary classification for the distinct tracer vs the rest
    catalog["comov_dist"] = cosmology.comoving_radial_distance(catalog["Z"])
    if jack_sampler: catalog["JACK"] = jack_sampler.label(get_rdd_positions(catalog), position_type = 'rdd')
    return catalog


setup_logging()
my_logger = logging.getLogger('run_xi_jack')
cosmology = TabulatedDESI()

# Settings for filenames
verspec = 'loa-v1'
version = "v1.1"
conf = "BAO/unblinded"
conf_alt = "unblinded" # for files under Nick's directory

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only mount works faster, and we don't need to write

fm = desi_y3_file_manager.get_data_file_manager(conf, verspec)

output_dir = "xi/smu" # here in scratch for faster write, to be moved later
os.makedirs(output_dir, exist_ok = True)

n_mu_bins = 200 # between -1 and 1
s_max = 200
split_above = 20
# list of edge configurations, so that non-split randoms are below split_above and above they are split
all_edges = [(s_edges, np.linspace(-1, 1, n_mu_bins+1)) for s_edges in (np.arange(split_above+1), np.arange(split_above, s_max+1))]

n_jack = 60

all_separate_tracers = [['LRG', 'ELG_LOPnotqso'], ['LRG+ELG_LOPnotqso', 'QSO']] # tracers to split the combined tracers into
distinct_tracer_ids = [1, 2] # TRACERID of the tracer to be treated differently from others (1 is trivial among 0, 1, but when we have 3, we need to make 2 of them for RascalC. 2 corresponds to QSO for the full combined tracer)
combined_tracers = ['LRG+ELG_LOPnotqso', 'FullCombined'] # the combined tracers
all_z_ranges = [((0.8, 1.1),)] * 2

for tracer, separate_tracers, z_ranges, distinct_tracer_id in zip(combined_tracers, all_separate_tracers, all_z_ranges, distinct_tracer_ids):
    corr_labels = [separate_tracers[0], "_".join(separate_tracers), separate_tracers[1]]
    n_randoms = 5 if tracer == 'FullCombined' else desi_y3_file_manager.list_nran[tracer]
    my_logger.info(f"Tracer: {tracer}")

    for reg in ("SGC", "NGC"):
        my_logger.info(f"Region: {reg}")
        
        for z_range in z_ranges:
            z_min, z_max = z_range
            my_logger.info(f"Redshift range: {z_min}-{z_max}")

            # skip catalog preparation if all output files exist
            output_files = [f"{output_dir}/allcounts_{corr_label}_{reg}_z{z_min}-{z_max}_default_FKP_lin_nran{n_randoms}_njack{n_jack}_split{split_above}.npy" for corr_label in corr_labels]
            if all(os.path.exists(output_file) for output_file in output_files):
                my_logger.info(f"All output files {output_files} exist, skipping")
                continue

            if tracer == 'FullCombined': # generate filenames procedurally
                data_dir = f"/dvs_ro/cfs/cdirs/desi/users/sandersn/DA2/{verspec}/{version}/{conf_alt}/full"
                galaxy_files = [f"{data_dir}/{tracer}_{reg}_clustering.dat.fits"]
            else: # get filenames from the file manager
                common_setup = {"tracer": tracer, "region": reg, "version": version, "grid_cosmo": None}
                galaxy_files = [f.filepath for f in fm.select(id='catalog_data_y3', **common_setup)]
                if (n := len(galaxy_files)) != 1:
                    my_logger.warning(f"Found not 1 but {n} galaxy files; skipping")
                    continue
            try: galaxies = prepare_catalog(galaxy_files[0], z_min, z_max, distinct_tracer_id=distinct_tracer_id)
            except Exception as e:
                my_logger.warning(f"Failed to prepare galaxy catalog: {e}. Skipping")
                continue
            jack_sampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack)
            galaxies["JACK"] = jack_sampler.label(get_rdd_positions(galaxies), position_type = 'rdd')

            if tracer == 'FullCombined': # generate filenames procedurally
                random_files = [f"{data_dir}/{tracer}_{reg}_{iran}_clustering.ran.fits" for iran in range(n_randoms)]
            else: # get filenames from the file manager
                random_files = [f.filepath for f in fm.select(id='catalog_randoms_y3', iran=range(n_randoms), **common_setup)]
                if (n := len(random_files)) != n_randoms:
                    my_logger.warning(f"Found not {n_randoms} but {n} random files; skipping")
                    continue
            try: all_randoms = [prepare_catalog(random_file, z_min, z_max, jack_sampler, distinct_tracer_id=distinct_tracer_id) for random_file in random_files]
            except Exception as e:
                my_logger.warning(f"Failed to prepare random catalogs: {e}. Skipping")
                continue

            for t1, t2, corr_label, output_file in zip(tracer1_corr, tracer2_corr, corr_labels, output_files):
                my_logger.info(f"Computing {corr_label} {'auto' if t1 == t2 else 'cross'}-correlation")

                if os.path.exists(output_file):
                    my_logger.info(f"Output file {output_file} exists, skipping")
                    continue

                galaxies1 = galaxies[galaxies["TRACERID"] == t1]
                galaxies2 = None if t1 == t2 else galaxies[galaxies["TRACERID"] == t2] # for auto-correlation, set the second tracer to None for proper handling in the TwoPointCorrelationFunction call. helper functions will propagate None into the positions, weights and samples

                results = []
                # compute
                for i_split_randoms, edges in enumerate(all_edges):
                    my_logger.info("Using " + ("split" if i_split_randoms else "concatenated") + " randoms")
                    result = 0
                    D1D2 = None
                    for i_random in range(n_randoms if i_split_randoms else 1):
                        if i_split_randoms: my_logger.info(f"Split random {i_random+1} of {n_randoms}")
                        these_randoms = all_randoms[i_random] if i_split_randoms else vstack(all_randoms)
                        these_randoms1 = these_randoms[these_randoms["TRACERID"] == t1]
                        these_randoms2 = None if t1 == t2 else these_randoms[these_randoms["TRACERID"] == t2] # for auto-correlation, set the second tracer to None for proper handling in the TwoPointCorrelationFunction call. helper functions will propagate None into the positions, weights and samples
                        tmp = TwoPointCorrelationFunction(mode='smu', edges=edges,
                                                          data_positions1=get_rdd_positions(galaxies1), data_weights1=get_weights(galaxies1), data_samples1=get_samples(galaxies1),
                                                          data_positions2=get_rdd_positions(galaxies2), data_weights2=get_weights(galaxies2), data_samples2=get_samples(galaxies2),
                                                          randoms_positions1=get_rdd_positions(these_randoms1), randoms_weights1=get_weights(these_randoms1), randoms_samples1=get_samples(these_randoms1),
                                                          randoms_positions2=get_rdd_positions(these_randoms2), randoms_weights2=get_weights(these_randoms2), randoms_samples2=get_samples(these_randoms2),
                                                          position_type='rdd', engine='corrfunc', D1D2=D1D2, gpu=True, nthreads=4)
                        D1D2 = tmp.D1D2
                        result += tmp
                    results.append(result)
                corr = results[0].concatenate_x(*results)
                corr.D1D2.attrs['nsplits'] = n_randoms

                corr.save(output_file)