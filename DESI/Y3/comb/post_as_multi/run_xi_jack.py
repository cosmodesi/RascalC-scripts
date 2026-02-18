# This is the custom script to compute the 2PCF with jackknives, using the pre-saved separation of the combined tracer into the original tracers (from label_tracers.py)

import os
import numpy as np
import logging
from astropy.table import Table, vstack
import desi_y3_files.file_manager as desi_y3_file_manager
from RascalC.pre_process import get_subsampler_xirunpc
from RascalC.utils import tracer1_corr, tracer2_corr
from pycorr import TwoPointCorrelationFunction, setup_logging, KMeansSubsampler
from LSS.tabulated_cosmo import TabulatedDESI


def get_rdd_positions(catalog: Table | None) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]] | None: # utility function to format positions from a catalog, handling the None case intended for the second tracer in auto-correlations
    if catalog is None: return None
    return (catalog["RA"], catalog["DEC"], catalog["comov_dist"])


def get_weights(catalog: Table | None) -> np.typing.NDArray[np.float64] | None: # utility function to format weights from a catalog, handling the None case intended for the second tracer in auto-correlations
    if catalog is None: return None
    return catalog["WEIGHT"]


def get_samples(catalog: Table | None) -> np.typing.NDArray[np.float64] | None: # utility function to format samples from a catalog, handling the None case intended for the second tracer in auto-correlations
    if catalog is None: return None
    return catalog["JACK"]


def prepare_catalog(filename: str, z_min: float = -np.inf, z_max: float = np.inf, jack_sampler: KMeansSubsampler | None = None, FKP_weight: bool = True, pre_recon: bool = False) -> Table:
    catalog: Table = Table.read(filename)
    if FKP_weight: catalog["WEIGHT"] *= catalog["WEIGHT_FKP"] # apply FKP weight multiplicatively
    catalog.keep_columns(["RA", "DEC", "Z", "WEIGHT"]) # discard everything else, including TARGETID (which is no longer needed)
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range
    catalog = catalog[filtering] # filtered catalog
    for key in catalog.keys():
        if catalog[key].dtype != float: # ensure all columns are float(64) for pycorr
            catalog[key] = catalog[key].astype(float)
    npz_filename = "../pre_as_multi/" * pre_recon + os.path.basename(filename).replace(".fits", ".npz") # load from pre-recon directory if set, otherwise from the current (post-recon) directory
    catalog["TRACERID"] = np.load(npz_filename)["TRACERID"][filtering] # load and add TRACERID to keep track of which separate tracer each object belongs to
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

target_tracer = 'LRG+ELG_LOPnotqso' # run the combined tracer
separate_tracers = ['LRG', 'ELG_LOPnotqso'] # tracers to split the combined tracer into
corr_labels = [separate_tracers[0], "_".join(separate_tracers), separate_tracers[1]]

for tracer, z_ranges in desi_y3_file_manager.list_zrange.items():
    if tracer != target_tracer: continue
    n_randoms = desi_y3_file_manager.list_nran[tracer]
    my_logger.info(f"Tracer: {tracer}")

    for reg in ("SGC", "NGC"):
        my_logger.info(f"Region: {reg}")
        
        for z_range in z_ranges:
            z_min, z_max = z_range
            my_logger.info(f"Redshift range: {z_min}-{z_max}")
            common_setup = {"tracer": tracer, "region": reg, "version": version, "grid_cosmo": None} # grid_cosmo provided for unshifted randoms
            recon_setup = desi_y3_file_manager.get_baseline_recon_setup(tracer, z_range)
            recon_setup.pop("zrange")
            xi_setup = desi_y3_file_manager.get_baseline_2pt_setup(tracer, z_range, recon=True)
            xi_setup.update({"zrange": z_range, "cut": None, "njack": n_jack})

            default_output_files = [f.filepath for f in fm.select(id = 'correlation_recon_y3', **(common_setup | xi_setup))]
            if (n := len(default_output_files)) != 1:
                my_logger.warning(f"Found not 1 but {n} jackknife counts files; skipping")
                continue
            default_output_file = default_output_files[0]

            galaxy_files = [f.filepath for f in fm.select(id = 'catalog_data_recon_y3', **(common_setup | recon_setup))]
            if (n := len(galaxy_files)) != 1:
                my_logger.warning(f"Found not 1 but {n} galaxy files; skipping")
                continue
            try: galaxies = prepare_catalog(galaxy_files[0], z_min, z_max)
            except Exception as e:
                my_logger.warning(f"Failed to prepare galaxy catalog: {e}. Skipping")
                continue
            jack_sampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack)
            galaxies["JACK"] = jack_sampler.label(get_rdd_positions(galaxies), position_type = 'rdd')

            shifted_random_files = [f.filepath for f in fm.select(id = 'catalog_randoms_recon_y3', iran = range(n_randoms), **(common_setup | recon_setup))] # shifted (post-recon) randoms
            if (n := len(shifted_random_files)) != n_randoms:
                my_logger.warning(f"Found not {n_randoms} but {n} shifted random files; skipping")
                continue
            try: all_shifted_randoms = [prepare_catalog(shifted_random_file, z_min, z_max, jack_sampler) for shifted_random_file in shifted_random_files]
            except Exception as e:
                my_logger.warning(f"Failed to prepare shifted random catalogs: {e}. Skipping")
                continue

            random_files = [f.filepath for f in fm.select(id = 'catalog_randoms_y3', iran = range(n_randoms), **common_setup)] # unshifted (pre-recon) randoms
            if (n := len(random_files)) != n_randoms:
                my_logger.warning(f"Found not {n_randoms} but {n} random files; skipping")
                continue
            try: all_randoms = [prepare_catalog(random_file, z_min, z_max, jack_sampler, pre_recon=True) for random_file in random_files]
            except Exception as e:
                my_logger.warning(f"Failed to prepare random catalogs: {e}. Skipping")
                continue

            for t1, t2, corr_label in zip(tracer1_corr, tracer2_corr, corr_labels):
                my_logger.info(f"Computing {corr_label.replace('_', ' x ')} {'auto' if t1 == t2 else 'cross'}-correlation")

                output_file = f"{output_dir}/{os.path.basename(default_output_file)}".replace(target_tracer, corr_label) # switch to the output dir
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
                        these_shifted_randoms = all_shifted_randoms[i_random] if i_split_randoms else vstack(all_shifted_randoms)
                        these_shifted_randoms1 = these_shifted_randoms[these_shifted_randoms["TRACERID"] == t1]
                        these_shifted_randoms2 = None if t1 == t2 else these_shifted_randoms[these_shifted_randoms["TRACERID"] == t2] # for auto-correlation, set the second tracer to None for proper handling in the TwoPointCorrelationFunction call. helper functions will propagate None into the positions, weights and samples
                        these_randoms = all_randoms[i_random] if i_split_randoms else vstack(all_randoms)
                        these_randoms1 = these_randoms[these_randoms["TRACERID"] == t1]
                        these_randoms2 = None if t1 == t2 else these_randoms[these_randoms["TRACERID"] == t2] # for auto-correlation, set the second tracer to None for proper handling in the TwoPointCorrelationFunction call. helper functions will propagate None into the positions, weights and samples
                        tmp = TwoPointCorrelationFunction(mode = 'smu', edges = edges,
                                                          data_positions1 = get_rdd_positions(galaxies1), data_weights1 = get_weights(galaxies1), data_samples1 = get_samples(galaxies1),
                                                          data_positions2 = get_rdd_positions(galaxies2), data_weights2 = get_weights(galaxies2), data_samples2 = get_samples(galaxies2),
                                                          shifted_positions1 = get_rdd_positions(these_shifted_randoms1), shifted_weights1 = get_weights(these_shifted_randoms1), shifted_samples1 = get_samples(these_shifted_randoms1),
                                                          shifted_positions2 = get_rdd_positions(these_shifted_randoms2), shifted_weights2 = get_weights(these_shifted_randoms2), shifted_samples2 = get_samples(these_shifted_randoms2),
                                                          randoms_positions1 = get_rdd_positions(these_randoms1), randoms_weights1 = get_weights(these_randoms1), randoms_samples1 = get_samples(these_randoms1),
                                                          randoms_positions2 = get_rdd_positions(these_randoms2), randoms_weights2 = get_weights(these_randoms2), randoms_samples2 = get_samples(these_randoms2),
                                                          position_type = 'rdd', engine = 'corrfunc', D1D2 = D1D2, gpu = True, nthreads = 4)
                        D1D2 = tmp.D1D2
                        result += tmp
                    results.append(result)
                corr = results[0].concatenate_x(*results)
                corr.D1D2.attrs['nsplits'] = n_randoms

                corr.save(output_file)