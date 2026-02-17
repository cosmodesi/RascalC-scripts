# This is the custom script to compute the 2PCF with jackknives

import os
import sys
import numpy as np
import logging
from astropy.table import Table, vstack
import desi_y3_files.file_manager as desi_y3_file_manager
from RascalC.pre_process import get_subsampler_xirunpc
from RascalC.utils import tracer1_corr, tracer2_corr
from pycorr import TwoPointCorrelationFunction, setup_logging, KMeansSubsampler
from LSS.tabulated_cosmo import TabulatedDESI


def get_rdd_positions(catalog: Table) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]: # utility function to format positions from a catalog
    return (catalog["RA"], catalog["DEC"], catalog["comov_dist"])


def prepare_catalog(filename: str, z_min: float = -np.inf, z_max: float = np.inf, jack_sampler: KMeansSubsampler | None = None, FKP_weight: bool = True) -> Table:
    catalog: Table = Table.read(filename)
    if FKP_weight: catalog["WEIGHT"] *= catalog["WEIGHT_FKP"] # apply FKP weight multiplicatively
    catalog.keep_columns(["RA", "DEC", "Z", "WEIGHT", "TARGETID" + "_DATA" * bool(jack_sampler)]) # discard everything else; need TARGETID for data and TARGETID_DATA for randoms to separate the combined tracer into the original tracers
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range
    catalog = catalog[filtering] # filtered catalog
    for key in catalog.keys():
        if catalog[key].dtype != float:
            catalog[key] = catalog[key].astype(float) # ensure everything is float(64)
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

        tracer_TARGETIDs = {}
        for separate_tracer in separate_tracers:
            common_setup = {"tracer": separate_tracer, "region": reg, "version": version, "grid_cosmo": None}
            galaxy_files = [f.filepath for f in fm.select(id = 'catalog_data_y3', **common_setup)]
            if (n := len(galaxy_files)) != 1:
                my_logger.error(f"Found not 1 but {n} galaxy files, can't proceed")
                sys.exit(1)
            tracer_TARGETIDs[separate_tracer] = Table.read(galaxy_files[0])["TARGETID"] # save TARGETIDs for later use
            _, counts = np.unique(tracer_TARGETIDs[separate_tracer], return_counts=True)
            if len(_) != len(tracer_TARGETIDs[separate_tracer]):
                my_logger.warning(f"{galaxy_files[0]} has " + ', '.join(f"{n_count} TARGETID(s) appearing {count} time(s)" for count, n_count in zip(*np.unique(counts, return_counts=True))))
        
        for z_range in z_ranges:
            z_min, z_max = z_range
            my_logger.info(f"Redshift range: {z_min}-{z_max}")
            common_setup = {"tracer": tracer, "region": reg, "version": version, "grid_cosmo": None}
            xi_setup = desi_y3_file_manager.get_baseline_2pt_setup(tracer, z_range)
            xi_setup.update({"zrange": z_range, "cut": None, "njack": n_jack})

            default_output_files = [f.filepath for f in fm.select(id = 'correlation_y3', **(common_setup | xi_setup))]
            if (n := len(default_output_files)) != 1:
                my_logger.warning(f"Found not 1 but {n} jackknife counts files; skipping")
                continue
            default_output_file = default_output_files[0]

            galaxy_files = [f.filepath for f in fm.select(id = 'catalog_data_y3', **common_setup)]
            if (n := len(galaxy_files)) != 1:
                my_logger.warning(f"Found not 1 but {n} galaxy files; skipping")
                continue
            galaxies = prepare_catalog(galaxy_files[0], z_min, z_max)
            jack_sampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack)
            galaxies["JACK"] = jack_sampler.label(get_rdd_positions(galaxies), position_type = 'rdd')

            galaxy_masks = [np.isin(galaxies["TARGETID"], tracer_TARGETIDs[separate_tracer]) for separate_tracer in separate_tracers]
            if not np.array_equal(np.sum(galaxy_masks, axis=0), np.ones(len(galaxies))):
                _, counts = np.unique(galaxies["TARGETID"], return_counts=True)
                my_logger.warning(f"{galaxy_files[0]} has " + ', '.join(f"{n_count} TARGETID(s) appearing {count} time(s)" for count, n_count in zip(*np.unique(counts, return_counts=True))) + f" for {z_min}<z<{z_max}")
                my_logger.warning("The combined tracer catalog contains objects that are not in exactly one of the separate tracers. Specifically, " + ", ".join(f"{n_count} object(s) appearing {count} time(s)" for count, n_count in zip(*np.unique(np.sum(galaxy_masks, axis=0), return_counts=True))) + ". Can't proceed")
                continue

            random_files = [f.filepath for f in fm.select(id = 'catalog_randoms_y3', iran = range(n_randoms), **common_setup)]
            if (n := len(random_files)) != n_randoms:
                my_logger.warning(f"Found not {n_randoms} but {n} random files; skipping")
                continue
            all_randoms = [prepare_catalog(random_file, z_min, z_max, jack_sampler) for random_file in random_files]
            all_random_masks = [[np.isin(randoms["TARGETID_DATA"], tracer_TARGETIDs[separate_tracer]) for randoms in all_randoms] for separate_tracer in separate_tracers]
            if not all(np.array_equal(np.sum([all_random_masks[t][i_random] for t in range(len(separate_tracers))], axis=0), np.ones(len(all_randoms[i_random]))) for i_random in range(n_randoms)):
                my_logger.warning("Combined tracer random catalogs contain objects that are not in exactly one of the separate tracers, can't proceed")
                continue

            for t1, t2, corr_label in zip(tracer1_corr, tracer2_corr, corr_labels):
                my_logger.info(f"Computing correlation {corr_label}")

                output_file = f"{output_dir}/{os.path.basename(default_output_file)}".replace(target_tracer, corr_label) # switch to the output dir
                if os.path.exists(output_file):
                    my_logger.info(f"Output file {output_file} exists, skipping")
                    continue

                galaxy_mask1 = galaxy_masks[t1]
                galaxy_mask2 = galaxy_masks[t2]
                all_randoms_mask1 = all_random_masks[t1]
                all_randoms_mask2 = all_random_masks[t2]

                results = []
                # compute
                for i_split_randoms, edges in enumerate(all_edges):
                    my_logger.info("Using " + ("split" if i_split_randoms else "concatenated") + " randoms")
                    result = 0
                    D1D2 = None
                    for i_random in range(n_randoms if i_split_randoms else 1):
                        if i_split_randoms: my_logger.info(f"Split random {i_random+1} of {n_randoms}")
                        these_randoms = all_randoms[i_random] if i_split_randoms else vstack(all_randoms)
                        these_randoms_mask1 = all_randoms_mask1[i_random] if i_split_randoms else np.concatenate(all_randoms_mask1)
                        these_randoms_mask2 = all_randoms_mask2[i_random] if i_split_randoms else np.concatenate(all_randoms_mask2)
                        tmp = TwoPointCorrelationFunction(mode = 'smu', edges = edges,
                                                          data_positions1 = get_rdd_positions(galaxies[galaxy_mask1]), data_weights1 = galaxies["WEIGHT"][galaxy_mask1], data_samples1 = galaxies["JACK"][galaxy_mask1],
                                                          data_positions2 = get_rdd_positions(galaxies[galaxy_mask2]) if t1 != t2 else None, data_weights2 = galaxies["WEIGHT"][galaxy_mask2] if t1 != t2 else None, data_samples2 = galaxies["JACK"][galaxy_mask2] if t1 != t2 else None,
                                                          randoms_positions1 = get_rdd_positions(these_randoms[these_randoms_mask1]), randoms_weights1 = these_randoms["WEIGHT"][these_randoms_mask1], randoms_samples1 = these_randoms["JACK"][these_randoms_mask1],
                                                          randoms_positions2 = get_rdd_positions(these_randoms[these_randoms_mask2]) if t1 != t2 else None, randoms_weights2 = these_randoms["WEIGHT"][these_randoms_mask2] if t1 != t2 else None, randoms_samples2 = these_randoms["JACK"][these_randoms_mask2] if t1 != t2 else None,
                                                          position_type = 'rdd', engine = 'corrfunc', D1D2 = D1D2, gpu = True, nthreads = 4)
                        D1D2 = tmp.D1D2
                        result += tmp
                    results.append(result)
                corr = results[0].concatenate_x(*results)
                corr.D1D2.attrs['nsplits'] = n_randoms

                corr.save(output_file)