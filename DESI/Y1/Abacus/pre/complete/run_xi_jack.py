# This is the custom script to compute the 2PCF with jackknives

import os
import numpy as np
import logging
from astropy.table import Table, vstack
import desi_y1_files.file_manager as desi_y1_file_manager
from RascalC.pre_process import get_subsampler_xirunpc
from pycorr import TwoPointCorrelationFunction, setup_logging, KMeansSubsampler
from LSS.tabulated_cosmo import TabulatedDESI


def get_rdd_positions(catalog: Table) -> tuple[np.ndarray[float]]: # utility function to format positions from a catalog
    return (catalog["RA"], catalog["DEC"], catalog["comov_dist"])


def prepare_catalog(filename: str, z_min: float = -np.inf, z_max: float = np.inf, jack_sampler: KMeansSubsampler | None = None, FKP_weight: bool = True):
    catalog = Table.read(filename)
    if FKP_weight: catalog["WEIGHT"] *= catalog["WEIGHT_FKP"] # apply FKP weight multiplicatively
    catalog.keep_columns(["RA", "DEC", "Z", "WEIGHT"]) # discard everything else
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range
    catalog = catalog[filtering] # filtered catalog
    for key in catalog.keys():
        if catalog[key].dtype != float:
            catalog[key] = catalog[key].astype(float) # ensure everything is float(64)
    catalog["comov_dist"] = cosmology.comoving_radial_distance(catalog["Z"])
    if jack_sampler: catalog["JACK"] = jack_sampler.label(get_rdd_positions(catalog), position_type = 'rdd')
    return catalog


setup_logging()
my_logger = logging.getLogger()
cosmology = TabulatedDESI()

version = "v4_2"
fa = "complete" # fiber assignment method

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only mount works faster, and we don't need to write

fm = desi_y1_file_manager.get_abacus_file_manager()

mock_id = 0

output_dir = "xi/smu" # here in scratch for faster write, to be moved later
os.makedirs(output_dir, exist_ok = True)

n_mu_bins = 200 # between -1 and 1
s_max = 200
split_above = 20
# list of edge configurations, so that non-split randoms are below split_above and above they are split
all_edges = [(s_edges, np.linspace(-1, 1, n_mu_bins+1)) for s_edges in (np.arange(split_above+1), np.arange(split_above, s_max+1))]

n_jack = 60

for tracer, z_ranges in desi_y1_file_manager.list_zrange.items():
    if tracer not in ("LRG", "ELG_LOPnotqso"): continue # skip other tracers
    n_randoms = desi_y1_file_manager.list_nran[tracer]
    my_logger.info(f"Tracer: {tracer}")

    for z_range in z_ranges[-1:]:
        z_min, z_max = z_range
        my_logger.info(f"Redshift range: {z_min}-{z_max}")
        for reg in ("SGC", "NGC"):
            my_logger.info(f"Region: {reg}")
            common_setup = {"tracer": tracer, "region": reg, "version": version, "fa": fa, "imock": mock_id}
            xi_setup = desi_y1_file_manager.get_baseline_2pt_setup(tracer, z_range)
            xi_setup.update({"zrange": z_range, "cut": None, "njack": n_jack})

            output_files = [f.filepath for f in fm.select(id = 'correlation_abacus_y1', **common_setup, **xi_setup)]
            if (n := len(output_files)) != 1:
                my_logger.info(f"Found not 1 but {n} jackknife counts files; skipping")
                continue
            output_file = output_files[0]
            if os.path.exists(output_file):
                my_logger.info(f"Output file {output_file} exists, skipping")
                continue
            output_file = f"{output_dir}/{os.path.basename(output_file)}" # switch to the output dir
            if os.path.exists(output_file):
                my_logger.info(f"Output file {output_file} exists, skipping")
                continue

            galaxy_files = [f.filepath for f in fm.select(id = 'catalog_data_abacus_y1', **common_setup)]
            if (n := len(galaxy_files)) != 1:
                my_logger.info(f"Found not 1 but {n} galaxy files; skipping")
                continue
            galaxies = prepare_catalog(galaxy_files[0], z_min, z_max)
            jack_sampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack)
            galaxies["JACK"] = jack_sampler.label(get_rdd_positions(galaxies), position_type = 'rdd')

            random_files = [f.filepath for f in fm.select(id = 'catalog_randoms_abacus_y1', iran = range(n_randoms), **common_setup)]
            if (n := len(random_files)) != n_randoms:
                my_logger.info(f"Found not {n_randoms} but {n} random files; skipping")
                continue
            all_randoms = [prepare_catalog(random_file, z_min, z_max, jack_sampler) for random_file in random_files]

            results = []
            # compute
            for i_split_randoms, edges in enumerate(all_edges):
                my_logger.info("Using " + ("split" if i_split_randoms else "concatenated") + " randoms")
                result = 0
                D1D2 = None
                for i_random in range(n_randoms if i_split_randoms else 1):
                    if i_split_randoms: my_logger.info(f"Split random {i_random+1} of {n_randoms}")
                    these_randoms = all_randoms[i_random] if i_split_randoms else vstack(all_randoms)
                    tmp = TwoPointCorrelationFunction(mode = 'smu', edges = edges,
                                                    data_positions1 = get_rdd_positions(galaxies), data_weights1 = galaxies["WEIGHT"], data_samples1 = galaxies["JACK"],
                                                    randoms_positions1 = get_rdd_positions(these_randoms), randoms_weights1 = these_randoms["WEIGHT"], randoms_samples1 = these_randoms["JACK"],
                                                    position_type = 'rdd', engine = 'corrfunc', D1D2 = D1D2, gpu = True, nthreads = 4)
                    D1D2 = tmp.D1D2
                    result += tmp
                results.append(result)
            corr = results[0].concatenate_x(*results)
            corr.D1D2.attrs['nsplits'] = n_randoms

            corr.save(output_file)