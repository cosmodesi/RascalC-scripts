# This is the custom script to compute the 2PCF with jackknives

import os
import numpy as np
from tqdm import trange
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
    catalog["comov_dist"] = cosmology.comoving_radial_distance(catalog["Z"])
    if jack_sampler: catalog["JACK"] = jack_sampler.label(get_rdd_positions(catalog), position_type = 'rdd')
    return catalog


setup_logging()
cosmology = TabulatedDESI()

mock_id = 1

input_dir = f"/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/ALTMTL_EZmock/altmtl{mock_id}/mock{mock_id}/LSScats"
output_dir = f"{input_dir}/xi/smu/"

n_mu_bins = 200 # between -1 and 1
s_max = 200
split_above = 20
# tuple of edges, so that non-split randoms are below split_above and above they are split
all_edges = ((s_edges, np.linspace(-1, 1, n_mu_bins+1)) for s_edges in (np.arange(split_above+1), np.arange(split_above, s_max+1)))

n_jack = 60

for tracer, z_ranges in desi_y1_file_manager.list_zrange:
    if tracer.startswith("BGS"): continue # skip BGS
    if "+" in tracer: continue # skip the combined tracer (LRG+ELG_LOPnotqso)
    n_randoms = desi_y1_file_manager.list_nran[tracer]

    for (z_min, z_max) in z_ranges:
        for reg in ("SGC", "NGC"):
            output_file = f"{output_dir}/allcounts_{tracer}_{reg}_z{z_min}-{z_max}_default_FKP_lin_nran{n_randoms}_njack{n_jack}_split{split_above}.npy"
            if os.path.exists(output_file):
                print(f"Output file {output_file} exists, skipping")
                continue

            galaxies = prepare_catalog(f"{input_dir}/{tracer}_{reg}_clustering.dat.fits", z_min, z_max)
            jack_sampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack)
            galaxies["JACK"] = jack_sampler.label(get_rdd_positions(galaxies), position_type = 'rdd')

            all_randoms = [prepare_catalog(f"{input_dir}/{tracer}_{reg}_{i}_clustering.ran.fits", z_min, z_max, jack_sampler) for i in range(n_randoms)]

            results = []
            # compute
            for i_split_randoms, edges in enumerate(all_edges):
                result = 0
                D1D2 = None
                for i_random in trange(n_randoms if i_split_randoms else 1, desc = "Computing xi with random"):
                    these_randoms = all_randoms[i_random] if i_split_randoms else vstack(all_randoms)
                    tmp = TwoPointCorrelationFunction(mode = 'smu', edges = edges,
                                                    data_positions1 = get_rdd_positions(galaxies), data_weights1 = galaxies["WEIGHT"], data_samples1 = galaxies["JACK"],
                                                    randoms_positions1 = get_rdd_positions(these_randoms), randoms_weights1 = these_randoms["WEIGHT"], randoms_samples1 = these_randoms["JACK"],
                                                    position_type = 'rdd', engine = 'corrfunc', D1D2 = D1D2, gpu = True, nthreads = 4)
                    # position_type='pos' corresponds to (N, 3) x,y,z positions shape like we have here
                    D1D2 = tmp.D1D2
                    result += tmp
                results.append(result)
            corr = results[0].concatenate_x(*results)
            corr.D1D2.attrs['nsplits'] = n_randoms

            corr.save(output_file)