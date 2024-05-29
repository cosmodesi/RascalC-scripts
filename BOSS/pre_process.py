# This is a custom pre-processing script.
# Not computationally heavy so could be run on login nodes to save time allocation
# Currently wrapped into a function to be imported by the other scripts

import numpy as np
from astropy.table import Table
from RascalC.pre_process.convert_to_xyz import comoving_distance_Mpch
from RascalC.pre_process.create_jackknives_pycorr import get_subsampler_xirunpc

def get_rdd_positions(catalog: Table) -> tuple[np.ndarray[float]]: # utility function to format positions from a catalog
    return (catalog["RA"], catalog["DEC"], catalog["comov_dist"])

z_min, z_max = 0.43, 0.7

Omega_m = 0.3089
Omega_k = 0
w_DE = -1

n_jack = 60

data_filename = "/global/cfs/projectdirs/desi/mocks/Uchuu/SKIES_AND_UNIVERSE/Obs_data/CMASS_N_data.dat.h5"
random_filename = "/global/cfs/projectdirs/desi/mocks/Uchuu/SKIES_AND_UNIVERSE/Obs_data/CMASS_N_data.ran.h5"

def prepare_galaxy_random_catalogs() -> tuple[Table]:
    # read data
    galaxies = Table.read(data_filename, include_columns = ["RA", "DEC", "Z", "WEIGHT_SYSTOT", "WEIGHT_CP", "WEIGHT_NOZ", "WEIGHT_FKP"])
    galaxies["WEIGHT"] = galaxies["WEIGHT_SYSTOT"] * (galaxies["WEIGHT_CP"] + galaxies["WEIGHT_NOZ"] - 1) * galaxies["WEIGHT_FKP"]
    galaxies.remove_columns(["WEIGHT_SYSTOT", "WEIGHT_CP", "WEIGHT_NOZ", "WEIGHT_FKP"])
    galaxies = galaxies[np.logical_and(galaxies["Z"] >= z_min, galaxies["Z"] <= z_max)]

    # read randoms, handling weights differently
    randoms = Table.read(random_filename, include_columns = ["RA", "DEC", "Z", "WEIGHT_FKP"])
    randoms["WEIGHT"] = randoms["WEIGHT_FKP"]
    randoms.remove_column("WEIGHT_FKP")
    randoms = randoms[np.logical_and(randoms["Z"] >= z_min, randoms["Z"] <= z_max)]

    # compute comoving distance
    galaxies["comov_dist"] = comoving_distance_Mpch(galaxies["Z"], Omega_m, Omega_k, w_DE)
    randoms["comov_dist"] = comoving_distance_Mpch(randoms["Z"], Omega_m, Omega_k, w_DE)

    # assign jackknives
    subsampler = get_subsampler_xirunpc(get_rdd_positions(galaxies), n_jack, position_type = "rdd") # "rdd" means RA, DEC in degrees and then distance (corresponding to pycorr)
    galaxies["JACK"] = subsampler.label(get_rdd_positions(galaxies), position_type = "rdd")
    randoms["JACK"] = subsampler.label(get_rdd_positions(randoms), position_type = "rdd")

    return galaxies, randoms