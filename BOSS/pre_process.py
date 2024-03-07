# This is a custom pre-processing script.
# Not computationally heavy so could be run on login nodes to save time allocation

import os
import numpy as np
import h5py
from RascalC.pre_process.convert_to_xyz import convert_to_xyz_files
from RascalC.pre_process.create_jackknives_pycorr import create_jackknives_pycorr_files

z_min, z_max = 0.43, 0.7

Omega_m = 0.3089
Omega_k = 0
w_dark_energy = -1

njack = 60

data_filename = "/global/cfs/projectdirs/desi/mocks/Uchuu/SKIES_AND_UNIVERSE/Obs_data/CMASS_N_data.dat.h5"
random_filename = "/global/cfs/projectdirs/desi/mocks/Uchuu/SKIES_AND_UNIVERSE/Obs_data/CMASS_N_data.ran.h5"
tmpdir = f"CMASS_N_data_{z_min}_{z_max}"

os.makedirs(tmpdir, exist_ok=1)

def change_extension(name: str, ext: str) -> str:
    return os.path.join(tmpdir, os.path.basename(".".join(name.split(".")[:-1] + [ext]))) # change extension and switch to tmpdir

def append_to_filename(name: str, appendage: str) -> str:
    return os.path.join(tmpdir, os.path.basename(name + appendage)) # append part and switch to tmpdir

# read data
with h5py.File(data_filename) as f:
    data_ra, data_dec, data_z, data_w_systot, data_w_cp, data_w_noz, data_w_fkp = (np.array(f[key]) for key in ("RA", "DEC", "Z", "WEIGHT_SYSTOT", "WEIGHT_CP", "WEIGHT_NOZ", "WEIGHT_FKP"))
data_w = data_w_systot * (data_w_cp + data_w_noz - 1) * data_w_fkp
data_filename_rdzw = change_extension(data_filename, "rdzw")
np.savetxt(data_filename_rdzw, np.array((data_ra, data_dec, data_z, data_w)).T[np.logical_and(data_z >= z_min, data_z <= z_max)])

# read randoms differently
with h5py.File(random_filename) as f:
    random_ra, random_dec, random_z, random_w = (np.array(f[key]) for key in ("RA", "DEC", "Z", "WEIGHT_FKP"))
random_filename = change_extension(random_filename, "rdzw")
np.savetxt(random_filename, np.array((random_ra, random_dec, random_z, random_w)).T[np.logical_and(random_z >= z_min, random_z <= z_max)])

# convert data to xyz
xyzw_filename = change_extension(data_filename_rdzw, "xyzw")
convert_to_xyz_files(data_filename_rdzw, xyzw_filename, Omega_m, Omega_k, w_dark_energy)
data_filename = xyzw_filename

# convert randoms to xyz
xyzw_filename = change_extension(random_filename, "xyzw")
convert_to_xyz_files(random_filename, xyzw_filename, Omega_m, Omega_k, w_dark_energy)
random_filename = xyzw_filename

# create jackknives for data
xyzwj_filename = change_extension(data_filename, "xyzwj")
create_jackknives_pycorr_files(data_filename_rdzw, data_filename, xyzwj_filename, njack)
data_filename = xyzwj_filename

# create jackknives for randoms
xyzwj_filename = change_extension(random_filename, "xyzwj")
create_jackknives_pycorr_files(data_filename_rdzw, random_filename, xyzwj_filename, njack)
random_filename = xyzwj_filename