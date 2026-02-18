# This is the script to separate the combined tracer into the original tracers

import os
import sys
import numpy as np
import logging
from astropy.table import Table, vstack
import desi_y3_files.file_manager as desi_y3_file_manager
from pycorr import setup_logging


def catalog_cut_z_range(catalog: Table, z_min: float, z_max: float) -> Table: # utility function to cut a catalog to a redshift range
    filtering = np.logical_and(catalog["Z"] >= z_min, catalog["Z"] <= z_max) # logical index of redshifts within the range. Strict vs not strict inequality should not matter
    catalog = catalog[filtering] # filtered catalog
    catalog.remove_column("Z") # no longer needed, free memory
    return catalog


def process_catalog(filename: str, ref_catalog: Table, random: bool = False) -> None:
    catalog: Table = Table.read(filename)
    catalog.keep_columns(["TARGETID"] + ["TARGETID_DATA"] * random) # keep only TARGETID (and TARGETID_DATA for randoms) for checking the match with the combined catalog
    if not np.array_equal(catalog['TARGETID'], ref_catalog['TARGETID']): # check match with reference TARGETIDs to ensure correct matching of separate tracers with the combined tracer
        raise ValueError(f"TARGETIDs in {filename} do not match the reference TARGETIDs, can't proceed")
    if random and not np.array_equal(catalog['TARGETID_DATA'], ref_catalog['TARGETID_DATA']): # for randoms, also check the match of TARGETID_DATA
        raise ValueError(f"TARGETID_DATA in {filename} do not match the reference TARGETID_DATA, can't proceed")
    npz_filename = os.path.basename(filename).replace(".fits", ".npz")
    my_logger.info(f"Writing the TRACERIDs for {filename} to a compressed npz file {npz_filename}")
    np.savez_compressed(npz_filename, TRACERID=ref_catalog["TRACERID"]) # save TRACERID as a compressed npz file to be read later when running the correlation computations


setup_logging()
my_logger = logging.getLogger('label_tracers')

# Settings for filenames
verspec = 'loa-v1'
version = "v1.1"
conf = "BAO/unblinded"

# Set DESI CFS before creating the file manager
os.environ["DESICFS"] = "/dvs_ro/cfs/cdirs/desi" # read-only mount works faster, and we don't need to write

fm = desi_y3_file_manager.get_data_file_manager(conf, verspec)

separate_tracers = ['LRG', 'ELG_LOPnotqso'] # tracers to split the combined tracer into
combined_tracer = '+'.join(separate_tracers) # the combined tracer
corr_labels = [separate_tracers[0], "_".join(separate_tracers), separate_tracers[1]]

n_randoms = desi_y3_file_manager.list_nran[combined_tracer]
my_logger.info(f"Tracer: {combined_tracer}")

for reg in ("SGC", "NGC"):
    my_logger.info(f"Region: {reg}")
    
    common_setup = {"region": reg, "version": version, "grid_cosmo": None} # better specify it for pre-recon catalog selection
    recon_setup = desi_y3_file_manager.get_baseline_recon_setup(combined_tracer) # should not need to specify zrange here
    recon_setup.pop("zrange") # zrange causes problem with catalog selection

    my_logger.info("Reading data catalogs for separate tracers")

    data_refs = []
    for separate_tracer in separate_tracers:
        galaxy_files = [f.filepath for f in fm.select(id='catalog_data_y3', tracer=separate_tracer, **common_setup)] # pre-recon
        if (n := len(galaxy_files)) != 1:
            my_logger.error(f"Found not 1 but {n} galaxy files for {separate_tracer}, can't proceed")
            sys.exit(1)
        my_logger.info(f"Reading data catalog for {separate_tracer} from {galaxy_files[0]}")
        data_refs.append(Table.read(galaxy_files[0]))
        data_refs[-1].keep_columns(["Z", "TARGETID"]) # keep only Z for filtering and TARGETID for checking the match with the combined catalog
    data_ref = vstack(data_refs)
    data_ref["TRACERID"] = np.repeat(np.arange(len(separate_tracers)), [len(data_ref) for data_ref in data_refs]) # add TRACERID to keep track of which separate tracer each object belongs to
    del data_refs # no longer needed, free memory
    data_ref = catalog_cut_z_range(data_ref, *recon_setup["recon_zrange"]) # cut to the recon z range, this should match the object selection for the recon catalog

    galaxy_files = [f.filepath for f in fm.select(id='catalog_data_recon_y3', tracer=combined_tracer, **(common_setup | recon_setup))] # post-recon
    my_logger.debug({'id': 'catalog_data_recon_y3', 'tracer': combined_tracer} | common_setup | recon_setup)
    if (n := len(galaxy_files)) != 1:
        my_logger.warning(f"Found not 1 but {n} galaxy files for {combined_tracer}; skipping")
        continue
    my_logger.info(f"Reading and matching data catalog for {combined_tracer} from {galaxy_files[0]}")
    try: process_catalog(galaxy_files[0], data_ref, random=False)
    except Exception as e:
        my_logger.warning(f"Failed to process {combined_tracer} galaxy catalog: {e}. Skipping")
        continue
    del data_ref # no longer needed, free memory

    for i_random in range(n_randoms):
        random_refs = []
        for separate_tracer in separate_tracers:
            random_files = [f.filepath for f in fm.select(id='catalog_randoms_y3', tracer=separate_tracer, iran=i_random, **common_setup)] # pre-recon
            if (n := len(random_files)) != 1:
                my_logger.error(f"Found not 1 but {n} random files for {separate_tracer} #{i_random}, can't proceed")
                sys.exit(1)
            my_logger.info(f"Reading random catalog for {separate_tracer} #{i_random} from {random_files[0]}")
            random_refs.append(Table.read(random_files[0]))
            random_refs[-1].keep_columns(["Z", "TARGETID", "TARGETID_DATA"]) # keep only Z for filtering TARGETID (and TARGETID_DATA) for checking the match with the combined catalog
        random_ref = vstack(random_refs)
        random_ref["TRACERID"] = np.repeat(np.arange(len(separate_tracers)), [len(this_random_ref) for this_random_ref in random_refs]) # add TRACERID to keep track of which separate tracer each random object belongs to
        del random_refs
        random_ref = catalog_cut_z_range(random_ref, *recon_setup["recon_zrange"]) # cut to the recon z range, this should match the object selection for the recon catalog

        random_files = [f.filepath for f in fm.select(id='catalog_randoms_recon_y3', tracer=combined_tracer, iran=i_random, **(common_setup | recon_setup))] # post-recon
        if (n := len(random_files)) != 1:
            my_logger.warning(f"Found not 1 but {n} random files for {combined_tracer} #{i_random}; skipping")
            continue
        try: process_catalog(random_files[0], random_ref, random=False) # set/keep random=False because apparently the post-recon random catalogs do not have TARGETID_DATA, so it can not be checked. Checking TARGETID is probably enough to convince ourselves that the match is not a coincidence
        except Exception as e:
            my_logger.warning(f"Failed to process random catalog for {combined_tracer} #{i_random}: {e}. Skipping")
            continue
        del random_ref # no longer needed, free memory