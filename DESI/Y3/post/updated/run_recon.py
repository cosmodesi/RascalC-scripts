### Python script for running on-the-fly reconstruction and saving shifted catalogs.
### Called per tracer; loops over regions. Output is consumed by run_covs.py.
import os
import numpy as np
from clustering_statistics.tools import read_clustering_catalog, propose_fiducial
from clustering_statistics.recon_tools import compute_reconstruction
from desipipe import setup_logging
from mpytools import Catalog
from warnings import filterwarnings
import argparse
import jax

setup_logging()
filterwarnings("always")

parser = argparse.ArgumentParser(description="Run reconstruction for a given tracer and save shifted catalogs")
parser.add_argument("--tracer", type=str, required=True, help="tracer name, e.g. LRG, ELG_LOPnotqso, QSO, BGS_BRIGHT-21.35")
args = parser.parse_args()

# Initialize JAX distributed BEFORE any catalog reading — cosmoprimo imports
# JAX at module level and creates arrays, which OOMs if all ranks target GPU 0.
jax.distributed.initialize()

version = 'data-dr2-v2'

tracer = args.tracer
regs = ['SGC', 'NGC']

recon_options = propose_fiducial('recon', tracer=tracer)
recon_zrange = recon_options.pop('zrange')
nran_recon = propose_fiducial('catalog', tracer=tracer)['nran']
print(f"{tracer}: recon_zrange={recon_zrange}, nran={nran_recon}, options={recon_options}")

recon_spec = 'recon_sm{smoothing_radius:.0f}_IFFT_{mode}'.format_map(recon_options)
outdir = os.path.join('catalogs', version, recon_spec)
if jax.process_index() == 0:
    os.makedirs(outdir, exist_ok=True)

for reg in regs:
    data_outfile = os.path.join(outdir, f"{tracer}_{reg}_clustering.dat.h5")
    if os.path.isfile(data_outfile):
        print(f"  {reg}: {data_outfile} already exists, skipping")
        continue

    catalog_options = dict(version=version, tracer=tracer, region=reg, zrange=recon_zrange, nran=nran_recon, weight="default-FKP")
    catalog_options = propose_fiducial(kind='catalog', tracer=tracer, zrange=recon_zrange, analysis='full_shape') | catalog_options

    data_catalog = read_clustering_catalog(kind='data', **catalog_options)
    randoms_catalogs = read_clustering_catalog(kind='randoms', concatenate=False, **catalog_options)
    print(f"  {reg}: loaded data ({len(data_catalog)}) and {len(randoms_catalogs)} randoms over recon zrange {recon_zrange}")

    data_catalog['POSITION_REC'], randoms_rec_positions = compute_reconstruction(
        lambda: {'data': data_catalog, 'randoms': Catalog.concatenate(randoms_catalogs)},
        **recon_options)
    print(f"  {reg}: reconstruction complete")

    data_catalog.write(data_outfile)
    print(f"  {reg}: saved data to {data_outfile}")

    # Assign reconstructed positions to random catalogs
    start = 0
    for iran, random in enumerate(randoms_catalogs):
        size = len(random['POSITION'])
        ran_outfile = os.path.join(outdir, f"{tracer}_{reg}_{iran}_clustering.ran.h5")
        random['POSITION_REC'] = randoms_rec_positions[start:start + size]
        random.write(ran_outfile)
        print(f"  {reg}: saved random catalog {iran} to {ran_outfile}")
        start += size

    del data_catalog, randoms_catalogs, randoms_rec_positions

jax.distributed.shutdown()
print(f"\n{tracer} reconstruction complete.")
