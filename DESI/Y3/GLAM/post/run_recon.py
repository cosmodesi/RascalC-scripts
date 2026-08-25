### Python script for running on-the-fly reconstruction and saving shifted catalogs (GLAM mocks).
### Adapted from DESI/Y5/post/run_recon.py: mocks need imock and parent_randoms expansion
### (as in DESI/Y3/GLAM/pre/run_covs.py and DESI/Y3/Uchuu/post/run_covs.py), which real Y5 data does not.
### Called per tracer; loops over regions. Output is consumed by run_covs.py.
import os
import numpy as np
from clustering_statistics.tools import get_catalog_fn, read_clustering_catalog, propose_fiducial
from clustering_statistics.recon_tools import compute_reconstruction
from desipipe import setup_logging
from mpytools import Catalog
from warnings import filterwarnings
import argparse
import jax

setup_logging()
filterwarnings("always")

parser = argparse.ArgumentParser(description="Run reconstruction for a given tracer and save shifted catalogs (GLAM mocks)")
parser.add_argument("--tracer", type=str, required=True, help="tracer name, e.g. LRG, ELG_LOPnotqso, QSO, BGS_BRIGHT-21.35")
args = parser.parse_args()

# Initialize JAX distributed BEFORE any catalog reading — cosmoprimo imports
# JAX at module level and creates arrays, which OOMs if all ranks target GPU 0.
jax.distributed.initialize()

version_dark = 'glam-uchuu-v2-altmtl'
version_bright = 'glam-uchuu-bgs-v2-altmtl'
mock_id = 150

tracer = args.tracer
version = version_bright if tracer.startswith('BGS') else version_dark

outdir = os.path.join(os.environ['SCRATCH'], 'rascalc', 'recon_catalogs', version, f"mock{mock_id}")
if jax.process_index() == 0:
    os.makedirs(outdir, exist_ok=True)

regs = ['SGC', 'NGC']

recon_options = propose_fiducial('recon', tracer=tracer)
recon_zrange = recon_options.pop('zrange')
nran_recon = propose_fiducial('catalog', tracer=tracer)['nran']
print(f"{tracer}: recon_zrange={recon_zrange}, nran={nran_recon}, options={recon_options}")

for reg in regs:
    data_outfile = os.path.join(outdir, f"{tracer}_{reg}_data.h5")
    if os.path.isfile(data_outfile):
        print(f"  {reg}: {data_outfile} already exists, skipping")
        continue

    catalog_options = dict(version=version, imock=mock_id, tracer=tracer, region=reg, zrange=recon_zrange, nran=nran_recon, weight="default-FKP")
    catalog_options = propose_fiducial(kind='catalog', tracer=tracer, zrange=recon_zrange, analysis='full_shape') | catalog_options
    expand = {'parent_randoms_fn': get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=nran_recon)}

    data_catalog = read_clustering_catalog(kind='data', **catalog_options)
    randoms_catalogs = read_clustering_catalog(kind='randoms', concatenate=False, expand=expand, **catalog_options)
    print(f"  {reg}: loaded data ({len(data_catalog)}) and {len(randoms_catalogs)} randoms over recon zrange {recon_zrange}")

    data_positions_rec, randoms_rec_positions = compute_reconstruction(
        lambda: {'data': data_catalog, 'randoms': Catalog.concatenate(randoms_catalogs)},
        **recon_options)
    print(f"  {reg}: reconstruction complete")

    data_catalog['Position'] = np.asarray(data_positions_rec)
    data_catalog.write(data_outfile)

    start = 0
    for iran, random in enumerate(randoms_catalogs):
        size = len(random['POSITION'])
        ran_outfile = os.path.join(outdir, f"{tracer}_{reg}_randoms_{iran}.h5")
        random['Position'] = np.asarray(randoms_rec_positions[start:start + size])
        random.write(ran_outfile)
        start += size

    if jax.process_index() == 0:
        print(f"  {reg}: saved data and {len(randoms_catalogs)} random catalogs")

    del data_catalog, randoms_catalogs, data_positions_rec, randoms_rec_positions

jax.distributed.shutdown()
print(f"\n{tracer} reconstruction complete.")
