### Python script for running on-the-fly reconstruction and saving shifted catalogs.
### Loops over all tracers and regions; output is consumed by run_covs.py.
import os
import numpy as np
from clustering_statistics.tools import read_clustering_catalog, propose_fiducial
from clustering_statistics.recon_tools import compute_reconstruction
from desipipe import setup_logging
from mpytools import Catalog
from warnings import filterwarnings

setup_logging()
filterwarnings("always")

version = 'data-dr3-matterhorn-v2-v0-bao'
outdir = os.path.join('recon_catalogs', version)
os.makedirs(outdir, exist_ok=True)

regs = ['SGC', 'NGC']
unique_tracers = ['BGS_BRIGHT-21.35', 'LRG', 'ELG_LOPnotqso', 'QSO']

for tracer in unique_tracers:
    recon_options = propose_fiducial('recon', tracer=tracer)
    recon_zrange = recon_options.pop('zrange')
    nran_recon = propose_fiducial('catalog', tracer=tracer)['nran']
    print(f"\n{tracer}: recon_zrange={recon_zrange}, nran={nran_recon}, options={recon_options}")

    for reg in regs:
        data_outfile = os.path.join(outdir, f"{tracer}_{reg}_data.npz")
        if os.path.isfile(data_outfile):
            print(f"  {reg}: {data_outfile} already exists, skipping")
            continue

        catalog_options = dict(version=version, tracer=tracer, region=reg, zrange=recon_zrange, nran=nran_recon, weight="default-FKP")
        catalog_options = propose_fiducial(kind='catalog', tracer=tracer, zrange=recon_zrange, analysis='full_shape') | catalog_options

        data_catalog = read_clustering_catalog(kind='data', **catalog_options)
        randoms_catalogs = read_clustering_catalog(kind='randoms', concatenate=False, **catalog_options)
        print(f"  {reg}: loaded data ({len(data_catalog)}) and {len(randoms_catalogs)} randoms over recon zrange {recon_zrange}")

        data_positions_rec, randoms_rec_positions = compute_reconstruction(
            lambda: {'data': data_catalog, 'randoms': Catalog.concatenate(randoms_catalogs)},
            **recon_options)
        print(f"  {reg}: reconstruction complete")

        np.savez(data_outfile,
                 position_rec=np.asarray(data_positions_rec),
                 z=np.asarray(data_catalog['Z']),
                 indweight=np.asarray(data_catalog['INDWEIGHT']))
        print(f"  {reg}: saved data to {data_outfile}")

        start = 0
        for iran, random in enumerate(randoms_catalogs):
            size = len(random['POSITION'])
            ran_outfile = os.path.join(outdir, f"{tracer}_{reg}_randoms_{iran}.npz")
            np.savez(ran_outfile,
                     position_rec=np.asarray(randoms_rec_positions[start:start + size]),
                     z=np.asarray(random['Z']),
                     indweight=np.asarray(random['INDWEIGHT']))
            start += size
        print(f"  {reg}: saved {len(randoms_catalogs)} random catalogs")

        del data_catalog, randoms_catalogs, data_positions_rec, randoms_rec_positions

print("\nAll reconstruction runs complete.")
